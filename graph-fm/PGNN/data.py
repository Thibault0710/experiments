from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.spatial import Delaunay
from torch.utils.data import Dataset
from torch_geometric.data import Data


def bilinear_sample(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sample img at continuous (x, y) points using bilinear interpolation.

    img: (H, W) for scalar or (H, W, C) for multi-channel.
    points: (N, 2) with points[:, 0] = x (column), points[:, 1] = y (row).
    Returns: (N,) float32 for scalar img, (N, C) for multi-channel.
    """
    if img.ndim == 2:
        H, W = img.shape
    elif img.ndim == 3:
        H, W, _ = img.shape
    else:
        raise ValueError(f"img must be 2D or 3D, got shape {img.shape}")
    x = np.clip(points[:, 0], 0, W - 1 - 1e-6)
    y = np.clip(points[:, 1], 0, H - 1 - 1e-6)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    if img.ndim == 3:
        wx = wx[:, None]
        wy = wy[:, None]
    val = (
        img[y0, x0] * (1 - wx) * (1 - wy)
        + img[y0, x1] * wx * (1 - wy)
        + img[y1, x0] * (1 - wx) * wy
        + img[y1, x1] * wx * wy
    )
    return val.astype(np.float32)


def build_mesh(
    n_interior: int,
    image_size: int,
    rng: np.random.Generator,
    upper_bias: float = 1.0,
    right_bias: float = 1.0,
):
    """Sample n_interior points in [0, image_size]^2 and Delaunay triangulate.

    No corner pinning: the mesh covers only the convex hull of the sampled
    points, so visualizations show genuine off-mesh regions near the corners.

    upper_bias: y-axis density ratio (upper half over lower half). 1.0 = uniform,
        2.0 = top is twice as dense.
    right_bias: x-axis density ratio (right half over left half). 1.0 = uniform.

    With either bias != 1.0, sampling is a hard quadrant split with point
    counts proportional to the product of axis weights. Both biases at 1.0
    falls back to a single uniform sample (reproduces pre-bias behavior).

    Returns:
        points: (N, 2) float64, x=col, y=row.
        edge_index: (2, E) int64, bidirectional, no self-loops.
    """
    if upper_bias == 1.0 and right_bias == 1.0:
        points = rng.uniform(0, image_size, size=(n_interior, 2))
    else:
        p_upper = upper_bias / (upper_bias + 1.0)
        p_right = right_bias / (right_bias + 1.0)
        # quadrants ordered: TL, TR, BL, BR
        weights = np.array(
            [
                (1 - p_right) * p_upper,
                p_right * p_upper,
                (1 - p_right) * (1 - p_upper),
                p_right * (1 - p_upper),
            ]
        )
        counts = np.floor(weights * n_interior).astype(int)
        counts[np.argmax(weights)] += n_interior - counts.sum()
        half = image_size / 2.0
        ranges = [
            ((0.0, half), (0.0, half)),
            ((half, image_size), (0.0, half)),
            ((0.0, half), (half, image_size)),
            ((half, image_size), (half, image_size)),
        ]
        parts = []
        for c, ((xl, xh), (yl, yh)) in zip(counts, ranges):
            if c == 0:
                continue
            parts.append(
                np.column_stack([rng.uniform(xl, xh, c), rng.uniform(yl, yh, c)])
            )
        points = np.vstack(parts)
    tri = Delaunay(points)
    edge_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
            if a == b:
                continue
            edge_set.add((min(a, b), max(a, b)))
    faces = np.asarray(tri.simplices, dtype=np.int64)
    if not edge_set:
        return points, np.zeros((2, 0), dtype=np.int64), faces
    edges = np.array(sorted(edge_set), dtype=np.int64)
    both = np.vstack([edges, edges[:, ::-1]])
    return points, both.T.copy(), faces


def build_grid_mesh(image_size: int, n_per_side: int):
    """Fixed n_per_side × n_per_side regular mesh spanning [0, image_size-1].

    Each square cell split by one diagonal (top-left → bottom-right) into 2
    triangles. Same return shape as `build_mesh`: (points, edge_index, faces).
    Convex hull is the full image (corners included), so Gouraud-render covers
    every pixel.
    """
    N = n_per_side
    side = np.linspace(0, image_size - 1, N, dtype=np.float64)
    points = np.array([(side[j], side[i]) for i in range(N) for j in range(N)],
                      dtype=np.float64)
    edges = []
    faces = []
    for i in range(N):
        for j in range(N):
            v = i * N + j
            if j + 1 < N:
                edges.append((v, v + 1))
            if i + 1 < N:
                edges.append((v, v + N))
            if i + 1 < N and j + 1 < N:
                v_diag = v + N + 1
                edges.append((v, v_diag))
                v_right = v + 1
                v_down  = v + N
                faces.append((v, v_right, v_diag))
                faces.append((v, v_diag,  v_down))
    edges = np.array(edges, dtype=np.int64)
    both = np.vstack([edges, edges[:, ::-1]])
    edge_index = both.T.copy()
    faces = np.array(faces, dtype=np.int64)
    return points, edge_index, faces


def compute_laplacian_pe(
    edge_index: np.ndarray,
    num_nodes: int,
    k: int,
    points: np.ndarray | None = None,
    anchor_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """k smallest non-trivial eigvecs of the symmetric normalized Laplacian.

    Returns (num_nodes, k) float32. Each column is rescaled so |max| = 1.
    Sign convention: if `points` is given, for each eigvec we compare the two
    extrema — argmax (would-be +1 node) and argmin (would-be -1 node). We flip
    the column's sign so that whichever extremum is closer to `anchor_xy`
    becomes the positive one. This is robust to eigvecs whose value at a
    fixed anchor node happens to lie near zero. Falls back to peak-magnitude
    sign fixing when `points` is None.
    """
    from scipy.sparse import coo_matrix, eye as sp_eye
    from scipy.sparse.linalg import eigsh

    n = num_nodes
    src, dst = edge_index[0], edge_index[1]
    data = np.ones(len(src), dtype=np.float32)
    A = coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(axis=1)).flatten()
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0).astype(np.float32)
    D_inv_sqrt = coo_matrix((deg_inv_sqrt, (np.arange(n), np.arange(n))), shape=(n, n)).tocsr()
    L = sp_eye(n, dtype=np.float32, format="csr") - (D_inv_sqrt @ A @ D_inv_sqrt)
    take = min(k, n - 1)
    eigvals, eigvecs = eigsh(L, k=take + 1, which="SM")
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    pe = eigvecs[:, 1 : 1 + take].astype(np.float32)
    anchor = np.asarray(anchor_xy, dtype=np.float32)
    for j in range(pe.shape[1]):
        col = pe[:, j]
        if points is not None:
            node_pos = int(np.argmax(col))
            node_neg = int(np.argmin(col))
            d_pos = np.sum((points[node_pos] - anchor) ** 2)
            d_neg = np.sum((points[node_neg] - anchor) ** 2)
            if d_neg < d_pos:
                col = -col
        else:
            peak = int(np.argmax(np.abs(col)))
            if col[peak] < 0:
                col = -col
        max_abs = np.abs(col).max()
        if max_abs > 0:
            col = col / max_abs
        pe[:, j] = col
    if take < k:
        pad = np.zeros((n, k - take), dtype=np.float32)
        pe = np.concatenate([pe, pad], axis=1)
    return pe


def compute_harmonic_pe(
    edge_index: np.ndarray,
    points: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """One scalar per node: the discrete harmonic function with u=1 at the
    node closest to (0, 0) (north pole) and u=0 at the node closest to
    (image_size-1, image_size-1) (south pole), and (D-A) u = 0 everywhere
    else (graph Laplace equation). By the maximum principle u ∈ [0, 1].

    Returns (N, 1) float32. Unlike LapPE this has no sign ambiguity — the
    boundary conditions pin orientation.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve

    n = points.shape[0]
    src, dst = edge_index[0], edge_index[1]
    data = np.ones(len(src), dtype=np.float64)
    A = coo_matrix((data, (src, dst)), shape=(n, n)).tocsr()
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(axis=1)).flatten()
    L = diags(deg) - A

    far = float(image_size - 1)
    d_north = np.sum(points ** 2, axis=1)
    d_south = np.sum((points - np.array([far, far])) ** 2, axis=1)
    north = int(np.argmin(d_north))
    south = int(np.argmin(d_south))

    boundary_mask = np.zeros(n, dtype=bool)
    boundary_mask[north] = True
    boundary_mask[south] = True
    interior = np.where(~boundary_mask)[0]

    L_II = L[interior][:, interior].tocsc()
    L_IN = np.asarray(L[interior, north].todense()).flatten()
    # u_south = 0 so the south-column contribution vanishes; only north matters.
    b = -L_IN  # since u_north = 1.0

    u_I = spsolve(L_II, b)

    u = np.zeros(n, dtype=np.float32)
    u[north] = 1.0
    u[south] = 0.0
    u[interior] = u_I.astype(np.float32)
    return u.reshape(-1, 1)


class MNISTMeshDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        min_vertices: int = 400,
        max_vertices: int = 700,
        image_size: int = 28,
        seed: int | None = None,
        vertex_bias: float = 1.0,
        pe_type: str = "fourier",
        pe_dim: int = 16,
        upper_bias: float = 1.0,
        right_bias: float = 1.0,
        compute_faces: bool = False,
    ):
        df = pd.read_csv(csv_path, header=None)
        self.labels = df.iloc[:, 0].to_numpy(dtype=np.int64)
        self.images = (
            df.iloc[:, 1:].to_numpy(dtype=np.float32).reshape(-1, image_size, image_size)
            / 255.0
        )
        self.min_v = min_vertices
        self.max_v = max_vertices
        self.image_size = image_size
        self.vertex_bias = vertex_bias
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        self.upper_bias = upper_bias
        self.right_bias = right_bias
        self.compute_faces = compute_faces
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Data:
        img = self.images[idx]
        label = int(self.labels[idx])
        # power-transform u^(1/bias): bias=1 → uniform, bias>1 → skewed toward max_v.
        u = self._rng.uniform()
        t = u ** (1.0 / self.vertex_bias) if self.vertex_bias != 1.0 else u
        n_interior = int(self.min_v + (self.max_v - self.min_v) * t)
        points, edge_index, faces = build_mesh(
            n_interior,
            self.image_size,
            self._rng,
            upper_bias=self.upper_bias,
            right_bias=self.right_bias,
        )
        colors = bilinear_sample(img, points) * 2 - 1  # [-1, 1]
        pos = torch.from_numpy(points / self.image_size).float()
        kwargs = dict(
            pos=pos,
            edge_index=torch.from_numpy(edge_index).long(),
            y=torch.from_numpy(colors).float(),
            label=torch.tensor(label, dtype=torch.long),
        )
        if self.pe_type == "laplacian":
            # Anchor sign by comparing each eigvec's two extrema: the +/- 1
            # node closer to the top-left corner (0, 0) gets the positive sign.
            pe = compute_laplacian_pe(
                edge_index, len(points), self.pe_dim, points=points, anchor_xy=(0.0, 0.0)
            )
            kwargs["x"] = torch.from_numpy(pe).float()
        elif self.pe_type == "harmonic":
            pe = compute_harmonic_pe(edge_index, points, self.image_size)
            kwargs["x"] = torch.from_numpy(pe).float()
        elif self.pe_type == "lap_harmonic":
            lap = compute_laplacian_pe(
                edge_index, len(points), self.pe_dim, points=points, anchor_xy=(0.0, 0.0)
            )
            harm = compute_harmonic_pe(edge_index, points, self.image_size)
            pe = np.concatenate([harm, lap], axis=1)
            kwargs["x"] = torch.from_numpy(pe).float()
        if self.compute_faces:
            # Centroid bilinear sample is the Gouraud-rendered value at the
            # triangle centroid (barycentric 1/3, 1/3, 1/3) — same point the
            # mean of the 3 vertex predictions corresponds to.
            centroids = points[faces].mean(axis=1)
            tri_colors = bilinear_sample(img, centroids) * 2 - 1
            kwargs["face"] = torch.from_numpy(faces.T).long()
            kwargs["tri_y"] = torch.from_numpy(tri_colors).float()
        return Data(**kwargs)


class _LazyImageList:
    """Lazy duck-typed equivalent of MNIST's `images` ndarray; only loads on access.

    Lets eval/viz code uniformly reach `dataset.images[idx]` and get a float32
    array in [0, 1] regardless of whether the dataset is MNIST (eager) or
    Tiny ImageNet (lazy JPEG decode).
    """

    def __init__(self, paths):
        self.paths = paths

    def __getitem__(self, idx):
        return (
            np.asarray(Image.open(self.paths[idx]).convert("RGB"), dtype=np.float32)
            / 255.0
        )

    def __len__(self):
        return len(self.paths)


class TinyImageNetMeshDataset(Dataset):
    """Same shape as MNISTMeshDataset, but RGB 64x64 JPEGs from Tiny ImageNet.

    Returns `Data` objects with `y` of shape (N, 3) — multi-channel target.
    Mesh / PE machinery is identical; only the image source and per-vertex
    target dimensionality change.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        min_vertices: int = 2000,
        max_vertices: int = 4000,
        image_size: int = 64,
        seed: int | None = None,
        vertex_bias: float = 1.0,
        pe_type: str = "fourier",
        pe_dim: int = 16,
        upper_bias: float = 1.0,
        right_bias: float = 1.0,
        compute_faces: bool = False,
        use_clip: bool = False,
        mesh_type: str = "random",
        grid_resolution: int = 16,
        t5_emb_dir: str | None = None,
    ):
        root = Path(root)
        wnids = sorted(root.joinpath("wnids.txt").read_text().split())
        self.wnid_to_idx = {w: i for i, w in enumerate(wnids)}
        if split == "train":
            paths = sorted(root.joinpath("train").glob("*/images/*.JPEG"))
            labels = [self.wnid_to_idx[p.parent.parent.name] for p in paths]
        elif split == "val":
            val_map = {}
            for line in root.joinpath("val/val_annotations.txt").read_text().splitlines():
                fn, wnid = line.split("\t")[:2]
                val_map[fn] = wnid
            paths = sorted(root.joinpath("val/images").glob("*.JPEG"))
            labels = [self.wnid_to_idx[val_map[p.name]] for p in paths]
        else:
            raise ValueError(f"unknown split: {split}")
        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.int64)
        self.images = _LazyImageList(self.paths)
        self.num_classes = len(wnids)
        self.min_v = min_vertices
        self.max_v = max_vertices
        self.image_size = image_size
        self.vertex_bias = vertex_bias
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        self.upper_bias = upper_bias
        self.right_bias = right_bias
        self.compute_faces = compute_faces
        self.use_clip = use_clip
        self.mesh_type = mesh_type
        self.grid_resolution = grid_resolution
        if mesh_type == "grid":
            self._grid_mesh = build_grid_mesh(image_size, grid_resolution)
        elif mesh_type == "random":
            self._grid_mesh = None
        else:
            raise ValueError(f"unknown mesh_type: {mesh_type}")
        if use_clip:
            clip_path = root / f"{split}_clip.pt"
            self.clip_embeddings = torch.load(clip_path, weights_only=True)
            assert len(self.clip_embeddings) == len(self.paths), (
                f"clip embeddings count {len(self.clip_embeddings)} != paths "
                f"{len(self.paths)} — regenerate via compute_clip_embeddings.py?"
            )
        else:
            self.clip_embeddings = None
        if t5_emb_dir is not None:
            emb_dir = Path(t5_emb_dir)
            emb_list, mask_list = [], []
            for lbl in range(self.num_classes):
                rec = torch.load(emb_dir / f"{lbl}.pt", weights_only=False)
                emb_list.append(rec["emb"])
                mask_list.append(rec["mask"])
            self.t5_cond  = torch.stack(emb_list)   # (C, T, D)
            self.t5_cmask = torch.stack(mask_list)  # (C, T)
        else:
            self.t5_cond, self.t5_cmask = None, None
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Data:
        img = self.images[idx]  # (H, W, 3) float32 in [0, 1]
        label = int(self.labels[idx])
        if self._grid_mesh is not None:
            points, edge_index, faces = self._grid_mesh
        else:
            u = self._rng.uniform()
            t = u ** (1.0 / self.vertex_bias) if self.vertex_bias != 1.0 else u
            n_interior = int(self.min_v + (self.max_v - self.min_v) * t)
            points, edge_index, faces = build_mesh(
                n_interior,
                self.image_size,
                self._rng,
                upper_bias=self.upper_bias,
                right_bias=self.right_bias,
            )
        colors = bilinear_sample(img, points) * 2 - 1  # (N, 3) float32 in [-1, 1]
        pos = torch.from_numpy(points / self.image_size).float()
        kwargs = dict(
            pos=pos,
            edge_index=torch.from_numpy(edge_index).long(),
            y=torch.from_numpy(colors).float(),
            label=torch.tensor(label, dtype=torch.long),
        )
        if self.pe_type == "laplacian":
            pe = compute_laplacian_pe(
                edge_index, len(points), self.pe_dim, points=points, anchor_xy=(0.0, 0.0)
            )
            kwargs["x"] = torch.from_numpy(pe).float()
        elif self.pe_type == "harmonic":
            pe = compute_harmonic_pe(edge_index, points, self.image_size)
            kwargs["x"] = torch.from_numpy(pe).float()
        elif self.pe_type == "lap_harmonic":
            lap = compute_laplacian_pe(
                edge_index, len(points), self.pe_dim, points=points, anchor_xy=(0.0, 0.0)
            )
            harm = compute_harmonic_pe(edge_index, points, self.image_size)
            pe = np.concatenate([harm, lap], axis=1)
            kwargs["x"] = torch.from_numpy(pe).float()
        if self.compute_faces:
            centroids = points[faces].mean(axis=1)
            tri_colors = bilinear_sample(img, centroids) * 2 - 1
            kwargs["face"] = torch.from_numpy(faces.T).long()
            kwargs["tri_y"] = torch.from_numpy(tri_colors).float()
        if self.clip_embeddings is not None:
            # (1, D) so PyG cat-batches it along dim 0 to (B, D).
            kwargs["clip"] = self.clip_embeddings[idx].unsqueeze(0)
        if self.t5_cond is not None:
            kwargs["cond"]      = self.t5_cond[label].unsqueeze(0)   # (1, T, D)
            kwargs["cond_mask"] = self.t5_cmask[label].unsqueeze(0)  # (1, T)
        return Data(**kwargs)


class CIFAR10MeshDataset(Dataset):
    """CIFAR-10 32×32 RGB. Same shape contract as TinyImageNetMeshDataset:
    returns Data with `.pos`, `.edge_index`, `.y` (N,3 in [-1,1]), `.label`,
    optional `.x` (PE) and `.cond` / `.cond_mask` (T5).

    Loads from a `cifar-10-batches-py/` directory (5 train pickles + 1 test).
    `t5_emb_dir` is optional and expects {0..9}.pt files like the Tiny ImageNet cache.
    """

    CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        min_vertices: int = 200,
        max_vertices: int = 400,
        image_size: int = 32,
        seed: int | None = None,
        vertex_bias: float = 1.0,
        pe_type: str = "fourier",
        pe_dim: int = 16,
        upper_bias: float = 1.0,
        right_bias: float = 1.0,
        compute_faces: bool = False,
        mesh_type: str = "random",
        grid_resolution: int = 16,
        t5_emb_dir: str | None = None,
        patch_size: int = 32,
        same_grid=False,
        use_dino: bool = True,
    ):
        import pickle
        root = Path(root)
        batch_files = (["data_batch_1", "data_batch_2", "data_batch_3",
                        "data_batch_4", "data_batch_5"] if split == "train"
                       else ["test_batch"])
        imgs, labels = [], []
        for fn in batch_files:
            with open(root / fn, "rb") as f:
                d = pickle.load(f, encoding="bytes")
            imgs.append(d[b"data"])
            labels.extend(d[b"labels"])
        raw = np.concatenate(imgs, axis=0)                                  # (N, 3072) uint8
        self.images = (raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                       .astype(np.float32) / 255.0)                         # (N, 32, 32, 3) in [0,1]
        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_classes = 10
        self.min_v = min_vertices
        self.max_v = max_vertices
        self.image_size = image_size
        self.vertex_bias = vertex_bias
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        self.upper_bias = upper_bias
        self.right_bias = right_bias
        self.compute_faces = compute_faces
        self.mesh_type = mesh_type
        self.grid_resolution = grid_resolution
        self.patch_size = patch_size
        self.same_grid = same_grid

        if mesh_type == "grid":
            self._grid_mesh = build_grid_mesh(image_size, grid_resolution)
        elif mesh_type == "random":
            self._grid_mesh = None
        else:
            raise ValueError(f"unknown mesh_type: {mesh_type}")
        if same_grid and self._grid_mesh is not None:
            from utils import partition_metis
            pts, ei, _ = self._grid_mesh
            N   = pts.shape[0]
            n_p = max(1, N // patch_size)
            ei_t = torch.from_numpy(ei).long()
            self._partition = partition_metis(
                edge_index=ei_t,
                partitions_size=torch.tensor([n_p]),
                batch=torch.zeros(N, dtype=torch.long),
            )
            self._ei_t = ei_t
        else:
            self._partition = None
            self._ei_t     = None
        if t5_emb_dir is not None:
            emb_dir = Path(t5_emb_dir)
            emb_list, mask_list = [], []
            for lbl in range(self.num_classes):
                rec = torch.load(emb_dir / f"{lbl}.pt", weights_only=False)
                emb_list.append(rec["emb"])
                mask_list.append(rec["mask"])
            self.t5_cond  = torch.stack(emb_list)
            self.t5_cmask = torch.stack(mask_list)
        else:
            self.t5_cond, self.t5_cmask = None, None
        if use_dino:
            dino_path = root / f"{'test' if split == 'val' else split}_dino_cls.pt"
            self.dinos = torch.load(dino_path, weights_only=True)  # (N, 384)
            assert len(self.dinos) == len(self.labels), (
                f"dino embeddings {len(self.dinos)} != dataset size {len(self.labels)}"
            )
        else:
            self.dinos = None
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Data:
        img = self.images[idx]
        label = int(self.labels[idx])
        if self._grid_mesh is not None and self.same_grid:
            points, edge_index, faces = self._grid_mesh
        else:
            u = self._rng.uniform()
            t = u ** (1.0 / self.vertex_bias) if self.vertex_bias != 1.0 else u
            n_interior = int(self.min_v + (self.max_v - self.min_v) * t)
            points, edge_index, faces = build_mesh(
                n_interior, self.image_size, self._rng,
                upper_bias=self.upper_bias, right_bias=self.right_bias,
            )
        colors = bilinear_sample(img, points) * 2 - 1
        pos = torch.from_numpy(points / self.image_size).float()
        if self._partition is not None:
            ei_t = self._ei_t
            membership, edge_index_coarse, edge_index_intra, batch_coarse = self._partition
        else:
            from utils import partition_metis
            ei_t = torch.from_numpy(edge_index).long()
            N    = points.shape[0]
            n_p  = max(1, N // self.patch_size)
            membership, edge_index_coarse, edge_index_intra, batch_coarse = partition_metis(
                edge_index=ei_t,
                partitions_size=torch.tensor([n_p]),
                batch=torch.zeros(N, dtype=torch.long),
            )
        kwargs = dict(
            pos=pos,
            edge_index=ei_t,
            y=torch.from_numpy(colors).float(),
            label=torch.tensor(label, dtype=torch.long),
            membership=membership,
            edge_index_coarse=edge_index_coarse,
            edge_index_intra=edge_index_intra,
            batch_coarse=batch_coarse,
        )
        if self.dinos is not None:
            kwargs["dino"] = self.dinos[idx].unsqueeze(0)  # (1, 384)
        if self.t5_cond is not None:
            kwargs["cond"]      = self.t5_cond[label].unsqueeze(0)
            kwargs["cond_mask"] = self.t5_cmask[label].unsqueeze(0)
        return Data(**kwargs)

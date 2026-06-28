import torch
from cross_attention_bloc import CrossAttentionBloc, sinusoidal_embedding_1d, FourierPositionalEncoder
import torch.nn as nn
from torch_geometric.utils import to_dense_batch, to_edge_index, scatter
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.utils import softmax as scatter_softmax
import math

class LayerNormPyG(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(num_channels))
        self.shift = nn.Parameter(torch.zeros(num_channels))
        self.eps   = eps

    def forward(self, x, batch_vec, modulation=None):
        # x: (N_total, C); normalize across channels per node
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std(dim=-1, keepdim=True, correction=0)
        xn   = (x - mean) / (std + self.eps)
        if modulation is not None:
            # modulation[0], modulation[1]: (B, C) -> gather to (N_total, C)
            mod_s = modulation[0][batch_vec]
            mod_h = modulation[1][batch_vec]
            return xn * (1 + self.gamma[None] + mod_s) + self.shift[None] + mod_h
        return xn * (1 + self.gamma[None]) + self.shift[None]

class PGNN(torch.nn.Module) :
    def __init__(self, hidden_dim=256, sdim=768, hidden_dim_clip=512, n_blocs=8, n_pe_freqs=8,
                pe="none", n_heads=4, expansion_factor=4.0, res_connexions=False,
                n_sage_blocs_enc=5, n_sage_blocs_dec=5, patch_size=16, n_conv_blocs=1):
        super().__init__()
        pe = (pe or "none").lower()
        self.hdim      = hidden_dim
        self.n_blocs   = n_blocs
        self.hdim_clip = hidden_dim_clip
        self.freq_dim  = 16
        self.n_heads   = n_heads
        self.pe        = pe
        self.expansion_factor  = expansion_factor
        self.res_connexions    = res_connexions
        self.n_sage_blocs_enc  = n_sage_blocs_enc
        self.n_sage_blocs_dec  = n_sage_blocs_dec
        self.n_pe_freqs        = n_pe_freqs
        self.n_conv_blocs = n_conv_blocs
        self.t_emb_dim_conv = self.sdim

        self.pe_ = FourierPositionalEncoder(num_frequencies=n_pe_freqs)
        self.sdim = sdim+self.pe_.output_dim

        in_ch = 3 + self.pe_.output_dim

        # Linear channel pyramids — encoder ramps in_ch -> hdim, decoder ramps hdim -> 3.
        # Each has its own depth and so its own pyramid (they coincide when enc == dec).
        n_e, n_d = n_sage_blocs_enc, n_sage_blocs_dec
        stem_out = [(k + 1) * self.hdim // n_e for k in range(n_e - 1)] + [self.hdim]
        stem_in  = [in_ch] + stem_out[:-1]
        head_out = [(n_d - 1 - k) * self.hdim // n_d for k in range(n_d - 1)] + [3]
        head_in  = [self.hdim] + head_out[:-1]

        # === Stem (encoder): n SAGEConv + (n-1) LayerNormPyG + (n-1) adaLN-zero t-emb MLPs ===
        # SAGEConv(in, out) has the linear projection (W_neighbor, W_self) built in — no extra Linear needed.
        self.stem_sage  = nn.ModuleList([SAGEConv(ci, co)
                                        for ci, co in zip(stem_in, stem_out)])
        self.stem_ln    = nn.ModuleList([LayerNormPyG(c)
                                        for c in stem_out[:-1]])
        self.stem_t_mlp = nn.ModuleList([nn.Linear(self.t_emb_dim_conv, 2 * c)
                                        for c in stem_out[:-1]])

        # === Time embedding: sinusoidal -> Linear -> SiLU -> Linear (6*hdim for attn adaLN) ===
        self.time_embedding  = nn.Linear(self.freq_dim, self.sdim)
        self.time_projection = nn.Sequential(nn.SiLU(),
                                            nn.Linear(self.sdim, self.sdim * 6))

        # === DiT attention blocks (run at full T=1024) ===
        self.blocs = nn.ModuleList([
            CrossAttentionBloc(self.sdim, self.sdim, self.sdim,
                            self.n_heads, expansion_factor=self.expansion_factor,
                            ffn=True, pe=pe)
            for _ in range(n_blocs)
        ])
        for bloc  in self.blocs :
            bloc.init_normal(std=0.03/math.sqrt(3*self.n_blocs)) # because of attn bloc + ffn

        # === Head (decoder): n SAGEConv + (n-1) LayerNormPyG + (n-1) adaLN-zero t-emb MLPs ===
        self.head_sage  = nn.ModuleList([SAGEConv(ci, co)
                                        for ci, co in zip(head_in, head_out)])
        self.head_ln    = nn.ModuleList([LayerNormPyG(c)
                                        for c in head_out[:-1]])
        self.head_t_mlp = nn.ModuleList([nn.Linear(self.t_emb_dim_conv, 2 * c)
                                        for c in head_out[:-1]])

        # DiT-Zero on the final output SAGE: zero all params -> 0 output at init.
        for p in self.head_sage[-1].parameters():
            nn.init.zeros_(p)

        # Zero-init every adaLN MLP so modulation starts at 0 (LN acts as plain norm at init).
        for mlp in list(self.stem_t_mlp) + list(self.head_t_mlp):
            nn.init.zeros_(mlp.weight)
            nn.init.zeros_(mlp.bias)

        # === Class-label conditioning: Embedding(11, 16) -> MLP -> per-block adaLN ===
        # 11 = CIFAR-10 (0..9) + 1 unconditional slot (10) used by cfg_drop in training.
        self.label_embedding_dim = 16
        self.label_embedding     = torch.nn.Embedding(11, self.label_embedding_dim)
        self.clip_linear         = torch.nn.Sequential(
            torch.nn.Linear(self.label_embedding_dim,      4 * self.label_embedding_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(4 * self.label_embedding_dim, n_blocs * 6 * self.sdim),
        )
        torch.nn.init.zeros_(self.clip_linear[-1].weight)
        torch.nn.init.zeros_(self.clip_linear[-1].bias)
        self.patch_size = patch_size

        self.coarsen_linear = torch.nn.Linear(self.hdim, sdim)
        self.score = nn.Sequential(
            nn.Linear(self.hdim+self.pe_.output_dim, self.hdim),
            nn.SiLU(),
            nn.Linear(self.hdim, 1),
        )
        self.pool = lambda h, membership, M: scatter(h, index=membership, dim=0, dim_size=M, reduce='mean')

        # we keep hdim for embedding dim of single nodes, we use sdim  for embedding dim of metanode
        self.t_emb_linear_conv = torch.nn.Sequential(torch.nn.Linear(self.freq_dim, self.t_emb_dim_conv), torch.nn.SiLU())

        conv_blocs = []
        stem_t_mlp_conv = []
        meta_mod_mlp   = []
        for i in range(self.n_blocs) :
            for k in range(self.n_conv_blocs) :
                conv_blocs.append(SAGEConv(self.hdim, self.hdim))
                conv_blocs.append(LayerNormPyG(self.hdim))
                stem_t_mlp_conv.append(torch.nn.Linear(self.t_emb_dim_conv, 2*self.hdim))
                meta_mod_mlp.append(torch.nn.Linear(self.sdim, 2*self.hdim))

        self.conv_blocs      = torch.nn.ModuleList(conv_blocs)
        self.stem_t_mlp_conv = torch.nn.ModuleList(stem_t_mlp_conv) # We also inject t in the convolution blocs
        self.meta_mod_mlp    = torch.nn.ModuleList(meta_mod_mlp)    # per-partition adaLN-zero from metanode features

        # Zero-init meta-mod MLPs: at step 0 meta contributes nothing, conv runs time-conditioned only.
        for mlp in self.meta_mod_mlp:
            torch.nn.init.zeros_(mlp.weight)
            torch.nn.init.zeros_(mlp.bias)

        torch.nn.init.normal_(self.coarsen_linear.weight, std=0.03/math.sqrt(3*self.n_blocs))
        torch.nn.init.zeros_(self.coarsen_linear.bias)
        

    def forward(self, data, labels, ts) :
        """Almost the same as in the original model,
            Except here we add extra convolution layer between attention blocs
        """
        B          = data.num_graphs
        batch_vec  = data.batch
        edge_index = data.edge_index

        # Scale ts into the range the sinusoidal embedding is calibrated for
        # (OpenAI/ADM convention: integer timesteps 0..999, max_period=10000).
        # Without this, ts ~ N(0,1) leaves ~12 of 16 embedding dims at near-constant
        # sin(0)/cos(0) values, wasting ~75% of the time-emb representational capacity.
        ts = ts * 999.0

        # Time embedding (shared base for both LN modulation and attention adaLN)
        t0 = sinusoidal_embedding_1d(self.freq_dim, ts).float() # (b, self.freq_dim)
        e  = self.time_embedding(t0)  # (B, hdim)
        e0 = self.time_projection(e).view(B, 6, self.sdim)                            # (B, 6, hdim)
        t_proj_conv = self.t_emb_linear_conv(t0)                # (B, t_emb_dim_conv)

        # Per-node input features: noisy RGB + Fourier-encoded position
        rgb_in = data.x[:, :3]
        pos_in = data.x[:, 3:5]
        pos_in_fourier = self.pe_(pos_in)
        h      = torch.cat([rgb_in, pos_in_fourier], dim=-1)                        # (N_total, in_ch)

        # Partitioning pre-computed in the dataloader worker — read directly from batch.
        # membership has no "index"/"face" → PyG doesn't offset it → apply manually.
        n_parts          = int(data.membership.max()) + 1
        membership       = data.membership + batch_vec * n_parts
        edge_index_intra = data.edge_index_intra   # auto-offset by PyG (contains "index")
        batch_coarse     = data.batch_coarse        # auto-offset by PyG ("batch" attr)

        # Stem (encoder) in PyG flat layout: SAGE_k -> SiLU -> LN_k(mod) (last SAGE has no LN after)
        for k, sage in enumerate(self.stem_sage):
            h = sage(h, edge_index)
            if k < len(self.stem_ln):
                s, sh = self.stem_t_mlp[k](t_proj_conv).chunk(2, dim=-1)
                h = F.silu(h)
                h = self.stem_ln[k](h, batch_vec, modulation=(s, sh))

        # we append to each metanode the mean of corresponding positions
        pos_metanodes = scatter(self.pe_(pos_in), membership, reduce='mean')

        # linear to pass from hdim to sdim then mean pool
        scores = self.score(torch.cat([h, pos_in_fourier], dim=-1))          # (N, 1)
        weights = scatter_softmax(scores, index=membership)                   # (N, 1), sums to 1 per metanode
        h_meta = self.coarsen_linear(
            scatter(weights * h, index=membership, dim_size=batch_coarse.numel(), dim=0, reduce='sum')
        )        
        h_meta = torch.cat([h_meta, pos_metanodes], dim=-1)
        x_metanode, mask_meta = to_dense_batch(h_meta, batch_coarse)                  # (B, M_max, sdim)

        # Per-block class-conditional adaLN modulations
        clips_modulations = self.clip_linear(self.label_embedding(labels)) \
                                .view(B, self.n_blocs, 6, self.sdim) \
                                .permute(1, 0, 2, 3)                                  # (n_blocs, B, 6, hdim)

        #x=h.clone()
        # Self-attention DiT stack (time + label adaLN) — meta conditions fine convs via per-partition adaLN-zero
        for i, bloc in enumerate(self.blocs):
            x_metanode  = bloc(x_metanode, x_metanode, e0 + clips_modulations[i])     # (B, M_max, sdim)
            h_meta_flat = x_metanode[mask_meta]                                       # (M_total, sdim)
            for k in range(self.n_conv_blocs):
                idx = k + i*self.n_conv_blocs
                s_t, sh_t = self.stem_t_mlp_conv[idx](t_proj_conv).chunk(2, dim=-1)   # (B, hdim) each
                s_m, sh_m = self.meta_mod_mlp[idx](h_meta_flat).chunk(2, dim=-1)      # (M_total, hdim) each

                h = self.conv_blocs[2*idx](h, edge_index_intra)
                h = F.silu(h)
                h = self.conv_blocs[2*idx + 1](h, batch_vec, modulation=(s_t, sh_t))  # LN with time adaLN
                h = h * (1 + s_m[membership]) + sh_m[membership]                      # extra per-partition meta adaLN
                #x += h

            scores = self.score(torch.cat([h, pos_in_fourier], dim=-1))          # (N, 1)
            weights = scatter_softmax(scores, index=membership)                   # (N, 1), sums to 1 per metanode
            h_meta = self.coarsen_linear(
                scatter(weights * h, index=membership, dim_size=batch_coarse.numel(), dim=0, reduce='sum')
            ) 
            x_residual  = torch.cat([to_dense_batch(h_meta, batch_coarse)[0], pos_metanodes.reshape(B, -1, self.pe_.output_dim)], dim=-1)
            x_metanode  = x_residual + x_metanode

        #h=x.clone()
        # Head (decoder) in PyG flat layout (mirror of stem)
        for k, sage in enumerate(self.head_sage):
            h = sage(h, edge_index)
            if k < len(self.head_ln):
                s, sh = self.head_t_mlp[k](t_proj_conv).chunk(2, dim=-1)
                h = F.silu(h)
                h = self.head_ln[k](h, batch_vec, modulation=(s, sh))
        return h                                                                      # (N_total, 3)

# wandb key : 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
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import grid
from torch_geometric.nn import SAGEConv
from clearml import Task
import os
import matplotlib.pyplot as plt
import torchvision
device = 'cpu'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def compare_color_distributions(img_path_none, img_path_laplace):
    # Chargement des images
    img_none = cv2.cvtColor(cv2.imread(img_path_none), cv2.COLOR_BGR2RGB)
    img_laplace = cv2.cvtColor(cv2.imread(img_path_laplace), cv2.COLOR_BGR2RGB)

    colors = ('r', 'g', 'b')
    plt.figure(figsize=(15, 5))

    for i, col in enumerate(colors):
        # Extraction du canal i
        hist_none = cv2.calcHist([img_none], [i], None, [256], [0, 256]).flatten()
        hist_laplace = cv2.calcHist([img_laplace], [i], None, [256], [0, 256]).flatten()

        # Normalisation pour avoir une distribution de probabilité
        hist_none /= hist_none.sum()
        hist_laplace /= hist_laplace.sum()

        # Calcul de la distance de Wasserstein pour ce canal
        # Elle représente le "coût" pour transformer une distribution en l'autre
        w_dist = wasserstein_distance(img_none[:,:,i].flatten(), img_laplace[:,:,i].flatten())

        plt.subplot(1, 3, i+1)
        plt.plot(hist_none, color=col, linestyle='--', label=f'None (Dist)')
        plt.fill_between(range(256), hist_laplace, color=col, alpha=0.3, label=f'Laplace (Dist)')
        plt.title(f'Canal {col.upper()}\nW-Dist: {w_dist:.2f}')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Utilisation
# compare_color_distributions('inference_features_None.png', 'inference_laplace.png')

class DummyLogger:
    def report_scalar(self, **kwargs):
        pass

def create_random_grids(batch_size, grid_size=64):
    edge_index, pos = grid(height=grid_size, width=grid_size)

    data_list = []
    for _ in range(batch_size):
        rgb_values = torch.randn((grid_size * grid_size, 3))        
        d = Data(x=rgb_values, edge_index=edge_index)
        data_list.append(d)
    
    return Batch.from_data_list(data_list)

def create_positional_features(batch, method='laplacian', num_frequencies=4):
    # Création de la grille de base [64*64, 2] entre 0 et 1
    x, y = torch.meshgrid(torch.arange(64), torch.arange(64), indexing='ij')
    coords = torch.stack([x, y], dim=2).float().reshape(-1, 2) / 63.0 # On utilise 63 pour avoir pile 1.0

    if method == 'indexing':
        batch_features = coords
    elif method == 'laplacian':
            N = 64
            i = torch.arange(N).view(N, 1).repeat(1, N).reshape(-1) # [4096]
            j = torch.arange(N).view(1, N).repeat(N, 1).reshape(-1) # [4096]
            # La 2ème et 3ème (k=1, l=0 et k=0, l=1) sont les vecteurs propres fondamentaux :
            # v_{k,l}(i,j) = cos(pi * k * (i + 0.5) / N) * cos(pi * l * (j + 0.5) / N)
            
            u2 = torch.cos(torch.pi * (i + 0.5) / N)
            u3 = torch.cos(torch.pi * (j + 0.5) / N)
            batch_features = torch.stack([u2, u3], dim=-1) # [4096, 2]
    elif method == 'fourier':
        # On crée des fréquences : 2^0, 2^1, 2^2...
        freqs = 2.0 ** torch.arange(num_frequencies)
        features = []
        
        for f in freqs:
            # On multiplie par 2*pi*f
            for c in range(2): # Pour x et y
                features.append(torch.sin(2 * torch.pi * f * coords[:, c]))
                features.append(torch.cos(2 * torch.pi * f * coords[:, c]))
        
        # On concatène tout : 4 fréquences * 2 (sin/cos) * 2 (x,y) = 16 features
        batch_features = torch.stack(features, dim=-1)
    elif method == 'None':
        batch_features = torch.zeros((64*64, 0))  # Pas de features de position
    else:
        raise ValueError(f"Unknown positional encoding method: {method}")

    # Répétition pour tout le batch
    num_graphs = batch.max().item() + 1
    return batch_features.repeat(num_graphs, 1)

def positional_features(data, indexing_method='None') :
    x = data.x
    print('x.shape = ', x.shape)
    pos = create_positional_features(data.batch, method=indexing_method)
    print('pos.shape = ', pos.shape)
    return torch.cat([x, pos], dim=-1)

def sample(steps=10, sampler='euler', shift=5) :
    if sampler == 'euler' :
        ts = torch.linspace(0, 1, steps+1)
    else :
        print(f"Sampler {sampler} is not currently supported")
    return ts*shift / (1 + (shift-1)*ts)

def get_dim(indexing_method) :
    if indexing_method == 'None' :
        return 0
    elif indexing_method == 'indexing' :
        return 2
    elif indexing_method == 'laplacian' :
        return 2
    elif indexing_method == 'fourier' :
        return 16
    else :
        raise ValueError(f"Unknown indexing method: {indexing_method}")

def apply_shift(t_mod, shift) :
    t_mod = t_mod * shift / (1 + (shift-1)*t_mod)
    return t_mod

def visualize_grid(data,save_path, t=None) :
    if isinstance(data, Data) :
        img = torch.sigmoid(data.x.reshape(-1, 64, 3)).cpu().numpy()
    else : # we suppose it is directly x
        img = torch.sigmoid(data.reshape(-1, 64, 3)).cpu().numpy()
    if t is not None:
        plt.title(f"t={t}")
    plt.imshow(img)
    plt.savefig(save_path)

class FlowGNNDataset(torch.utils.data.Dataset):
    def __init__(self, path_img, grid_size=64):
        super().__init__()
        self.path_img = path_img
        self.grid_size = grid_size
        self.list_files = os.listdir(path_img)

        self.edge_index, self.pos = grid(height=grid_size, width=grid_size)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img = torch.logit(torchvision.io.read_image(
            os.path.join(self.path_img, self.list_files[index])
        ).float() / 255.0, eps=1e-3)  # (C,H,W)

        img = F.interpolate(img[None], size=(self.grid_size, self.grid_size), mode='bilinear')
        img = img[0]  # (C,H,W)

        x = img.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)

        return Data(x=x, edge_index=self.edge_index)
    
class FlowGNN(torch.nn.Module):
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=128, depth=3, indexing_method='None'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.indexing_method = indexing_method
        self.depth = depth
        in_dim = input_dim + get_dim(indexing_method)

        dims = [in_dim]

        # hidden layers
        for _ in range(depth - 1):
            dims.append(hidden_dim)

        dims.append(output_dim)

        self.convs = torch.nn.ModuleList([
            SAGEConv(dims[i], dims[i+1], root_weight=True)
            for i in range(len(dims) - 1)
        ])

        # --------- Time modulation size ---------
        self.hidden_dims = dims[1:]  # exclude input
        self.mod_size = sum(self.hidden_dims)

        self.time_modulation = torch.nn.Sequential(
            torch.nn.Linear(1, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 2 * self.mod_size)
        )


    def apply_mod(self, x, mod, batch_index):
        # x (\somme graphes du batch * nnodes, input_dim)
        # mod ()
        gamma, beta = mod.chunk(2, dim=-1)
        gamma       = gamma[batch_index]
        beta        = beta[batch_index]

        return x * (1 + gamma) + beta

    def forward(self, data, t):
        x, edge_indices, batch_index = data.x, data.edge_index, data.batch
        x = positional_features(data, self.indexing_method)  # On ajoute les features de position (i,j) à l'entrée
        print('x shape: ', x.shape)
        # Génération des paramètres de modulation
        t_mod = self.time_modulation(t.view(-1, 1))
        mods = torch.split(t_mod, [2 * d for d in self.hidden_dims], dim=-1)

        # x (\somme graphes du batch * nnodes, input_dim)
        # Flow

        h = x
        residuals = []

        for i, conv in enumerate(self.convs):
            h = conv(h, edge_indices)

            gamma_beta = mods[i]
            gamma, beta = gamma_beta.chunk(2, dim=-1)

            gamma = gamma[batch_index]
            beta = beta[batch_index]

            h = h * (1 + gamma) + beta

            if i < len(self.convs) - 1:
                h = F.silu(h)

        return h

    def load_ckpt(self, ckpt_path) :
        state_dict = torch.load(ckpt_path) 
        self.load_state_dict(state_dict)

    def step(self, input_batch, shift=5):
        B = input_batch.num_graphs
        #t = torch.ones(B, device=input_batch.x.device) * 0.5  # On peut aussi échantillonner t aléatoirement
        t = torch.rand(B, device=input_batch.x.device)
        t = apply_shift(t, 1/shift)

        noise = create_random_grids(B)
        noise = noise.to(input_batch.x.device)

        batch_index = input_batch.batch
        t_node = t[batch_index].unsqueeze(-1)  # [N_total, 1]

        x1  = input_batch.x
        x0  = noise.x

        x_t = t_node * x1 + (1-t_node) * x0

        u_t = input_batch.clone()
        u_t.x = x_t

        v_pred = self.forward(u_t, t)
        loss   = torch.nn.functional.mse_loss(v_pred, x1 - x0)

        return x_t + t[batch_index][...,None]*v_pred, x_t, t, loss
    
    @torch.no_grad()
    def inference(self, grid_size=64, steps=10, sampler="euler", shift=5, device='cpu'):
        x0 = create_random_grids(1, grid_size=grid_size).to(device)
        t  = sample(steps, sampler, shift).to(device)

        print('t: ', t)

        for step in range(steps) :
            v_pred   = self.forward(x0, t[step])
            x0.x     = x0.x + (t[step+1] - t[step])*v_pred
        
        return torch.sigmoid((x0.x).reshape(grid_size, grid_size, 3))

def train_loop(model, dataloader, logger, name='training', lr=1e-3, epochs=10, device=device) :
    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr)
    global_step = 0

    save_fig    = 100
    save_model  = 500

    for epoch in range(epochs):
        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            pred, noise, t, loss = model.step(batch)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
            logger.report_scalar(
                title="train",
                series="loss",
                value=loss.item(),
                iteration=global_step
            )
       
            if global_step % save_fig == 0 :
                with torch.no_grad() :
                    visualize_grid(noise, save_path=f"/home/thibault/graph/outputs/original_{global_step}.png", t=t)
                    visualize_grid(pred, save_path=f"/home/thibault/graph/outputs/pred_{global_step}.png", t=t)

            global_step += 1

            if global_step % save_model == 0 :
                torch.save(model.state_dict(), f"/home/thibault/graph/ckpt/model_{name}_{global_step}.pth")


def create_dataloader(path_images, batch_size=4) :
    dataset = FlowGNNDataset(path_images)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Flow matching GNN',
        description='Simple implementation of a flow matching algorithm using GNNs',
    )
    parser.add_argument('--debug', action='store_true', help='Disable ClearML logging')
    args = parser.parse_args()

    if not args.debug:
        task = Task.init(
            project_name="FlowGNN",
            task_name="training_run",
            task_type=Task.TaskTypes.training
        )
        logger = task.get_logger()
    else:
        logger = DummyLogger()  # ou un logger factice

    torch.manual_seed(41)
    model = FlowGNN(input_dim=3, output_dim=3, hidden_dim=128, depth=6, indexing_method='None').to(device)
    print(model)

    # Simple forward with random data
    # # Dummy data
    # edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    # x = torch.randn((3, 3))  # 3 nodes, 3 features each
    # data = Data(x=x, edge_index=edge_index)
    # # We create a batch of 4 graphs for testing
    # batch_data = Batch.from_data_list([data, data, data, data])
    # t = torch.tensor([0.6, 0.6, 0.6, 0.6])  # Time step
    # output = model(batch_data, t)
    # print(output.shape)
    # print(output)

    # Lauch training
    path_img = "/home/thibault/graph/my_dataset/overfit_data"
    batch_size = 8
    dataloader = create_dataloader(path_img, batch_size=batch_size)
    train_loop(model, dataloader, logger, name='features_None', epochs=1000, lr=3e-4)

    ckpt_path = "/home/thibault/graph/ckpt/model_features_None_1000.pth"
    model.load_ckpt(ckpt_path)
    generated_sample = model.inference(steps=50, sampler="euler", shift=1/5, device='cpu')
    img_tensor = (generated_sample.permute(2, 0, 1)*255.0).to(torch.uint8)
    torchvision.io.write_png(img_tensor, 'outputs/inference_features_None.png')
    plt.imshow(generated_sample.cpu().numpy())
    plt.show()
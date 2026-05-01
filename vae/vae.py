import torch
import os
import pandas as pd
import tqdm
from utils_visu import test_reconstruction, random_sampling
import matplotlib.pyplot as plt
import random

from clearml import Task
import numpy as np
from PIL import Image

import logging
logging.getLogger("clearml").setLevel(logging.DEBUG)

class VAEModele(torch.nn.Module) :
    def __init__(self, input_dim=784, latent_dim=128):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2*input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*input_dim, 2*latent_dim), # outputs mu, log sigma
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2*input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*input_dim, input_dim),
            torch.nn.Sigmoid()
        )

        self.decoder[-2].bias.data.fill_(torch.logit(torch.tensor(0.1307)))
    
    def forward(self, x, sample = False) :
        """
        x is (B, input_dim)
        """

        latent_statistics = self.encoder(x)
        mus, log_sigmas   = latent_statistics[:, :self.latent_dim], latent_statistics[:, self.latent_dim:2*self.latent_dim]

        zs = mus
        if sample :
            zs = mus + torch.randn_like(mus) * torch.exp(log_sigmas)
            return self.decoder(zs), mus, log_sigmas, zs
        
        return self.decoder(zs)
    
def train_loop(model, dataloader, val_dataloader, epochs, lr=3e-4) :
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr)
    n_steps    = 0
    losses     = []
    loss_recon = []
    loss_kl    = []
    loss_val   = []
    mus_list   = []
    sigmas_list = []
    mus_sample = []
    sigmas_sampled = []

    beta = 1e-3
    for epoch in range(epochs) :
        loss_epoch = 0
        for x in tqdm.tqdm(dataloader) :
            optimizer.zero_grad()

            x_pred, mus, log_sigmas, zs = model.forward(x, sample=True)

            beta_kl = -0.5 * (1 + 2*log_sigmas - mus**2 - torch.exp(2*log_sigmas)).sum(dim=1).mean() * beta
            mse_loss = torch.nn.functional.mse_loss(x, x_pred)
            loss = mse_loss + beta_kl
            loss_epoch += loss.item()

            loss.backward()
            optimizer.step()
            n_steps += 1
            logger.report_scalar("Loss",                  "total",    loss.item(),                                       iteration=n_steps)
            logger.report_scalar("Loss",                  "recon",    mse_loss.item(),                                   iteration=n_steps)
            logger.report_scalar("Loss",                  "KL",       beta_kl.item(),                                     iteration=n_steps)
            logger.report_scalar("Latent / mus",          "L2 norm",  mus.mean(dim=0).norm(p=2).item(),                  iteration=n_steps)
            logger.report_scalar("Latent / sigmas",       "L2 norm",   log_sigmas.exp().norm(p=2, dim=1).mean().item(),   iteration=n_steps)
            logger.report_scalar("Latent / z samples",    "mean L2",  zs.mean(dim=0).norm(p=2).item(),                   iteration=n_steps)
            logger.report_scalar("Latent / z samples",    "std L2",   zs.std(dim=0).norm(p=2).item(),                    iteration=n_steps)


        # Val
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad() :
            for x in val_dataloader :
                x_pred = model.forward(x, sample=False)
                beta_kl  = -0.5 * (1 + 2*log_sigmas - mus**2 - torch.exp(2*log_sigmas)).sum(dim=1).mean() * beta
                mse_loss = torch.nn.functional.mse_loss(x, x_pred)
                val_loss_epoch += (mse_loss + beta_kl).item()
        loss_val.append(val_loss_epoch / len(val_dataloader))
        model.train()


        logger.report_scalar("Loss", "val", loss_val[-1], iteration=n_steps)


        # ── reconstruction debug image ───────────────────────────────────
        try:
            recon_path = os.path.join('outputs', f'vae_reconstruction_epoch_{epoch}.png')
            test_reconstruction(
                model, dataloader,
                index=random.randint(0, len(dataloader) - 1),
                save_path=recon_path,
            )
            fig = plt.figure()
            plt.imshow(plt.imread(recon_path))
            plt.axis('off')
            logger.report_matplotlib_figure("Reconstruction", "epoch", iteration=epoch, figure=fig)
            plt.close(fig)
            print(">>> reconstruction sent to clearml")
        except Exception as e:
            print(f">>> reconstruction FAILED: {e}")
        # ─────────────────────────────────────────────────────────────────

        model.train()

    # ── final random samples ─────────────────────────────────────────────
    samples_path = os.path.join('outputs', 'vae_random_samples.png')
    random_sampling(model, num_samples=10, device=None, save_path=samples_path)
    fig = plt.figure()
    plt.imshow(plt.imread(samples_path))
    plt.axis('off')
    logger.report_matplotlib_figure("Random Samples", "final", iteration=epochs, figure=fig)
    plt.close(fig)
    logger.flush()
    # ─────────────────────────────────────────────────────────────────────


if __name__ == '__main__' :
    torch.manual_seed(42)
    random.seed(42)

    BATCH_SIZE  = 128
    LR = 4e-3
    EPOCHS = 30

    # ── ClearML task ────────────────────────────────────────────────────────
    task = Task.init(
        project_name="VAE-MNIST",
        task_name="classic-vae",
        auto_connect_frameworks={"pytorch": True},
        output_uri="https://files.clear.ml",  # use ClearML's own file server
    )
    task.connect({
        "batch_size": BATCH_SIZE,
        "lr":         LR,
        "epochs":     EPOCHS,
        "latent_dim": 128,
        "beta":       1e-3,
    })
    logger = task.get_logger()
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_title("test plot")
    logger.report_matplotlib_figure("Test", "test", iteration=0, figure=fig)
    plt.close(fig)
    logger.flush()

    print(">>> test plot sent")

    # ────────────────────────────────────────────────────────────────────────

    data = pd.read_csv('/home/thibault/NN/MNIST/MNIST2/mnist_train.csv').iloc[:,1:]
    data_torch = torch.Tensor(data.to_numpy())/255.0
    print(data_torch.shape)
    print(data_torch.min())
    print(data_torch.max())
    print(data_torch.std())
    print(data_torch.mean())

    # Split train/val
    val_size   = int(0.15 * len(data_torch))
    train_size = len(data_torch) - val_size
    train_data, val_data = torch.utils.data.random_split(data_torch, [train_size, val_size])

    dataloader     = torch.utils.data.DataLoader(train_data, BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, BATCH_SIZE, shuffle=False)

    model = VAEModele()
    train_loop(model, dataloader, val_dataloader, EPOCHS, LR)
    torch.save(model.state_dict(), './ckpt/vae_final.pth')
    random_sampling(model, num_samples=10, device=None, save_path=os.path.join('outputs', 'vae_random_samples.png'))
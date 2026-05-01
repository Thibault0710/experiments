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

class VAEModele(torch.nn.Module):
    def __init__(self, input_dim=784, latent_dim=128):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 2*input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*input_dim, 2*latent_dim),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 2*input_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2*input_dim, input_dim),
            torch.nn.Sigmoid()
        )

        self.decoder[-2].bias.data.fill_(torch.logit(torch.tensor(0.1307)))

    def forward(self, x, sample=False):
        latent_statistics = self.encoder(x)
        mus       = latent_statistics[:, :self.latent_dim]
        log_sigmas = latent_statistics[:, self.latent_dim:2*self.latent_dim]

        zs = mus
        if sample:
            zs = mus + torch.randn_like(mus) * torch.exp(log_sigmas)
            return self.decoder(zs), mus, log_sigmas, zs

        return self.decoder(zs)


def sigloss(x_pred): # used from le-wm repo
    num_proj = 1024
    knots = 17 * 2
    ts = torch.linspace(-3, 3, knots, dtype=torch.float32)
    dt = ts[1] - ts[0]
    phis = torch.exp(-ts.square() / 2.0)
    weights = torch.full((knots,), 2 * dt, dtype=torch.float32) * phis
    weights[[0, -1]] = dt

    A = torch.randn(x_pred.shape[-1], num_proj)
    A = A.div_(A.norm(p=2, dim=0))

    x_t = (x_pred @ A).unsqueeze(-1) * ts
    err = (x_t.cos().mean(-3) - phis).square() + x_t.sin().mean(-3).square()

    statistic = (err @ weights) * x_pred.size(-2)
    return statistic.mean()


def train_loop(model, dataloader, val_dataloader, epochs, lr=3e-4):
    # ── ClearML logger ──────────────────────────────────────────────────────
    task   = Task.current_task()
    logger = task.get_logger()
    # ────────────────────────────────────────────────────────────────────────

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n_steps   = 0
    loss_val  = []

    beta = 9e-3

    for epoch in range(epochs):
        loss_epoch = 0
        model.train()

        for x in tqdm.tqdm(dataloader):
            optimizer.zero_grad()

            x_pred, mus, log_sigmas, zs = model.forward(x, sample=True)

            sigreg   = sigloss(zs) * beta
            mse_loss = torch.nn.functional.mse_loss(x, x_pred)
            loss     = mse_loss + sigreg

            loss.backward()
            optimizer.step()

            # ── log every step ───────────────────────────────────────────
            logger.report_scalar("Loss",                  "total",    loss.item(),                                       iteration=n_steps)
            logger.report_scalar("Loss",                  "recon",    mse_loss.item(),                                   iteration=n_steps)
            logger.report_scalar("Loss",                  "sigreg",   sigreg.item(),                                     iteration=n_steps)
            logger.report_scalar("Latent / mus",          "L2 norm",  mus.mean(dim=0).norm(p=2).item(),                  iteration=n_steps)
            logger.report_scalar("Latent / sigmas",       "L2 norm",   log_sigmas.exp().norm(p=2, dim=1).mean().item(),   iteration=n_steps)
            logger.report_scalar("Latent / z samples",    "mean L2",  zs.mean(dim=0).norm(p=2).item(),                   iteration=n_steps)
            logger.report_scalar("Latent / z samples",    "std L2",   zs.std(dim=0).norm(p=2).item(),                    iteration=n_steps)
            # ─────────────────────────────────────────────────────────────

            loss_epoch += loss.item()
            n_steps    += 1

        # ── validation ───────────────────────────────────────────────────
        model.eval()
        val_loss_epoch = 0
        with torch.no_grad():
            for x in val_dataloader:
                x_pred, _, _, zs = model.forward(x, sample=True)
                sigreg   = sigloss(zs) * beta
                mse_loss = torch.nn.functional.mse_loss(x, x_pred)
                val_loss_epoch += (mse_loss + sigreg).item()

        val_loss = val_loss_epoch / len(val_dataloader)
        loss_val.append(val_loss)

        logger.report_scalar("Loss", "val", val_loss, iteration=n_steps)
        # ─────────────────────────────────────────────────────────────────

        print(f"epoch: {epoch}  loss_epoch={loss_epoch/len(dataloader):.5f}")

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


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)

    BATCH_SIZE = 128
    LR         = 4e-3
    EPOCHS     = 30

    # ── ClearML task ────────────────────────────────────────────────────────
    task = Task.init(
        project_name="VAE-MNIST",
        task_name="sigreg-vae",
        auto_connect_frameworks={"pytorch": True},
        output_uri="https://files.clear.ml",  # use ClearML's own file server
    )
    task.connect({
        "batch_size": BATCH_SIZE,
        "lr":         LR,
        "epochs":     EPOCHS,
        "latent_dim": 128,
        "beta":       9e-3,
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

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('ckpt',    exist_ok=True)

    data       = pd.read_csv('/home/thibault/NN/MNIST/MNIST2/mnist_train.csv').iloc[:, 1:]
    data_torch = torch.Tensor(data.to_numpy()) / 255.0

    val_size   = int(0.15 * len(data_torch))
    train_size = len(data_torch) - val_size
    train_data, val_data = torch.utils.data.random_split(data_torch, [train_size, val_size])

    dataloader     = torch.utils.data.DataLoader(train_data, BATCH_SIZE, shuffle=True,  drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_data,   BATCH_SIZE, shuffle=False, drop_last=True)

    model = VAEModele()
    train_loop(model, dataloader, val_dataloader, EPOCHS, LR)

    torch.save(model.state_dict(), './ckpt/vae_final.pth')
    task.close()

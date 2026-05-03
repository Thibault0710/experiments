import torch
from latent_model import MNISTGeneratorLatent
from tqdm import tqdm
from train import shift
import pandas as pd
from clearml import Task
import random

import logging
logging.getLogger("clearml").setLevel(logging.DEBUG)

SHIFT=5.0

def step(model, x) :
    """
    x of shape (B, latent_dim)
    """
    B = x.shape[0]
    eps = torch.randn_like(x)
    ts  = shift(torch.rand(B)[:, None], SHIFT) * torch.ones_like(x) 

    xt     = (1-ts)*x + ts*eps # x + (eps - x)*ts
    v_pred = model.forward(xt, ts[:,0].unsqueeze(-1))

    return v_pred, eps - x

def inference(model, vae, x, steps, t_start=1, t_end=0) :
    # We denoise using euler simple
    
    ts = torch.linspace(t_start, t_end, steps)
    ts = shift(ts, 5.0)
    with torch.no_grad() :
        for step in range(steps-1) :
            v_pred = model(x, ts[step].unsqueeze(0))
            x = x - (ts[step] - ts[step+1])*v_pred
    
        return vae.decoder(x).reshape(-1, 28 , 28)

def train_loop(model, dataset, epochs, lr=3e-4, bs=16) :
    dataset   = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loss = []
    imgs = []
    for epoch in range(epochs) :
        print(f'Epoch : {epoch}')
        epoch_loss = []
        for x in tqdm(dataset) :
            optimizer.zero_grad()

            v_pred, v_gt = step(model, x)
            loss = torch.nn.functional.mse_loss(v_pred, v_gt)
            epoch_loss.append(loss.item())
            loss.backward()

            gradients = model.get_gradients_updates()
            for d, gradient in enumerate(gradients) :
                logger.report_scalar("train_gradients", f"layer_{d}", gradient, iteration=epoch*len(dataset) + len(epoch_loss))
            optimizer.step()
            
            logger.report_scalar("train_loss", "loss", loss.item(), iteration=epoch*len(dataset) + len(epoch_loss))
            optimizer.step()
        print(f'Epoch {epoch} — loss: {sum(epoch_loss)/len(epoch_loss):.4f}')

        if epoch != 0 and epoch % 10 == 0 : # save the model every 10 epochs
            torch.save(model.state_dict(), f'ckpts/model_latent_{epoch}.pth')

    torch.save(model.state_dict(), 'model_latent.pth')

def main() :
    data_torch = torch.load('/home/thibault/VAE/vae_sigloss/latents/mnist_latents.pt')
    print(data_torch.shape)
    
    flow_model = MNISTGeneratorLatent(latent_dim=128)
    train_loop(flow_model, data_torch, epochs=31, lr=1e-4, bs=64)


if __name__ == '__main__' :
    torch.manual_seed(42)
    random.seed(42)

    task = Task.init(
        project_name="VAE-MNIST",
        task_name="flow matching residual",
        auto_connect_frameworks={"pytorch": True},
        output_uri="https://files.clear.ml",  # use ClearML's own file server
    )
    logger = task.get_logger()

    main()
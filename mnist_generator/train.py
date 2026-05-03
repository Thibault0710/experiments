import random
from model import MNISTGenerator
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from clearml import Task

import logging
logging.getLogger("clearml").setLevel(logging.DEBUG)

SHIFT=5.0

def shift(ts, shift) :
    return shift*ts / (shift*ts + (1-ts)) 

def step(model, x) :
    """
    x of shape (B, H, W)
    """
    B = x.shape[0]

    eps = torch.randn_like(x)
    ts  = shift(torch.rand(B)[:, None, None], SHIFT) * torch.ones_like(x) 
    #ts = torch.rand(B)[:, None, None] * torch.ones_like(x) # maybe apply a shift later

    xt     = (1-ts)*x + ts*eps # x + (eps - x)*ts
    v_pred = model.forward(xt, ts[:,0,0].unsqueeze(-1))

    return v_pred, eps - x

def inference(model, x, steps, t_start=1, t_end=0) :
    # We denoise using euler simple
    
    ts = torch.linspace(t_start, t_end, steps)
    ts = shift(ts, 5.0)
    for step in range(steps-1) :
        v_pred = model(x, ts[step].unsqueeze(0))
        x = x - (ts[step] - ts[step+1])*v_pred
    
    return torch.clamp((x+1)/2, 0, 1)

def mask_loss(pred, gt) :
    full = (pred - gt)**2 
    masked = (full < 1).float() * full
    return masked.mean()
    
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

            v_pred, v_gt = step(model, x.reshape(-1, 28, 28))
            l = torch.nn.functional.mse_loss(v_pred, v_gt, reduction='mean') # maybe apply a mask later
            #l = mask_loss(v_pred, v_gt)
            loss.append(l.item())
            epoch_loss.append(l.item())
            l.backward()
            gradients = model.get_gradients_updates()
            for d, gradient in enumerate(gradients) :
                logger.report_scalar("train_gradients", f"layer_{d}", gradient, iteration=epoch*len(dataset) + len(epoch_loss))
            optimizer.step()
            
            logger.report_scalar("train_loss", "loss", l.item(), iteration=epoch*len(dataset) + len(epoch_loss))


        # We run one inference at the end of each epoch to see how the model is doing
        with torch.no_grad() :
            x = torch.randn(16, 28, 28)
            x = inference(model, x, steps=100)
        
        imgs.append(x[0].detach())
        # 
        # img = x[0].detach().cpu().numpy()
        # plt.figure()
        # plt.imshow(img, cmap="gray")
        # plt.axis("off")
        # plt.savefig(f"logs/samples_epoch_{epoch}.png", bbox_inches="tight", pad_inches=0)
        # plt.close()

        print(f'Epoch {epoch} — loss: {sum(epoch_loss)/len(epoch_loss):.4f}')

        if epoch != 0 and epoch % 10 == 0 : # save the model every 10 epochs
            torch.save(model.state_dict(), f'ckpts/model_{epoch}.pth')

    torch.save(model.state_dict(), 'model.pth')

def main() :
    data = pd.read_csv('/home/thibault/NN/MNIST/MNIST2/mnist_train.csv').iloc[:,1:]
    print(data.shape)
    data_torch = torch.tensor(data.to_numpy(), dtype=torch.float32) / 255.0 * 2 - 1
    
    flow_model = MNISTGenerator()
    # flow_model.load_state_dict(torch.load('model.pth'))
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
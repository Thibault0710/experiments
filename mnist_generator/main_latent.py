import torch
from train_latent import inference
import matplotlib.pyplot as plt
from latent_model import MNISTGeneratorLatent
import pandas as pd
import os
import sys
sys.path.append('/home/thibault/VAE/')
from vae_sigloss.vae import VAEModele

def inference_half(model, vae) :
    # we do an inference noising at half an existing sample, then denoising it to see how the model is doing
    data = pd.read_csv('/home/thibault/NN/MNIST/MNIST2/mnist_train.csv').iloc[:,1:]
    print(data.shape)
    data_torch = torch.tensor(data.to_numpy(), dtype=torch.float32) / 255.0
    x = vae.encoder(data_torch[0]).unsqueeze(0)[..., :128] # we take only the mean
    x0 = x.clone()
    sigma = 0.5
    x = (1-sigma)*x + sigma*torch.randn_like(x)
    print("x.shape: ", x.shape)
    x = inference(model, vae, x, steps=100, t_start=sigma, t_end=0.0)
    return x, vae.decoder(x0).reshape(-1, 28, 28)

def main() :
    ckpt_path = 'model_latent.pth'
    model = MNISTGeneratorLatent(latent_dim=128) # the fm model
    model.load_state_dict(torch.load(ckpt_path))

    vae_model = VAEModele()
    vae_model.load_state_dict(torch.load('/home/thibault/VAE/vae_sigloss/ckpt/vae_final.pth'))

    # Sample from the model
    with torch.no_grad() :
        x = torch.randn(16, 128)
        x = inference(model, vae_model, x, steps=1000)
    # Plot the results
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16) :
        ax = axes[i//4, i%4]
        ax.imshow(x[i].cpu().numpy(), cmap='gray')
        ax.axis('off')
    #plt.show()
    plt.savefig("logs/generated.png")

    # inference from half noised sample next to fresh sample
    x, x0 = inference_half(model, vae_model)
    x = x.detach().cpu().numpy()
    x0 = x0.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x0[0], cmap='gray')
    axes[0].set_title("Original Sample")
    axes[0].axis('off')
    axes[1].imshow(x[0], cmap='gray')
    axes[1].set_title("Denoised Sample")
    axes[1].axis('off')
    #plt.show()
    plt.savefig("logs/half_noised.png")

if __name__ == "__main__" :
    main()



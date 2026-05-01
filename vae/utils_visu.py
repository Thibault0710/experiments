import matplotlib.pyplot as plt
import torch

def test_reconstruction(model, dataloader, index=0, device=None, save_path="reconstruction.png"):
    dataset = dataloader.dataset
    x = dataset[index]
    if isinstance(x, tuple):
        x = x[0]
    x_flat = x.view(1, -1).to(device)
    with torch.no_grad():
        x_recon = model.forward(x_flat, sample=False)
    x_img       = x_flat.cpu().view(28, 28)
    x_recon_img = x_recon.cpu().view(28, 28)
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(x_img, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(x_recon_img, cmap='gray')
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def random_sampling(model, num_samples=10, device=None, save_path="random_samples.png"):
    with torch.no_grad():
        z_random = torch.randn(num_samples, model.latent_dim).to(device)
        x_random_recon = model.decoder(z_random)
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples*2, 2))
    for i in range(num_samples):
        x_random_img = x_random_recon[i].cpu().view(28, 28)
        axes[i].imshow(x_random_img, cmap='gray')
        axes[i].set_title(f"Sample {i+1}")
        axes[i].axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
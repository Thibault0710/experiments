import torch
from train import inference
import matplotlib.pyplot as plt
from model import MNISTGenerator
import pandas as pd

def inference_half(model) :
    # we do an inference noising at half an existing sample, then denoising it to see how the model is doing
    data = pd.read_csv('/home/thibault/NN/MNIST/MNIST2/mnist_train.csv').iloc[:,1:]
    print(data.shape)
    data_torch = torch.tensor(data.to_numpy(), dtype=torch.float32) / 255.0 * 2 - 1
    x = data_torch[0].reshape(1, 28, 28)
    x0 = x.clone()
    sigma = 0.5
    x = (1-sigma)*x + sigma*torch.randn_like(x)
    x = inference(model, x, steps=100, t_start=sigma, t_end=0.0)
    return x, x0

def main() :
    ckpt_path = 'model_good.pth'
    model = MNISTGenerator()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # Sample from the model
    with torch.no_grad() :
        x = torch.randn(16, 28, 28)
        x = inference(model, x, steps=1000)
    # Plot the results
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(16) :
        ax = axes[i//4, i%4]
        ax.imshow(x[i].cpu().numpy(), cmap='gray')
        ax.axis('off')
    plt.savefig("logs/generated.png")

    # inference from half noised sample next to fresh sample
    x, x0 = inference_half(model)
    x = x.detach().cpu().numpy()
    x0 = x0.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(x0[0], cmap='gray')
    axes[0].set_title("Original Sample")
    axes[0].axis('off')
    axes[1].imshow(x[0], cmap='gray')
    axes[1].set_title("Denoised Sample")
    axes[1].axis('off')
    plt.savefig("logs/half_noised.png")

if __name__ == "__main__" :
    main()

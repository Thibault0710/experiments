# Flow Matching on MNIST

## Description
This repo implements flow matching models to generate MNIST images.

Two approaches are explored:
- Using a VAE (from the `vae` folder)
- Direct generation in pixel space

Both models have comparable architectures and number of parameters, and are not conditioned on the digit.

## Results
- Visually, the VAE-based approach performs significantly better than direct pixel-space generation

### Comparison

**Latent space (VAE):**
![Latent Space]
(latent_space/generated.png)

**Pixel space:**
![Pixel Space]
(pixel_space/generated.png)

## TODO
- Compare models using FID scores

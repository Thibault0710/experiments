# Flow Matching on MNIST

## Description
This repo implements flow matching models to generate MNIST images.

Two approaches are explored:
- Using a VAE (from the `vae` folder)
- Direct generation in pixel space

Both models have comparable architectures and number of parameters, and are not conditioned on the digit.

## Results
Visually, the VAE-based approach performs significantly better than direct pixel-space generation.
The results can be seen in the 2 subfolders.

## TODO
- Compare models using FID scores

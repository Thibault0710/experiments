# Flow Matching on MNIST

## Description
This repo implements flow matching models to generate MNIST images.

Two approaches are explored:
- Using a VAE (from the `vae` folder)
- Direct generation in pixel space


## Results
- Visually, the VAE-based approach performs significantly better than direct pixel-space generation

### Comparison

<p align="center">
  <img src="latent_space/generated.png" width="300"/>
  <img src="pixel_space/generated.png" width="300"/>
</p>

<p align="center">
  <b>Left:</b> Latent space (VAE) &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; <b>Right:</b> Pixel space
</p>
The samples are generated using main.py (for pixel space version) and main_latent.py (for the VAE based version).
Both models have comparable architectures and number of parameters, they are not conditioned on the digit. They ve been trained during 30 epochs on 60k samples from MNIST dataset.
We used shifted towards high noise timestep distribution at train and inference time.

## TODO
- Compare models using FID scores.
- Try to vary the beta coefficient of the VAE.
- Add a digit condition.

# VAE vs SigReg VAE

## Description
This project compares a standard VAE (`vae.py`) with a VAE trained using a SigReg loss (`vae_sigloss.py`).

## Method
- One reconstructed sample is saved per epoch
- Results are stored in two separate folders
- Evaluation is done visually

## TODO
- Add FID metric on generated samples
- Clarify and analyze reconstruction loss
- Combine FID + reconstruction analysis

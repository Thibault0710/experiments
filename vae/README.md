# VAE vs SigReg VAE

## Description
This project compares a standard VAE (`vae.py`) with a VAE trained using a SigReg loss (`vae_sigloss.py`).

## Method
- One reconstructed sample is saved per epoch  
- Results are stored in two separate folders  
- Evaluation is done visually  

## Results

<p align="center">
  <b>Classic VAE</b><br>
  <img src="vae_classic_1e-3/vae_random_samples.png" width="900"/>
</p>

<p align="center">
  <b>SigReg VAE</b><br>
  <img src="sigreg_var_5e-3/vae_random_samples.png" width="900"/>
</p>
> ⚠️ Comparison should be interpreted carefully:  
> the KL coefficient (β) and the SigReg coefficient (λ) are not directly comparable.

## TODO
- Add FID metric on generated samples  
- Clarify and analyze reconstruction loss  
- Combine FID + reconstruction analysis  

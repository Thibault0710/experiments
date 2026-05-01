# Flow Matching for Graphs

## Description
Simple experiment on flow matching for graphs.  
The goal is to generate images from a Stable Diffusion dataset using graph convolution.

## Method
Images are represented as graph signals on a grid graph, Node spatial information is injected via concatenation at the start of training.
Tested positional encodings:
  - Laplacian eigenvectors
  - Raw indexing
  - Fourier frequencies
We evaluate the distinction between these methods by overfitting on a sample.
The results are in output subfolder.

## Preliminary results
The injection of node localisation in the model clearly improved the results (see None output vs laplace or Fourier one). 
Fourier method seems the best for now but it is not generalizable as laplacian values are. 

## TODO
- Try Graph Attention
- Add text conditioning (e.g. UMT5) to generate pixels on a random mesh

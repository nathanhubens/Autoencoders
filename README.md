# Autoencoders

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nathanhubens/Autoencoders)

**Autoencoders (AE)** are neural networks that aims to copy their inputs to their outputs. They work by compressing the input into a latent-space representation, and then reconstructing the output from this representation. This kind of network is composed of two parts :

1. **Encoder**: This is the part of the network that compresses the input into a latent-space representation. It can be represented by an encoding function _h=f(x)_.
2. **Decoder**: This part aims to reconstruct the input from the latent space representation. It can be represented by a decoding function _r=g(h)_.

<img src="https://nathanhubens.github.io/posts/images/autoencoders/AE.png" alt="drawing" width="750"/>

This notebook show the implementation of five types of autoencoders :

* Vanilla Autoencoder
* Multilayer Autoencoder
* Convolutional Autoencoder
* Regularized Autoencoder
* Variational Autoencoder

The explanation of each (except VAE) can be found [here](https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f)


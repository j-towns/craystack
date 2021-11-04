__Informal disclaimer:__ Craystack is experimental software for prototyping and research in lossless compression. We will change (and probably break) parts of Craystack, and do not make any guarantees about API stability.

# Craystack [![DOI](https://zenodo.org/badge/155861648.svg)](https://zenodo.org/badge/latestdoi/155861648) [![craystack](https://github.com/j-towns/craystack/actions/workflows/tests.yml/badge.svg)](https://github.com/j-towns/craystack/actions/workflows/tests.yml)

The basic building blocks for doing lossless compression are

1. Data to compress.
2. A (probabilistic, generative) model for the data. You'll need to be able to
   evaluate the likelihood (or ELBO) of the data under this model so no GANs
   allowed!
3. Software to do the compression, given data and a model.

Craystack is software to do compression, given data and model.

Craystack is based on **codecs**. In Craystack a codec is a pair of functions,
one for mapping data to a compressed representation (encoding) and one for
mapping back again (decoding). Simple codecs are provided, along with codec
builders for various kinds of model.

The codecs in Craystack are 

 - **Composable**, meaning you can combine and mix them to code different
   combinations of data types, according to sophisticated models, in parallel or
   in series.
 - **Stacky**, meaning last-in-first-out (LIFO). The last item that an encoder
   compresses will be the first item that a decoder decompresses.

The core of Craystack is a vectorized version of Asymmetric Numeral Systems (ANS),
implemented using NumPy. ANS is a last-in-first-out (i.e. stack-like) entropy 
coding method, invented by Jarek Duda. The vectorized method is based
on a [paper](https://arxiv.org/abs/1402.3392), and [accompanying code](
https://github.com/rygorous/ryg_rans), by Fabian Giesen. Jamie has also written
an accessible [tutorial on ANS](https://arxiv.org/abs/2001.09186), with
a pedagogical implementation at https://github.com/j-towns/ans-notes.

# Installation
To install craystack:

```bash
git clone git@github.com:j-towns/craystack.git
cd craystack
pip install -e .
```

Then to run the tests do

```bash
pytest
```

you may have to install pytest with

```bash
pip install pytest
```
# Example: compress the MNIST test set in a few seconds using BB-ANS
Note: this requires pytorch to be installed.

In the examples directory, run `python binary_mnist_vae.py mnist_data`. This
compresses the MNIST test set using a small VAE model, using the BB-ANS
algorithm. This uses the [VAE codec](craystack/bb_ans.py#L39), and runs in a
few seconds thanks to the vectorization of craystack.

# Authors
Craystack was written by [Jamie Townsend](https://j-towns.github.io), [Tom Bird](https://tom-bird.github.io/) and [Julius Kunze](https://juliuskunze.com/). If you use Craystack in your research, please cite [this paper](https://openreview.net/forum?id=r1lZgyBYwS). Bibtex:

```
@inproceedings{
townsend2020hilloc,
title={Hi{\{}LL{\}}oC: lossless image compression with hierarchical latent variable models},
author={James Townsend and Thomas Bird and Julius Kunze and David Barber},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=r1lZgyBYwS}
}
```

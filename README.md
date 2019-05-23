# Craystack
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

In the examples directory, run `python binary_mnist_vae.py`. This compresses the MNIST test set using a small VAE model, using the BB-ANS algorithm. This uses the [VAE codec](craystack/bb_ans.py#L39), and runs in a few seconds thanks to the vectorization of craystack.

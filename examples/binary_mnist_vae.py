import torch
import numpy as np
from autograd.builtins import tuple as ag_tuple
import craystack as cs
from torch_vae import BinaryVAE
from torch_util import torch_fun_to_numpy_fun
from torchvision import datasets
import craystack.bb_ans as bb_ans
import time

rng = np.random.RandomState(0)

prior_precision = 8
bernoulli_precision = 16
q_precision = 14

num_images = 10000
num_pixels = num_images * 784
batch_size = 10
assert num_images % batch_size == 0
num_batches = num_images // batch_size

latent_dim = 40
latent_shape = (batch_size, latent_dim)
latent_size = np.prod(latent_shape)
obs_shape = (batch_size, 28 * 28)
obs_size = np.prod(obs_shape)

## Setup codecs
# VAE codec
model = BinaryVAE(hidden_dim=100, latent_dim=40)
model.load_state_dict(torch.load('vae_params'))

rec_net = torch_fun_to_numpy_fun(model.encode)
gen_net = torch_fun_to_numpy_fun(model.decode)

obs_codec = lambda p: cs.Bernoulli(p, bernoulli_precision)

def vae_view(head):
    return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                     np.reshape(head[latent_size:], obs_shape)))

vae_append, vae_pop = cs.repeat(cs.substack(
    bb_ans.VAE(gen_net, rec_net, obs_codec, prior_precision, q_precision),
    vae_view), num_batches)

# Codec for adding extra bits to the start of the chain (necessary for bits
# back).
other_bits_append, _ = cs.substack(cs.Uniform(q_precision), lambda h: vae_view(h)[0])

## Load mnist images
images = datasets.MNIST('mnist', train=False, download=True).data.numpy()
images = np.uint64(rng.random_sample(np.shape(images)) < images / 255.)
images = np.split(np.reshape(images, (num_images, -1)), num_batches)

## Encode
# Initialize message with some 'extra' bits
encode_t0 = time.time()
init_message = cs.empty_message(obs_size + latent_size)

# Enough bits to pop a single latent
other_bits = rng.randint(1 << q_precision, size=latent_shape, dtype=np.uint64)
init_message = other_bits_append(init_message, other_bits)

# Encode the mnist images
message = vae_append(init_message, images)

flat_message = cs.flatten(message)
encode_t = time.time() - encode_t0

print("All encoded in {:.2f}s.".format(encode_t))

message_len = 32 * len(flat_message)
print("Used {} bits.".format(message_len))
print("This is {:.2f} bits per pixel.".format(message_len / num_pixels))


## Decode
decode_t0 = time.time()
message = cs.unflatten(flat_message, obs_size + latent_size)

message, images_ = vae_pop(message)
decode_t = time.time() - decode_t0

print('All decoded in {:.2f}s.'.format(decode_t))

np.testing.assert_equal(images, images_)
np.testing.assert_equal(message, init_message)

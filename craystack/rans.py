"""
Vectorized implementation of rANS based on https://arxiv.org/abs/1402.3392
"""
from warnings import warn

import numpy as np


rng = np.random.default_rng(0)
rans_l = 1 << 31  # the lower bound of the normalisation interval
atleast_1d = lambda x: np.atleast_1d(x).astype('uint64')

def base_message(shape, randomize=False):
    """
    Returns a base ANS message of given shape. If randomize=True,
    populates the lower bits of the head with samples from a Bernoulli(1/2)
    distribution. The tail is empty.
    """
    assert shape and np.prod(shape), 'Shape must be an int > 0' \
                                     'or tuple with length > 0.'
    head = np.full(shape, rans_l, "uint64")
    if randomize:
        head += rng.integers(0, rans_l, size=shape, dtype='uint64')
    return head, ()

def stack_extend(stack, arr):
    return arr, stack

def stack_slice(stack, n):
    slc = []
    while n > 0:
        if stack:
            arr, stack = stack
        else:
            warn('Popping from empty message. Generating random data.')
            arr, stack = rng.integers(1 << 32, size=n, dtype='uint32'), ()
        if n >= len(arr):
            slc.append(arr)
            n -= len(arr)
        else:
            slc.append(arr[:n])
            stack = arr[n:], stack
            break
    return stack, np.concatenate(slc)

def push(x, starts, freqs, precisions):
    starts, freqs, precisions = map(atleast_1d, (starts, freqs, precisions))
    head, tail = x
    # assert head.shape == starts.shape == freqs.shape
    idxs = head >= ((rans_l >> precisions) << 32) * freqs
    if np.any(idxs):
        tail = stack_extend(tail, np.uint32(head[idxs]))
        head = np.copy(head)  # Ensure no side-effects
        head[idxs] >>= 32
    head_div_freqs, head_mod_freqs = np.divmod(head, freqs)
    return (head_div_freqs << precisions) + head_mod_freqs + starts, tail

def pop(x, precisions):
    precisions = atleast_1d(precisions)
    head_, tail_ = x
    cfs = head_ & ((1 << precisions) - 1)
    def pop(starts, freqs):
        starts, freqs = map(atleast_1d, (starts, freqs))
        head = freqs * (head_ >> precisions) + cfs - starts
        idxs = head < rans_l
        n = np.sum(idxs)
        if n > 0:
            tail, new_head = stack_slice(tail_, n)
            head[idxs] = (head[idxs] << 32) | new_head
        else:
            tail = tail_
        return head, tail
    return cfs, pop

def flatten(x):
    """Flatten a vrans state x into a 1d numpy array."""
    head, x = np.ravel(x[0]), x[1]
    out = [np.uint32(head >> 32), np.uint32(head)]
    while x:
        head, x = x
        out.append(head)
    return np.concatenate(out)

def unflatten(arr, shape):
    """Unflatten a 1d numpy array into a vrans state."""
    size = np.prod(shape)
    head = np.uint64(arr[:size]) << 32 | np.uint64(arr[size:2 * size])
    return np.reshape(head, shape), (arr[2 * size:], ())

def message_equal(message1, message2):
    return np.all(flatten(message1) == flatten(message2))

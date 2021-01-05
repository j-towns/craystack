"""
Vectorized implementation of rANS based on https://arxiv.org/abs/1402.3392
"""

import numpy as np


rans_l = 1 << 31  # the lower bound of the normalisation interval

def base_message(shape, randomize=False):
    """
    Returns a base ANS message of given shape. If randomize=True,
    populates the lower bits of the head with samples from a Bernoulli(1/2)
    distribution. The tail is empty.
    """
    head = np.full(shape, rans_l, "uint64")
    if randomize:
        head += np.random.randint(0, rans_l, size=shape, dtype='uint64')
    return (head, ())

def stack_extend(stack, arr):
    return arr, stack

def stack_slice(stack, n):
    slc = []
    shape = stack[0].shape
    while n > 0:

        if len(stack) < 2:
            # empty pop
            arr, stack = base_message(shape, randomize=True)
        else:
            arr, stack = stack

        if n >= len(arr):
            slc.append(arr)
            n -= len(arr)
        else:
            slc.append(arr[:n])
            stack = arr[n:], stack
            break
    return stack, np.concatenate(slc)

def push(x, starts, freqs, precisions):
    head, tail = x
    # assert head.shape == starts.shape == freqs.shape
    idxs = head >= ((rans_l >> precisions) << 32) * freqs
    if np.any(idxs) > 0:
        tail = stack_extend(tail, np.uint32(head[idxs]))
        head = np.copy(head)  # Ensure no side-effects
        head[idxs] >>= 32
    head_div_freqs, head_mod_freqs = np.divmod(head, freqs)
    return (head_div_freqs << precisions) + head_mod_freqs + starts, tail

def pop(x, precisions):
    head_, tail_ = x
    cfs = head_ & ((1 << precisions) - 1)
    def pop(starts, freqs):
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

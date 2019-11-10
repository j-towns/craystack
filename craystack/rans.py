"""
Vectorized implementation of rANS based on https://arxiv.org/abs/1402.3392
"""

import numpy as np


rans_l = 1 << 31  # the lower bound of the normalisation interval

def empty_message(shape):
    """
    Returns an empty ANS message of given shape.
    """
    return (np.full(shape, rans_l, "uint64"), ())

def cons_list_extend(ls, els):
    return (els, ls) if len(els) else ls

def cons_list_slice(ls, n):
    slc = []
    while n > 0:
        el, ls = ls
        if n >= len(el):
            slc.append(el)
            n -= len(el)
        else:
            slc.append(el[:n])
            ls = el[n:], ls
            break
    return ls, np.concatenate(slc)

def push(x, starts, freqs, precisions):
    head, tail = x
    # assert head.shape == starts.shape == freqs.shape
    idxs = head >= ((rans_l >> precisions) << 32) * freqs
    if np.any(idxs) > 0:
        tail = cons_list_extend(tail, np.uint32(head[idxs]))
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
            tail, new_head = cons_list_slice(tail_, n)
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

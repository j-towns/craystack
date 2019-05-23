import numpy as np


rans_l = 1 << 31  # the lower bound of the normalisation interval

def message_init(shape):
    """
    Returns an empty ANS message of given shape.
    """
    return (np.full(shape, rans_l, "uint64"), ())

def cons_list_extend(ls, els):
    for el in els[::-1]:
        ls = el, ls
    return ls

def cons_list_slice(ls, n):
    if n == 0:
        return ls, np.array([], dtype="uint32")
    slc = []
    while n > 0:
        el, ls = ls
        slc.append(el)
        n -= 1
    return ls, np.asarray(slc)

def push(x, starts, freqs, precisions):
    head, tail = x
    # assert head.shape == starts.shape == freqs.shape
    idxs = head >= ((rans_l >> precisions) << 32) * freqs
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
        tail, new_head = cons_list_slice(tail_, n)
        head[idxs] = (head[idxs] << 32) | new_head
        return head, tail
    return cfs, pop

def flatten(x):
    """Flatten a vrans state x into a 1d numpy array."""
    head, x = np.ravel(x[0]), x[1]
    out = list(np.uint32(head >> 32))
    out.extend(np.uint32(head))
    while x:
        head, x = x
        out.append(head)
    return np.asarray(out)

def unflatten(arr, shape):
    """Unflatten a 1d numpy array into a vrans state."""
    size = np.prod(shape)
    ret = ()
    for head in arr[:2 * size - 1:-1]:
        ret = head, ret
    head = np.uint64(arr[:size]) << 32 | np.uint64(arr[size:2 * size])
    return np.reshape(head, shape), ret

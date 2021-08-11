from functools import reduce
from collections import namedtuple
from jax import tree_util  # TODO: copy-paste the old Python tree_util
                           # implementation to avoid depending on JAX.

import numpy as np

# By defining a reversible, we promise that
#   do(undo(x)) == x
# and
#   undo(do(x)) == x
Reversible = namedtuple('Reversible', ['do', 'undo'])

def compose(step1, step2):
    def do(x):
        return step2.do(step1.do(x))
    def undo(x):
        return step1.undo(step2.undo(x))
    return Reversible(do, undo)

def invert(r):
    return Reversible(do=r.undo, undo=r.do)

# keep_left: (x -> y <-> z) -> (x, y) <-> (x, z)
def keep_left(fun):
    def do(xy):
        x, y = xy
        z = fun(x).do(y)
        return x, z

    def undo(xz):
        x, z = xz
        y = fun(x).undo(z)
        return x, y
    return Reversible(do, undo)

# keep_right: (x -> y <-> z) -> (y, x) <-> (z, x)
def keep_right(fun):
    return serial([swap, keep_left(fun), swap])

def restructure(before_tree, after_tree):
    before_flat, before_tree  = tree_util.tree_flatten(before_tree)
    after_flat,  after_tree   = tree_util.tree_flatten(after_tree)
    # Check that before and after have no repeats and contain the same elements
    # as one-another.
    assert len(before_flat) == len(set(before_flat))
    assert len(after_flat)  == len(set(after_flat))
    assert sorted(before_flat) == sorted(after_flat)

    def argsort(l):
        return [i for i, _ in sorted(enumerate(l), key=lambda ix: ix[1])]

    before_perm = argsort(before_flat)
    after_perm  = argsort(after_flat)

    do_perm   = [before_perm[i] for i in argsort(after_perm) ]
    undo_perm = [after_perm[i]  for i in argsort(before_perm)]

    def do(before):
        before_flat, before_tree_ = tree_util.tree_flatten(before)
        assert before_tree == before_tree_
        return after_tree.unflatten([before_flat[i] for i in do_perm])

    def undo(after):
        after_flat, after_tree_ = tree_util.tree_flatten(after)
        assert after_tree == after_tree_
        return before_tree.unflatten([after_flat[i] for i in undo_perm])

    return Reversible(do, undo)

# swap: (x, y) <-> (y, x)
swap = restructure(('x', 'y'), ('y', 'x'))

def serial(steps):
    return reduce(compose, steps)

def split(indices_or_sections, axis=0):
    def do(arr):
        return np.split(arr, indices_or_sections, axis=axis)
    def undo(arrs):
        return np.concatenate(arrs, axis=axis)
    return Reversible(do, undo)

def reshape(before_shape, after_shape):
    def do(arr):
        assert arr.shape == before_shape
        return np.reshape(arr, after_shape)
    def undo(arr):
        assert arr.shape == after_shape
        return np.reshape(arr, before_shape)
    return Reversible(do, undo)

def bb_ans(prior, likelihood, posterior):
    # We could adopt a convention that decoders are assigned to 'do', encoders
    # to 'undo', so
    #   prior:      message <-> (message, z)
    #   likelihood: z -> message <-> (message, x)
    #   posterior:  x -> message <-> (message, z)
    return serial([
        # message
        prior,
        # (message, z)
        keep_right(likelihood),
        # ((message, x), z)
        restructure((('message', 'x'), 'z'), (('message', 'z'), 'x')),
        # ((message, z), x)
        keep_right(lambda x: invert(posterior(x))),
        # (message, x)
    ])

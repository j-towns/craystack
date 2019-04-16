import numpy as np
import numpy.random as rng
import pytest

import craystack as cs
import craystack.vectorans as vrans
import craystack.codecs as codecs


def check_codec(head_shape, codec, data):
    message = vrans.x_init(head_shape)
    append, pop = codec
    message_ = append(message, data)
    message_, data_ = pop(message_)
    assert_message_equal(message, message_)
    np.testing.assert_equal(data, data_)


def test_uniform():
    precision = 4
    shape = (2, 3, 5)
    data = rng.randint(precision, size=shape, dtype="uint64")
    check_codec(shape, codecs.Uniform(precision), data)


def test_benford():
    shape = (2, 3, 5)
    data_len = rng.randint(31, 63, shape, dtype="uint64")
    data = ((1 << data_len) | (rng.randint(1 << 63, size=shape, dtype="uint64")
                               & ((1 << data_len) - 1)))
    check_codec(shape, codecs.Benford64, data)


def test_benford_inverse():
    shape = (2, 3, 5)
    data_len = rng.randint(31, 63, shape, dtype="uint64")
    data = ((1 << data_len) | (rng.randint(1 << 63, size=shape, dtype="uint64")
                               & ((1 << data_len) - 1)))
    check_codec(shape, codecs.Benford64, data)


def test_repeat():
    precision = 4
    n_data = 7
    shape = (2, 3, 5)
    data = rng.randint(precision, size=(n_data,) + shape, dtype="uint64")
    check_codec(shape, cs.repeat(codecs.Uniform(precision), n_data), data)


def test_substack():
    n_data = 100
    prec = 4
    head, tail = vrans.x_init((4, 4))
    head = np.split(head, 2)
    message = head, tail
    data = rng.randint(1 << prec, size=(n_data, 2, 4), dtype='uint64')
    view_fun = lambda h: h[0]
    append, pop = cs.substack(cs.repeat(codecs.Uniform(prec), n_data), view_fun)
    message_ = append(message, data)
    np.testing.assert_array_equal(message_[0][1], message[0][1])
    message_, data_ = pop(message_)
    np.testing.assert_equal(message, message_)
    np.testing.assert_equal(data, data_)

def test_parallel():
    precs = [1, 2, 4, 8, 16]
    szs = [2, 3, 4, 5, 6]
    u_codecs = [codecs.Uniform(p) for p in precs]
    view_fun = lambda slc: lambda head: head[slc]
    view_funs = []
    start = 0
    for s in szs:
        view_funs.append(view_fun(slice(start, start + s)))
        start += s
    data = [rng.randint(1 << p, size=size, dtype='uint64')
            for p, size in zip(precs, szs)]
    check_codec(sum(szs), cs.parallel(u_codecs, view_funs), data)


def test_bernoulli():
    precision = 4
    shape = (2, 3, 5)
    p = rng.random(shape)
    data = np.uint64(rng.random(shape) < p)
    check_codec(shape, codecs.Bernoulli(p, precision), data)


def test_categorical():
    precision = 4
    shape = (2, 3, 5)
    ps = rng.random((np.prod(shape), 4))
    ps = ps / np.sum(ps, axis=-1, keepdims=True)
    data = np.reshape([rng.choice(4, p=p) for p in ps], shape)
    ps = np.reshape(ps, shape + (4,))
    check_codec(shape, codecs.Categorical(ps, precision), data)


def test_logistic():
    precision = 16
    batch_size = 4
    means = rng.uniform(0, 1, batch_size)
    log_scale = rng.randn() - 4
    # type is important!
    data = np.array([rng.choice(256) for _ in range(batch_size)]).astype('uint64')
    check_codec((batch_size,), codecs.Logistic(means, log_scale, precision), data)


def test_logistic_mixture():
    precision = 12
    batch_size = 2
    nr_mix = 10
    shape = (batch_size, nr_mix)
    means, log_scales, logit_probs = rng.randn(*shape), rng.randn(*shape), rng.randn(*shape)
    means = means + 100
    log_scales = log_scales - 10
    theta = np.concatenate((means, log_scales, logit_probs), axis=-1)
    # type is important!
    data = np.array([rng.choice(256) for _ in range(batch_size)]).astype('uint64')
    check_codec((shape[0],), codecs.LogisticMixture(theta, precision), data)


def test_autoregressive():
    precision = 8
    batch_size = 3
    data_size = 10
    choices = 8
    data = np.array([rng.choice(choices) for _ in range(batch_size * data_size)])
    data = np.reshape(data, (batch_size, data_size))
    fixed_probs = rng.random((batch_size, data_size, choices))
    fixed_probs = fixed_probs / np.sum(fixed_probs, axis=-1, keepdims=True)
    elem_idxs = [(slice(None), i) for i in range(10)]  # slice for the batch dimension
    elem_codec = lambda p, idx: codecs.Categorical(p, precision)
    check_codec((batch_size,),
                codecs.AutoRegressive(lambda *x: fixed_probs,
                                  (batch_size, data_size,),
                                  fixed_probs.shape,
                                  elem_idxs,
                                  elem_codec),
                data)

def test_gaussian_db():
    bin_precision = 8
    coding_precision = 12
    batch_size = 5

    bin_means = rng.randn()
    bin_stdds = np.exp(rng.randn()/10)

    # if the gaussian distributions have little overlap then will
    # get zero freq errors
    means = bin_means + rng.randn()/10
    stdds = bin_stdds * np.exp(rng.randn()/10.)

    data = np.array([rng.choice(1 << bin_precision) for _ in range(batch_size)])

    check_codec((batch_size,),
                codecs.DiagGaussianLatent(means, stdds, bin_means, bin_stdds,
                                      coding_precision, bin_precision),
                data)

def test_flatten_unflatten_benford():
    n = 100
    shape = (7, 3)
    p = 12
    state = vrans.x_init(shape)
    some_bits = rng.randint(1 << p, size=(n,) + shape).astype(np.uint64)
    freqs = np.ones(shape, dtype="uint64")
    for b in some_bits:
        state = vrans.append(state, b, freqs, p)
    flat = codecs.flatten_benford(state)
    flat_ = vrans.flatten(state)
    print('Normal flat len: {}'.format(len(flat_) * 32))
    print('Benford flat len: {}'.format(len(flat) * 32))
    assert flat.dtype is np.dtype("uint32")
    state_ = codecs.unflatten_benford(flat, shape)
    flat_ = codecs.flatten_benford(state_)
    assert np.all(flat == flat_)
    assert np.all(state[0] == state_[0])
    assert state[1] == state_[1]


def assert_message_equal(message1, message2):
    np.testing.assert_equal(message1, message2)


@pytest.mark.parametrize('old_size', [141, 32, 17, 6, 3])
@pytest.mark.parametrize('new_size', [141, 32, 17, 6, 3])
def test_resize_head_1d(old_size, new_size, depth=1000):
    old_shape = (old_size,)

    np.random.seed(0)
    p = 8
    bits = np.random.randint(1 << p, size=(depth,) + old_shape, dtype=np.uint64)

    message = vrans.x_init(old_shape)

    other_bits_append, _ = cs.repeat(codecs.Uniform(p), depth)

    message = other_bits_append(message, bits)

    resized = codecs.resize_head_1d(message, new_size)
    reconstructed = codecs.resize_head_1d(resized, old_size)

    init_head, init_tail = message
    recon_head, recon_tail = reconstructed
    np.testing.assert_equal(init_head, recon_head)
    while init_tail:
        el, init_tail = init_tail
        el_, recon_tail = recon_tail
        assert el == el_


import numpy as np
import numpy.random as rng
from scipy.special import expit, logit
import pytest

import craystack as cs


def check_codec(head_shape, codec, data):
    message = cs.empty_message(head_shape)
    push, pop = codec
    message_ = push(message, data)
    message_, data_ = pop(message_)
    assert_message_equal(message, message_)
    np.testing.assert_equal(data, data_)


def test_uniform():
    precision = 4
    shape = (2, 3, 5)
    data = rng.randint(precision, size=shape, dtype="uint64")
    check_codec(shape, cs.Uniform(precision), data)


def test_benford():
    shape = (2, 3, 5)
    data_len = rng.randint(31, 63, shape, dtype="uint64")
    data = ((1 << data_len) | (rng.randint(1 << 63, size=shape, dtype="uint64")
                               & ((1 << data_len) - 1)))
    check_codec(shape, cs.Benford64, data)


def test_benford_inverse():
    shape = (2, 3, 5)
    data_len = rng.randint(31, 63, shape, dtype="uint64")
    data = ((1 << data_len) | (rng.randint(1 << 63, size=shape, dtype="uint64")
                               & ((1 << data_len) - 1)))
    check_codec(shape, cs.Benford64, data)


def test_repeat():
    precision = 4
    n_data = 7
    shape = (2, 3, 5)
    data = rng.randint(precision, size=(n_data,) + shape, dtype="uint64")
    check_codec(shape, cs.repeat(cs.Uniform(precision), n_data), data)


def test_substack():
    n_data = 100
    prec = 4
    head, tail = cs.empty_message((4, 4))
    head = np.split(head, 2)
    message = head, tail
    data = rng.randint(1 << prec, size=(n_data, 2, 4), dtype='uint64')
    view_fun = lambda h: h[0]
    append, pop = cs.substack(cs.repeat(cs.Uniform(prec), n_data), view_fun)
    message_ = append(message, data)
    np.testing.assert_array_equal(message_[0][1], message[0][1])
    message_, data_ = pop(message_)
    np.testing.assert_equal(message, message_)
    np.testing.assert_equal(data, data_)


def test_parallel():
    precs = [1, 2, 4, 8, 16]
    szs = [2, 3, 4, 5, 6]
    u_codecs = [cs.Uniform(p) for p in precs]
    view_fun = lambda slc: lambda head: head[slc]
    view_funs = []
    start = 0
    for s in szs:
        view_funs.append(view_fun(slice(start, start + s)))
        start += s
    data = [rng.randint(1 << p, size=size, dtype='uint64')
            for p, size in zip(precs, szs)]
    check_codec(sum(szs), cs.parallel(u_codecs, view_funs), data)


def test_serial(precision=32):
    shape = (2, 3, 4)
    data1 = rng.randint(precision, size=(7,) + shape, dtype="uint64")
    data2 = rng.randint(2 ** 31, 2 ** 63, size=(5,) + shape, dtype="uint64")
    data = list(data1) + list(data2)

    check_codec(shape, cs.serial([cs.Uniform(precision) for _ in data1] +
                                 [cs.Benford64 for _ in data2]), data)


@pytest.mark.parametrize('shape2', [(1, 6, ), (1, 5, ), (1, 4, )])
def test_serial_resized(shape2, shape1=(5, ), precision=4):
    data1 = rng.randint(precision, size=(7,) + shape1, dtype="uint64")
    data2 = rng.randint(precision, size=(20,) + shape2, dtype="uint64")
    data = list(data1) + list(data2)

    codec = cs.Uniform(precision)
    push, pop = codec

    def push_resize(message, symbol):
        assert message[0].shape == shape2
        message = cs.reshape_head(message, shape1)
        message = push(message, symbol)
        return message

    def pop_resize(message):
        assert message[0].shape == shape1
        message, symbol = pop(message)
        message = cs.reshape_head(message, shape2)
        return message, symbol

    resize_codec = cs.Codec(push_resize, pop_resize)

    check_codec(shape2, cs.serial([codec for _ in data1[:-1]] +
                                  [resize_codec] +
                                  [codec for _ in data2]), data)

def test_bernoulli():
    precision = 4
    shape = (2, 3, 5)
    p = rng.random(shape)
    data = np.uint64(rng.random(shape) < p)
    check_codec(shape, cs.Bernoulli(p, precision), data)


def test_categorical():
    precision = 4
    shape = (2, 3, 5)
    ps = rng.random((np.prod(shape), 4))
    ps = ps / np.sum(ps, axis=-1, keepdims=True)
    data = np.reshape([rng.choice(4, p=p) for p in ps], shape)
    ps = np.reshape(ps, shape + (4,))
    check_codec(shape, cs.Categorical(ps, precision), data)


def test_logistic():
    rng = np.random.RandomState(0)
    coding_precision = 16
    bin_precision = 8
    batch_size = 4
    means = rng.uniform(0, 1, batch_size)
    log_scale = rng.randn()
    # type is important!
    data = np.array([rng.choice(256) for _ in range(batch_size)]).astype('uint64')
    check_codec((batch_size,), cs.Logistic_UnifBins(means, log_scale,
                                                        coding_precision, bin_precision,
                                                        bin_lb=-0.5, bin_ub=0.5),
                data)

def test_discretized():
    rng = np.random.RandomState()
    coding_prec = 16
    bin_prec = 8
    n = 10000
    data = rng.randint(1 << bin_prec, size=10000)
    check_codec((n,),
                cs.codecs._discretize(expit, logit, -.5, .5, bin_prec, coding_prec),
                data)

def test_logistic_mixture():
    precision = 12
    batch_size = 2
    nr_mix = 10
    shape = (batch_size, nr_mix)
    means, log_scales, logit_probs = rng.randn(*shape), rng.randn(*shape), rng.randn(*shape)
    means = means + 100
    log_scales = log_scales - 10
    # type is important!
    data = np.array([rng.choice(256) for _ in range(batch_size)]).astype('uint64')
    check_codec((shape[0],), cs.LogisticMixture_UnifBins(means, log_scales, logit_probs,
                                                             precision, bin_prec=8, bin_lb=-1., bin_ub=1.), data)


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
    elem_codec = lambda p, idx: cs.Categorical(p, precision)
    check_codec((batch_size,),
                cs.AutoRegressive(lambda *x: fixed_probs,
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
    bin_stdds = np.exp(rng.randn() / 10)

    # if the gaussian distributions have little overlap then will
    # get zero freq errors
    means = bin_means + rng.randn() / 10
    stdds = bin_stdds * np.exp(rng.randn() / 10.)

    data = np.array([rng.choice(1 << bin_precision) for _ in range(batch_size)])

    check_codec((batch_size,),
                cs.DiagGaussian_GaussianBins(means, stdds, bin_means, bin_stdds,
                                          coding_precision, bin_precision),
                data)


def test_gaussian_ub():
    bin_lb = np.array([-2., -3.])
    bin_ub = np.array([2., 3.])
    n_bins = 1000
    coding_precision = 16
    batch_size = 5

    means = rng.randn(batch_size, 2) / 10
    stdds = np.exp(rng.random((batch_size, 2)) / 2)

    data = np.array([rng.choice(n_bins, 2) for _ in range(batch_size)])

    check_codec((batch_size, 2),
                cs.DiagGaussian_UnifBins(means, stdds, bin_lb, bin_ub,
                                                  coding_precision, n_bins),
                data)


def test_flatten_unflatten():
    n = 100
    shape = (7, 3)
    p = 12
    state = cs.empty_message(shape)
    some_bits = rng.randint(1 << p, size=(n,) + shape).astype(np.uint64)
    freqs = np.ones(shape, dtype="uint64")
    for b in some_bits:
        state = cs.rans.push(state, b, freqs, p)
    flat = cs.flatten(state)
    flat_ = cs.rans.flatten(state)
    print('Normal flat len: {}'.format(len(flat_) * 32))
    print('Benford flat len: {}'.format(len(flat) * 32))
    assert flat.dtype is np.dtype("uint32")
    state_ = cs.unflatten(flat, shape)
    flat_ = cs.flatten(state_)
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

    message = cs.empty_message(old_shape)

    other_bits_push, _ = cs.repeat(cs.Uniform(p), depth)

    message = other_bits_push(message, bits)

    resized = cs.codecs._resize_head_1d(message, new_size)
    reconstructed = cs.codecs._resize_head_1d(resized, old_size)

    init_head, init_tail = message
    recon_head, recon_tail = reconstructed
    np.testing.assert_equal(init_head, recon_head)
    while init_tail:
        el, init_tail = init_tail
        el_, recon_tail = recon_tail
        assert el == el_


@pytest.mark.parametrize('old_shape', [(100,), (1, 23), (2, 4, 5)])
@pytest.mark.parametrize('new_shape', [(100,), (1, 23), (2, 4, 5)])
def test_reshape_head(old_shape, new_shape, depth=1000):
    np.random.seed(0)
    p = 8
    bits = np.random.randint(1 << p, size=(depth,) + old_shape, dtype=np.uint64)

    message = cs.empty_message(old_shape)

    other_bits_push, _ = cs.repeat(cs.Uniform(p), depth)

    message = other_bits_push(message, bits)

    resized = cs.reshape_head(message, new_shape)
    reconstructed = cs.reshape_head(resized, old_shape)

    init_head, init_tail = message
    recon_head, recon_tail = reconstructed
    np.testing.assert_equal(init_head, recon_head)
    while init_tail:
        el, init_tail = init_tail
        el_, recon_tail = recon_tail
        assert el == el_


@pytest.mark.parametrize('shape', [(100,), (1, 23), (2, 4, 5)])
def test_flatten_unflatten(shape, depth=1000):
    np.random.seed(0)
    p = 8
    bits = np.random.randint(1 << p, size=(depth,) + shape, dtype=np.uint64)

    message = cs.empty_message(shape)

    other_bits_push, _ = cs.repeat(cs.Uniform(p), depth)

    message = other_bits_push(message, bits)

    flattened = cs.flatten(message)
    reconstructed = cs.unflatten(flattened, shape)

    init_head, init_tail = message
    recon_head, recon_tail = reconstructed
    np.testing.assert_equal(init_head, recon_head)
    while init_tail:
        el, init_tail = init_tail
        el_, recon_tail = recon_tail
        assert el == el_


def test_flatten_rate():
    rng.seed(0)
    init_size = 500000
    head_size = 250000
    head, tail = cs.random_message(init_size, (head_size,))
    tail_size = len(cs.flatten((np.array([2 ** 31]), tail)))
    tail_diff = init_size - tail_size
    rate = tail_diff / head_size
    assert abs(rate / ((5+31+15.5)/32) - 1) < 0.001

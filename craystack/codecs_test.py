import numpy as np
import numpy.random as rng
from scipy.special import expit, logit
import pytest

import craystack as cs
from craystack import codecs
from craystack import rans


def check_codec(head_shape, codec, data):
    message = cs.base_message(head_shape)
    push, pop = codec
    message_, data_ = pop(*push(message, data))
    assert_message_equal(message, message_)
    np.testing.assert_equal(data, data_)

def check_cdf_inverse(cdf, ppf, input_prec, coder_prec):
    assert cdf(0) == 0
    cfs = np.arange(1 << coder_prec, dtype='uint64')
    symbols = ppf(cfs)
    assert np.all((cdf(symbols) <= cfs) & (cfs < cdf(symbols + 1)))
    assert cdf(1 << input_prec) == 1 << coder_prec

def test_gaussian_cdfs():
    # We test at very high precision in order to expose floating point
    # imprecision in scipy.stats.norm.[cdf/ppf]
    prior_prec = 22
    posterior_prec = 22
    mean = 1
    sigma = 1
    check_cdf_inverse(codecs._gaussian_cdf(
                          mean, sigma, prior_prec, posterior_prec),
                      codecs._gaussian_ppf(
                          mean, sigma, prior_prec, posterior_prec),
                      prior_prec, posterior_prec)

def test_uniform():
    precision = 4
    shape = (2, 3, 5)
    data = rng.randint(precision, size=shape, dtype="uint64")
    check_codec(shape, cs.Uniform(precision), data)

def test_big_uniform():
    shape = (2, 3, 5)
    precision = rng.randint(64, size=shape, dtype="uint64")
    data = rng.randint(1 << 64, size=shape, dtype="uint64") % (1 << precision)
    check_codec(shape, cs.BigUniform(precision), data)

def test_benford_higher_bits():
    data_prec = 3
    prec = 16
    shape = (400,)
    data = rng.randint(1 << data_prec, size=shape, dtype="uint64")
    check_codec(shape, cs.codecs._benford_high_bits(data_prec, prec), data)

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
    data = rng.randint(1 << precision, size=(n_data,) + shape, dtype="uint64")
    check_codec(shape, cs.repeat(cs.Uniform(precision), n_data), data)

def test_substack():
    n_data = 100
    prec = 4
    head, tail = cs.base_message((4, 4))
    head = np.split(head, 2)
    message = head, tail
    data = rng.randint(1 << prec, size=(n_data, 2, 4), dtype='uint64')
    view_fun = lambda h: h[0]
    append, pop = cs.substack(cs.repeat(cs.Uniform(prec), n_data), view_fun)
    message_, = append(message, data)
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

def test_from_iterable(precision=16):
    shape = (2, 3, 4)
    data1 = rng.randint(1 << precision, size=(7,) + shape, dtype="uint64")
    data2 = rng.randint(2 ** 31, 2 ** 63, size=(5,) + shape, dtype="uint64")
    data = list(data1) + list(data2)

    check_codec(shape, cs.from_iterable(
        [cs.Uniform(precision) for _ in data1]
        + [cs.Benford64 for _ in data2]), data)

def test_from_generator_simple():
    def gen_factory():
        yield cs.Uniform(8)
    data = [np.array(rng.randint(8, size=(1,), dtype="uint64"))]
    check_codec((1,), cs.from_generator(gen_factory), data)

def test_from_generator_serial(precision=16):
    shape = (2, 3, 4)
    data1 = rng.randint(1 << precision, size=(7,) + shape, dtype="uint64")
    data2 = rng.randint(2 ** 31, 2 ** 63, size=(5,) + shape, dtype="uint64")
    data = list(data1) + list(data2)

    def gen_factory():
        for _ in range(7):
            yield cs.Uniform(precision)
        for _ in range(5):
            yield cs.Benford64

    check_codec(shape, cs.from_generator(gen_factory), data)

def test_from_generator_dependent():
    shape = (2, 3, 4)
    precs = rng.randint(16, size=shape, dtype="uint64")
    data = rng.randint(1 << precs, size=shape, dtype="uint64")
    data = [precs, data]

    def gen_factory():
        precs = (yield cs.Uniform(16))
        yield cs.Uniform(precs)

    check_codec(shape, cs.from_generator(gen_factory), data)

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

def test_categorical_new():
    rng = np.random.RandomState(2)
    precision = 4
    shape = (20, 3, 5)
    weights = rng.random((np.prod(shape), 4)) + 1
    ps = weights / np.sum(weights, axis=-1, keepdims=True)
    data = np.reshape([rng.choice(4, p=p) for p in ps], shape)
    weights = np.reshape(weights, shape + (4,))
    check_codec(shape, cs.CategoricalNew(weights, precision), data)

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
    state = cs.base_message(shape)
    some_bits = rng.randint(1 << p, size=(n,) + shape).astype(np.uint64)
    freqs = np.ones(shape, dtype="uint64")
    for b in some_bits:
        state, = cs.rans.push(state, b, freqs, p)
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
    assert rans.message_equal(message1, message2)

@pytest.mark.parametrize('old_size', [141, 32, 17, 6, 3])
@pytest.mark.parametrize('new_size', [141, 32, 17, 6, 3])
def test_resize_head_1d(old_size, new_size, depth=1000):
    old_shape = (old_size,)

    np.random.seed(0)
    p = 8
    bits = np.random.randint(1 << p, size=(depth,) + old_shape, dtype=np.uint64)

    message = cs.base_message(old_shape)

    other_bits_push, _ = cs.repeat(cs.Uniform(p), depth)

    message, = other_bits_push(message, bits)

    resized = cs.codecs._resize_head_1d(message, new_size)
    reconstructed = cs.codecs._resize_head_1d(resized, old_size)

    assert_message_equal(message, reconstructed)

@pytest.mark.parametrize('old_shape', [(100,), (1, 23), (2, 4, 5)])
@pytest.mark.parametrize('new_shape', [(100,), (1, 23), (2, 4, 5)])
def test_reshape_head(old_shape, new_shape, depth=1000):
    np.random.seed(0)
    p = 8
    bits = np.random.randint(1 << p, size=(depth,) + old_shape, dtype=np.uint64)

    message = cs.base_message(old_shape)

    other_bits_push, _ = cs.repeat(cs.Uniform(p), depth)

    message, = other_bits_push(message, bits)

    resized = cs.reshape_head(message, new_shape)
    reconstructed = cs.reshape_head(resized, old_shape)

    assert_message_equal(message, reconstructed)

@pytest.mark.parametrize('shape', [(100,), (1, 23), (2, 4, 5)])
def test_flatten_unflatten(shape, depth=1000):
    np.random.seed(0)
    p = 8
    bits = np.random.randint(1 << p, size=(depth,) + shape, dtype=np.uint64)

    message = cs.base_message(shape)

    other_bits_push, _ = cs.repeat(cs.Uniform(p), depth)

    message, = other_bits_push(message, bits)

    flattened = cs.flatten(message)
    reconstructed = cs.unflatten(flattened, shape)

    assert_message_equal(message, reconstructed)

def test_flatten_rate():
    n = 1000

    init_data = np.random.randint(1 << 16, size=8 * n, dtype='uint64')

    init_message = cs.base_message((1,))

    for datum in init_data:
        init_message, = cs.Uniform(16).push(init_message, datum)

    l_init = len(cs.flatten(init_message))

    ps = np.random.rand(n, 1)
    data = np.random.rand(n, 1) < ps

    message = init_message
    for p, datum in zip(ps, data):
        message, = cs.Bernoulli(p, 14).push(message, datum)

    l_scalar = len(cs.flatten(message))

    message = init_message
    message = cs.reshape_head(message, (n, 1))
    message, = cs.Bernoulli(ps, 14).push(message, data)

    l_vector = len(cs.flatten(message))

    assert (l_vector - l_init) / (l_scalar - l_init) - 1 < 0.001

def test_multiset_codec():
    multiset = cs.build_multiset([0, 255, 128, 128])

    ans_state = rans.base_message(shape=(1,))
    symbol_codec = cs.Uniform(8)
    codec = cs.Multiset(symbol_codec)

    ans_state, = codec.push(ans_state, multiset)
    ans_state, multiset_decoded = codec.pop(ans_state, multiset_size=4)

    assert cs.check_multiset_equality(multiset, multiset_decoded)


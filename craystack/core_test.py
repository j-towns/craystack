import numpy as np
import numpy.random as rng

import craystack as cs
import craystack.vectorans as vrans
import craystack.bb_ans as bb


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
    head, tail = vrans.x_init((4, 4))
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
    codecs = [cs.Uniform(p) for p in precs]
    view_fun = lambda slc: lambda head: head[slc]
    view_funs = []
    start = 0
    for s in szs:
        view_funs.append(view_fun(slice(start, start + s)))
        start += s
    data = [rng.randint(1 << p, size=size, dtype='uint64')
            for p, size in zip(precs, szs)]
    check_codec(sum(szs), cs.parallel(codecs, view_funs), data)


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
    precision = 12
    batch_size = 4
    means = rng.randn(batch_size)
    log_scale = rng.randn()
    # type is important!
    data = np.array([rng.choice(256) for _ in range(batch_size)]).astype('uint64')
    check_codec((batch_size,), cs.Logistic(means, log_scale, precision), data)


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
    check_codec((shape[0],), cs.LogisticMixture(theta, precision), data)


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
    elem_codec = lambda p: cs.Categorical(p, precision)
    check_codec((batch_size,),
                cs.AutoRegressive(lambda x: fixed_probs,
                                  (batch_size, data_size,),
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
                bb.DiagGaussianLatent(means, stdds, bin_means, bin_stdds,
                                      coding_precision, bin_precision),
                data)


def assert_message_equal(message1, message2):
    np.testing.assert_equal(message1, message2)

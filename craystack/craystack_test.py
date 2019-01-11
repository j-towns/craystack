import numpy as np
import numpy.random as rng

import craystack.craystack as cs
import craystack.vectorans as vrans


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
                               & ((1 << data_len) -1)))
    check_codec(shape, cs.Benford64, data)

def test_benford_inverse():
    shape = (2, 3, 5)
    data_len = rng.randint(31, 63, shape, dtype="uint64")
    data = ((1 << data_len) | (rng.randint(1 << 63, size=shape, dtype="uint64")
                               & ((1 << data_len) -1)))
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

def assert_message_equal(message1, message2):
    np.testing.assert_equal(message1, message2)

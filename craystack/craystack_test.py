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

def assert_message_equal(message1, message2):
    head1, m1 = message1
    head2, m2 = message2
    np.testing.assert_equal(head1, head2)
    assert m1 == m2

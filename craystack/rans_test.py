from craystack import rans
from numpy.testing import assert_equal
import numpy as np

rng = np.random.RandomState(0)
num_lower_bits = int(np.log2(rans.rans_l)) + 1


def test_rans():
    shape = (8, 7)
    precision = 8
    n_data = 1000

    x = rans.base_message(shape)
    starts = rng.randint(0, 256, size=(n_data,) + shape).astype("uint64")
    freqs = (rng.randint(1, 256, size=(n_data,) + shape).astype("uint64")
             % (256 - starts))
    freqs[freqs == 0] = 1
    assert np.all(starts + freqs <= 256)
    print("Exact entropy: " + str(np.sum(np.log2(256 / freqs))) + " bits.")
    # Encode
    for start, freq in zip(starts, freqs):
        x = rans.push(x, start, freq, precision)
    coded_arr = rans.flatten(x)
    assert coded_arr.dtype == np.uint32
    print("Actual output shape: " + str(32 * len(coded_arr)) + " bits.")

    # Decode
    x = rans.unflatten(coded_arr, shape)
    for start, freq in reversed(list(zip(starts, freqs))):
        cf, pop = rans.pop(x, precision)
        assert np.all(start <= cf) and np.all(cf < start + freq)
        x = pop(start, freq)
    assert np.all(x[0] == rans.base_message(shape)[0])


def test_flatten_unflatten():
    n = 100
    shape = (7, 3)
    prec = 12
    state = rans.base_message(shape)
    some_bits = rng.randint(1 << prec, size=(n,) + shape).astype(np.uint64)
    freqs = np.ones(shape, dtype="uint64")
    for b in some_bits:
        state = rans.push(state, b, freqs, prec)
    flat = rans.flatten(state)
    assert flat.dtype is np.dtype("uint32")
    state_ = rans.unflatten(flat, shape)
    flat_ = rans.flatten(state_)
    assert np.all(flat == flat_)
    assert np.all(state[0] == state_[0])
    # assert state[1] == state_[1]


def test_base_message():
    def calc_num_ones(head):
        return sum([((head >> i) % 2).sum()
                    for i in range(num_lower_bits)])


    head = rans.base_message(1_000)[0]
    assert calc_num_ones(head) == 1_000

    head_rnd = rans.base_message(100_000, randomize=True)[0]
    num_bits = num_lower_bits*100_000
    assert (head_rnd >> (num_lower_bits - 1) == 1).all()
    assert  0.48 < calc_num_ones(head_rnd)/num_bits < 0.52

def test_stack_slice():
    stack = (np.array([0, 1]).astype('uint64'),
             (np.array([2, 3]).astype('uint64'),
              (np.array([4, 5]).astype('uint64'),
               ())))

    stack, sliced = rans.stack_slice(stack, 1)
    assert_equal(        sliced, np.array([0   ]))
    assert_equal(      stack[0], np.array([1   ]))
    assert_equal(   stack[1][0], np.array([2, 3]))
    assert_equal(stack[1][1][0], np.array([4, 5]))
    assert stack[1][1][1] == ()

    stack, sliced = rans.stack_slice(stack, 2)
    assert_equal(     sliced, np.array([1, 2]))
    assert_equal(   stack[0], np.array([3   ]))
    assert_equal(stack[1][0], np.array([4, 5]))
    assert stack[1][1] == ()

    stack, sliced = rans.stack_slice(stack, 5)
    assert len(sliced) == 5
    assert_equal(sliced[3:] >> num_lower_bits-1, np.array([1, 1]))
    assert_equal(sliced[:3], np.array([3, 4, 5]))
    assert stack == ()

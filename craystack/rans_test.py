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
    def popcount(head):
        return sum([((head >> i) % 2).sum()
                    for i in range(num_lower_bits)])

    head = rans.base_message(1_000)[0]
    assert popcount(head) == 1_000

    head_rnd = rans.base_message(100_000, randomize=True)[0]
    num_bits = num_lower_bits*100_000
    assert (head_rnd >> (num_lower_bits - 1) == 1).all()
    assert  0.48 < popcount(head_rnd)/num_bits < 0.52

def test_finer_prec():
    # Regression test.
    # The renormalization in rans.push_with_finer_prec will keep moving
    # bits into the tail until the head is less than
    # (rans_l // 3) << r_t      == 3074457342754947072

    # However, the true upper bound on the head after rans.pop_with_finer_prec
    # is less strict, it's equal to
    # ((1 << r_s) - 1) // 3 + 1 == 3074457345618258603

    # We can break the inverse property by constructing a message and
    # pop such that the result is between the two values above. That means
    # that push thinks it needs to undo a renorm when actually no renorm
    # took place in the pop:

    # Set s to the largest possible value
    m = np.uint64([(1 << 63) - 1]), ()

    # Use N = 3 as the denominator
    N = 3

    u, popfun = rans.pop_with_finer_prec(m, N)
    m_ = popfun(u, 1)

    # u  == array([1], dtype=uint64)
    # m_ == (array([3074457345618258602], dtype=uint64), ())
    # s is between the two values above, and that causes a problem...

    m__ = rans.push_with_finer_prec(m_, u, 1, N)

    assert rans.message_equal(m, m__), 'm = ' + str(m) + '\n m__ = ' + str(m__)

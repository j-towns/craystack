import craystack.vectorans as vrans
import numpy as np

rng = np.random.RandomState(0)


def test_vectorans():
    shape = (8, 7)
    precision = 8
    n_data = 1000

    x = vrans.message_init(shape)
    starts = rng.randint(0, 256, size=(n_data,) + shape).astype("uint64")
    freqs = (rng.randint(1, 256, size=(n_data,) + shape).astype("uint64")
             % (256 - starts))
    freqs[freqs == 0] = 1
    assert np.all(starts + freqs <= 256)
    print("Exact entropy: " + str(np.sum(np.log2(256 / freqs))) + " bits.")
    # Encode
    for start, freq in zip(starts, freqs):
        x = vrans.push(x, start, freq, precision)
    coded_arr = vrans.flatten(x)
    assert coded_arr.dtype == np.uint32
    print("Actual output shape: " + str(32 * len(coded_arr)) + " bits.")

    # Decode
    x = vrans.unflatten(coded_arr, shape)
    for start, freq in reversed(list(zip(starts, freqs))):
        cf, pop = vrans.pop(x, precision)
        assert np.all(start <= cf) and np.all(cf < start + freq)
        x = pop(start, freq)
    assert np.all(x[0] == vrans.message_init(shape)[0])


def test_flatten_unflatten():
    n = 100
    shape = (7, 3)
    prec = 12
    state = vrans.message_init(shape)
    some_bits = rng.randint(1 << prec, size=(n,) + shape).astype(np.uint64)
    freqs = np.ones(shape, dtype="uint64")
    for b in some_bits:
        state = vrans.push(state, b, freqs, prec)
    flat = vrans.flatten(state)
    assert flat.dtype is np.dtype("uint32")
    state_ = vrans.unflatten(flat, shape)
    flat_ = vrans.flatten(state_)
    assert np.all(flat == flat_)
    assert np.all(state[0] == state_[0])
    assert state[1] == state_[1]

from itertools import product
import numpy as np
import craystack.vectorans as vrans
import craystack.util as util
from scipy.special import expit as sigmoid


def NonUniform(enc_statfun, dec_statfun, precision):
    """
    Codec for symbols which are not uniformly distributed. The statfuns specify
    the following mappings:

        enc_statfun: symbol |-> start, freq
        dec_statfun: cf |-> symbol

    The interval [0, 1) is modelled by the range of integers
    [0, 2 ** precision). The operation performed by enc_statfun is used for
    compressing data and is visualised below for a distribution over a set of
    symbols {a, b, c, d}.

    0                                                         2 ** precision
    |    a              !b!              c              d         |
    |----------|--------------------|---------|-------------------|
               |------ freq --------|
             start

    Calling enc_statfun(b) must return the pair (start, freq), where start is
    the start of the interval representing the symbol b and freq is its width.
    Start and freq must satisfy the following constraints:

        0 <  freq
        0 <= start        <  2 ** precision
        0 <  start + freq <= 2 ** precision

    The value of start is analagous to the cdf of the distribution, evaluated
    at b, while freq is analagous to the pmf evaluated at b.

    The function dec_statfun essentially inverts enc_statfun. It is
    necessary for decompressing data, to recover the original symbol.

    0                                                         2 ** precision
    |    a               b               c              d         |
    |----------|-----+--------------|---------|-------------------|
                     ↑
                     cf

    For a number cf in the range [0, 2 ** precision), dec_statfun must return
    the symbol whose range cf lies in, which in the picture above is b.
    """
    def append(message, symbol):
        start, freq = enc_statfun(symbol)
        return vrans.append(message, start, freq, precision)

    def pop(message):
        cf, pop_fun = vrans.pop(message, precision)
        symbol = dec_statfun(cf)
        start, freq = enc_statfun(symbol)
        assert np.all(start <= cf) and np.all(cf < start + freq)
        return pop_fun(start, freq), symbol
    return append, pop

_uniform_enc_statfun = lambda s: (s, 1)
_uniform_dec_statfun = lambda cf: cf

def _cdf_to_enc_statfun(cdf):
    def enc_statfun(s):
        lower = cdf(s)
        return lower, cdf(s + 1) - lower
    return enc_statfun

def Uniform(precision):
    """
    Codec for symbols uniformly distributed over range(1 << precision).
    """
    # TODO: special case this in vectorans.py
    return NonUniform(_uniform_enc_statfun, _uniform_dec_statfun, precision)

def Benford64():
    """
    Simple self-delimiting code for numbers x with

        2 ** 31 <= x < 2 ** 63

    with log(x) approximately uniformly distributed. Useful for coding
    vectorans stack heads.
    """
    length_append, length_pop = Uniform(5)
    x_lower_append, x_lower_pop = Uniform(31)
    def append(message, x):
        message = x_lower_append(message, x & ((1 << 31) - 1))
        x_len = np.uint64(np.log2(x))
        x = x & ((1 << x_len) - 1)  # Rm leading 1
        x_higher_append, _ = Uniform(x_len - 31)
        message = x_higher_append(message, x >> 31)
        message = length_append(message, x_len - 31)
        return message

    def pop(message):
        message, x_len = length_pop(message)
        x_len = x_len + 31
        _, x_higher_pop = Uniform(x_len - 31)
        message, x_higher = x_higher_pop(message)
        message, x_lower = x_lower_pop(message)
        return message, (1 << x_len) | (x_higher << 31) | x_lower
    return append, pop
Benford64 = Benford64()

def repeat(codec, n):
    """
    Repeat codec n times.

    Assumes that symbols is a Numpy array with symbols.shape[0] == n. Assume
    that the codec doesn't change the shape of the ANS stack head.
    """
    append_, pop_ = codec
    def append(message, symbols):
        assert np.shape(symbols)[0] == n
        for symbol in reversed(symbols):
            message = append_(message, symbol)
        return message

    def pop(message):
        symbols = []
        for i in range(n):
            message, symbol = pop_(message)
            symbols.append(symbol)
        return message, np.asarray(symbols)
    return append, pop

def substack(codec, view_fun):
    append_, pop_ = codec
    def append(message, data):
        head, tail = message
        subhead, update = util.view_update(head, view_fun)
        subhead, tail = append_((subhead, tail), data)
        return update(subhead), tail
    def pop(message):
        head, tail = message
        subhead, update = util.view_update(head, view_fun)
        (subhead, tail), data = pop_((subhead, tail))
        return (update(subhead), tail), data
    return append, pop

def parallel(codecs, view_funs):
    """
    Run a number of independent codecs on different substacks. This could be
    executed in parallel, but when running in the Python interpreter will run
    in series.

    Assumes data is a list, arranged the same way as codecs and view_funs.
    """
    codecs = [substack(codec, view_fun)
              for codec, view_fun in zip(codecs, view_funs)]
    def append(message, symbols):
        assert len(symbols) == len(codecs)
        for (append, _), symbol in reversed(list(zip(codecs, symbols))):
            message = append(message, symbol)
        return message
    def pop(message):
        symbols = []
        for _, pop in codecs:
            message, symbol = pop(message)
            symbols.append(symbol)
        assert len(symbols) == len(codecs)
        return message, symbols
    return append, pop

def shape(message):
    head, _ = message
    def _shape(head):
        if type(head) is tuple:
            return tuple(_shape(h) for h in head)
        else:
            return np.shape(head)
    return _shape(head)

# VAE observation codecs
def _nearest_int(arr):
    return np.uint64(np.ceil(arr - 0.5))

def _ensure_nonzero_freq_bernoulli(p, precision):
    p[p == 0] += 1
    p[p == (1 << precision)] -=1
    return p

def _bernoulli_cdf(p, precision, safe=True):
    def cdf(s):
        ret = np.zeros(np.shape(s), "uint64")
        onemp = _nearest_int((1 - p[s==1]) * (1 << precision))
        onemp = (_ensure_nonzero_freq_bernoulli(onemp, precision) if safe
                 else onemp)
        ret[s == 1] += onemp
        ret[s == 2] = 1 << precision
        return ret
    return cdf

def _bernoulli_ppf(p, precision, safe=True):
    onemp = _nearest_int((1 - p) * (1 << precision))
    onemp = _ensure_nonzero_freq_bernoulli(onemp, precision) if safe else onemp
    return lambda cf: np.uint64((cf + 0.5) > onemp)

def Bernoulli(p, prec):
    enc_statfun = _cdf_to_enc_statfun(_bernoulli_cdf(p, prec))
    dec_statfun = _bernoulli_ppf(p, prec)
    return NonUniform(enc_statfun, dec_statfun, prec)

def _cumulative_buckets_from_probs(probs, precision):
    """Ensure each bucket has at least frequency 1"""
    probs = np.rint(probs * (1 << precision)).astype('int64')
    probs[probs == 0] = 1
    # TODO(@j-towns): look at simplifying this
    # Normalize the probabilities by decreasing the maxes
    argmax_idxs = np.argmax(probs, axis=-1)[..., np.newaxis]
    max_value = np.take_along_axis(probs, argmax_idxs, axis=-1)
    diffs = (1 << precision) - np.sum(probs, axis=-1, keepdims=True)
    assert not np.any(max_value + diffs <= 0), \
        "cannot rebalance buckets, consider increasing precision"
    lowered_maxes = (max_value + diffs)
    np.put_along_axis(probs, argmax_idxs, lowered_maxes, axis=-1)
    return np.concatenate((np.zeros(np.shape(probs)[:-1] + (1,), dtype='uint64'),
                           np.cumsum(probs, axis=-1)), axis=-1).astype('uint64')

def _cdf_from_cumulative_buckets(c_buckets):
    def cdf(s):
        ret = np.take_along_axis(c_buckets, s[..., np.newaxis],
                                 axis=-1)
        return ret[..., 0]
    return cdf

def _ppf_from_cumulative_buckets(c_buckets):
    *shape, n = np.shape(c_buckets)
    cumulative_buckets = np.reshape(c_buckets, (-1, n))
    def ppf(cfs):
        cfs = np.ravel(cfs)
        ret = np.array(
            [np.searchsorted(bucket, cf, 'right') - 1 for bucket, cf in
             zip(cumulative_buckets, cfs)])
        return np.reshape(ret, shape)
    return ppf

def Categorical(p, prec):
    """Assume that the last dim of p contains the probability vectors,
    i.e. np.sum(p, axis=-1) == ones"""
    cumulative_buckets = _cumulative_buckets_from_probs(p, prec)
    enc_statfun = _cdf_to_enc_statfun(_cdf_from_cumulative_buckets(cumulative_buckets))
    dec_statfun = _ppf_from_cumulative_buckets(cumulative_buckets)
    return NonUniform(enc_statfun, dec_statfun, prec)

def _create_logistic_mixture_buckets(means, log_scales, logit_probs, prec):
    inv_stdv = np.exp(-log_scales)
    buckets = np.linspace(-1, 1, 257)  # TODO: change hardcoding
    buckets = np.broadcast_to(buckets, means.shape + (257,))
    cdfs = inv_stdv[..., np.newaxis] * (buckets - means[..., np.newaxis])
    cdfs[..., 0] = -np.inf
    cdfs[..., -1] = np.inf
    cdfs = sigmoid(cdfs)
    prob_cpts = cdfs[..., 1:] - cdfs[..., :-1]
    mixture_probs = util.softmax(logit_probs, axis=1)
    probs = np.sum(prob_cpts * mixture_probs[..., np.newaxis], axis=1)
    return _cumulative_buckets_from_probs(probs, prec)

def LogisticMixture(theta, prec):
    """theta: means, log_scales, logit_probs"""
    means, log_scales, logit_probs = np.split(theta, 3, axis=-1)
    cumulative_buckets = _create_logistic_mixture_buckets(means, log_scales,
                                                          logit_probs, prec)
    enc_statfun = _cdf_to_enc_statfun(_cdf_from_cumulative_buckets(cumulative_buckets))
    dec_statfun = _ppf_from_cumulative_buckets(cumulative_buckets)
    return NonUniform(enc_statfun, dec_statfun, prec)

def AutoRegressive(elem_param_fn, data_shape, params_shape, elem_idxs, elem_codec):
    def append(message, data, all_params=None):
        if not all_params:
            all_params = elem_param_fn(data)
        for idx in reversed(elem_idxs):
            elem_params = all_params[idx]
            elem_append, _ = elem_codec(elem_params, idx)
            message = elem_append(message, data[idx].astype('uint64'))
        return message

    def pop(message):
        data = np.zeros(data_shape, dtype=np.uint64)
        all_params = np.zeros(params_shape, dtype=np.float32)
        for idx in elem_idxs:
            all_params = elem_param_fn(data, all_params)
            elem_params = all_params[idx]
            _, elem_pop = elem_codec(elem_params, idx)
            message, elem = elem_pop(message)
            data[idx] = elem
        return message, (data, all_params)
    return append, pop

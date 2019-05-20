import math
from functools import partial

from scipy.stats import norm

from scipy.special import expit as sigmoid

import numpy as np
import craystack.vectorans as vrans
import craystack.util as util


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

def serial(codecs):
    """
    Applies given codecs in series.

    Codecs and symbols can be any iterable.
    Codecs are allowed to change the shape of the ANS stack head.
    """
    def append(message, symbols):
        for (append, _), symbol in reversed(list(zip(codecs, symbols))):
            message = append(message, symbol)
        return message

    def pop(message):
        symbols = []
        for _, pop in codecs:
            message, symbol = pop(message)
            symbols.append(symbol)
        return message, symbols

    return append, pop

def substack(codec, view_fun):
    append_, pop_ = codec
    def append(message, data, *args, **kwargs):
        head, tail = message
        subhead, update = util.view_update(head, view_fun)
        subhead, tail = append_((subhead, tail), data, *args, **kwargs)
        return update(subhead), tail
    def pop(message, *args, **kwargs):
        head, tail = message
        subhead, update = util.view_update(head, view_fun)
        (subhead, tail), data = pop_((subhead, tail), *args, **kwargs)
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

_uniform_enc_statfun = lambda s: (s, 1)
_uniform_dec_statfun = lambda cf: cf

def _cdf_to_enc_statfun(cdf):
    def enc_statfun(s):
        lower = cdf(s)
        return lower, cdf(s + 1) - lower
    return enc_statfun

# VAE observation codecs
def _nearest_int(arr):
    return np.uint64(np.ceil(arr - 0.5))

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

def flatten_benford(x):
    return vrans.flatten(reshape_head(x, (1, )))

def unflatten_benford(arr, shape):
    return reshape_head(vrans.unflatten(arr, (1,)), shape)

def _resize_head_1d_codecs(small, big):
    sizes = []
    half = big
    while True:
        sizes.append(half)

        if small == half:
            break

        half = math.ceil(half / 2) if half >= 2 * small else small

    sizes = np.array(list(reversed(sizes)))
    smaller_sizes = sizes[:-1]
    bigger_sizes = sizes[1:]
    steps = bigger_sizes - smaller_sizes
    view_funs = [partial(lambda h, s=s: h[:s]) for s in steps]
    codecs = [substack(Benford64, view_fun) for view_fun in view_funs]
    return list(zip(codecs, smaller_sizes))

def _reshape_head_1d(message, size):
    head, tail = message
    should_reduce = size < head.shape[0]
    return (_reduce_head_1d if should_reduce else _grow_head_1d)(message, size)

def _reduce_head_1d(message, size):
    head, tail = message

    for (append, _), new_size in reversed(_resize_head_1d_codecs(small=size, big=head.shape[0])):
        head, tail = message
        message = head[:new_size], tail
        message = append(message, head[new_size:])

    return message

def _grow_head_1d(message, size):
    head, tail = message
    for (_, pop), _ in _resize_head_1d_codecs(small=head.shape[0], big=size):
        message, head_extension = pop(message)
        head, tail = message
        message = np.concatenate([head, head_extension]), tail

    return message

def reshape_head(message, shape):
    head, tail = message
    message = (np.ravel(head), tail)
    head, tail = _reshape_head_1d(message, size=np.prod(shape))
    return np.reshape(head, shape), tail

def random_stack(flat_len, shape, rng=np.random):
    """Generate a random vrans stack"""
    arr = rng.randint(1 << 32, size=flat_len, dtype='uint32')
    return unflatten_benford(arr, shape)

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

def _create_logistic_buckets(means, log_scale, coding_prec, bin_prec):
    buckets = np.linspace(-0.5, 0.5, (1 << bin_prec)+1)
    buckets = np.broadcast_to(buckets, means.shape + ((1 << bin_prec)+1,))
    inv_stdv = np.exp(-log_scale)
    cdfs = inv_stdv * (buckets - means[..., np.newaxis])
    cdfs[..., 0] = -np.inf
    cdfs[..., -1] = np.inf
    cdfs = sigmoid(cdfs)
    probs = cdfs[..., 1:] - cdfs[..., :-1]
    return _cumulative_buckets_from_probs(probs, coding_prec)

def _logistic_cdf(means, log_scale, coding_prec, bin_prec):
    inv_stdv = np.exp(-log_scale)
    def cdf(idx):
        # can reduce mem footprint
        buckets = np.linspace(-0.5, 0.5, (1 << bin_prec)+1)
        buckets = np.append(buckets, np.inf)
        bucket_ub = buckets[idx+1]
        scaled = inv_stdv * (bucket_ub - means)
        cdf = sigmoid(scaled)
        return _nearest_int(cdf * (1 << coding_prec))
    return cdf

def _logistic_ppf(means, log_scale, coding_prec, bin_prec):
    stdv = np.exp(log_scale)
    def ppf(cf):
        x = (cf + 0.5) / (1 << coding_prec)
        logit = np.log(x) - np.log(1-x)
        x = logit * stdv + means
        bins = np.linspace(-0.5, 0.5, (1 << bin_prec)+1)[1:]
        return np.uint64(np.digitize(x, bins) - 1)
    return ppf

def Logistic(mean, log_scale, coding_prec, bin_prec, no_zero_freqs=True, log_scale_min=-6):
    if no_zero_freqs:
        cumulative_buckets = _create_logistic_buckets(mean, log_scale, coding_prec, bin_prec)
        enc_statfun = _cdf_to_enc_statfun(_cdf_from_cumulative_buckets(cumulative_buckets))
        dec_statfun = _ppf_from_cumulative_buckets(cumulative_buckets)
    else:
        log_scale = max(log_scale, log_scale_min)
        enc_statfun = _cdf_to_enc_statfun(_logistic_cdf(mean, log_scale, coding_prec, bin_prec))
        dec_statfun = _logistic_ppf(mean, log_scale, coding_prec, bin_prec)
    return NonUniform(enc_statfun, dec_statfun, coding_prec)

def _create_logistic_mixture_buckets(means, log_scales, logit_probs, coding_prec, bin_prec):
    inv_stdv = np.exp(-log_scales)
    buckets = np.linspace(-1, 1, (1 << bin_prec)+1)
    buckets = np.broadcast_to(buckets, means.shape + ((1 << bin_prec)+1,))
    cdfs = inv_stdv[..., np.newaxis] * (buckets - means[..., np.newaxis])
    cdfs[..., 0] = -np.inf
    cdfs[..., -1] = np.inf
    cdfs = sigmoid(cdfs)
    prob_cpts = cdfs[..., 1:] - cdfs[..., :-1]
    mixture_probs = util.softmax(logit_probs, axis=1)
    probs = np.sum(prob_cpts * mixture_probs[..., np.newaxis], axis=1)
    return _cumulative_buckets_from_probs(probs, coding_prec)

def LogisticMixture(theta, coding_prec, bin_prec=8):
    """theta: means, log_scales, logit_probs"""
    means, log_scales, logit_probs = np.split(theta, 3, axis=-1)
    cumulative_buckets = _create_logistic_mixture_buckets(means, log_scales,
                                                          logit_probs, coding_prec, bin_prec)
    enc_statfun = _cdf_to_enc_statfun(_cdf_from_cumulative_buckets(cumulative_buckets))
    dec_statfun = _ppf_from_cumulative_buckets(cumulative_buckets)
    return NonUniform(enc_statfun, dec_statfun, coding_prec)

std_gaussian_bucket_cache = {}  # Stores bucket endpoints
std_gaussian_centres_cache = {}  # Stores bucket centres

def std_gaussian_buckets(precision):
    """
    Return the endpoints of buckets partioning the domain of the prior. Each
    bucket has mass 1 / (1 << precision) under the prior.
    """
    if precision in std_gaussian_bucket_cache:
        return std_gaussian_bucket_cache[precision]
    else:
        buckets = norm.ppf(np.linspace(0, 1, (1 << precision) + 1))
        std_gaussian_bucket_cache[precision] = buckets
        return buckets

def std_gaussian_centres(precision):
    """
    Return the centres of mass of buckets partioning the domain of the prior.
    Each bucket has mass 1 / (1 << precision) under the prior.
    """
    if precision in std_gaussian_centres_cache:
        return std_gaussian_centres_cache[precision]
    else:
        centres = np.float32(
            norm.ppf((np.arange(1 << precision) + 0.5) / (1 << precision)))
        std_gaussian_centres_cache[precision] = centres
        return centres

def _gaussian_latent_cdf(mean, stdd, prior_prec, post_prec):
    def cdf(idx):
        x = std_gaussian_buckets(prior_prec)[idx]
        return _nearest_int(norm.cdf(x, mean, stdd) * (1 << post_prec))
    return cdf

def _gaussian_latent_ppf(mean, stdd, prior_prec, post_prec):
    def ppf(cf):
        x = norm.ppf((cf + 0.5) / (1 << post_prec), mean, stdd)
        # Binary search is faster than using the actual gaussian cdf for the
        # precisions we typically use, however the cdf is O(1) whereas search
        # is O(precision), so for high precision cdf will be faster.
        return np.uint64(np.digitize(x, std_gaussian_buckets(prior_prec)) - 1)
    return ppf

def DiagGaussianLatentStdBins(mean, stdd, coding_prec, bin_prec):
    enc_statfun = _cdf_to_enc_statfun(
        _gaussian_latent_cdf(mean, stdd, bin_prec, coding_prec))
    dec_statfun = _gaussian_latent_ppf(mean, stdd, bin_prec, coding_prec)
    return NonUniform(enc_statfun, dec_statfun, coding_prec)

def DiagGaussianLatent(mean, stdd, bin_mean, bin_stdd, coding_prec, bin_prec):
    """To code Gaussian data according to the bins of a different Gaussian"""

    def cdf(idx):
        x = norm.ppf(idx / (1 << bin_prec), bin_mean, bin_stdd)  # this gives lb of bin
        return _nearest_int(norm.cdf(x, mean, stdd) * (1 << coding_prec))

    def ppf(cf):
        x_max = norm.ppf((cf + 0.5) / (1 << coding_prec), mean, stdd)
        # if our gaussians have little overlap, then the cdf could be exactly 1
        # therefore cut off at (1<<bin_prec)-1 to make sure we return a valid bin
        return np.uint64(np.minimum((1 << bin_prec) - 1,
                                    norm.cdf(x_max, bin_mean, bin_stdd) * (1 << bin_prec)))

    enc_statfun = _cdf_to_enc_statfun(cdf)
    return NonUniform(enc_statfun, ppf, coding_prec)

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
            all_params = elem_param_fn(data, all_params, idx)
            elem_params = all_params[idx]
            _, elem_pop = elem_codec(elem_params, idx)
            message, elem = elem_pop(message)
            data[idx] = elem
        return message, data
    return append, pop

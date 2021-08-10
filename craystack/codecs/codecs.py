import itertools
from functools import lru_cache
from warnings import warn
from collections import namedtuple

from scipy.stats import norm
from scipy.special import expit as sigmoid
from scipy.special import expit, logit

import numpy as np
import craystack.rans as vrans
import craystack.util as util

Codec = namedtuple('Codec', ['push', 'pop'])

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
    if np.any(precision >= 28):
        warn('Detected precision over 28. Codecs lose accuracy at high '
             'precision.')

    def push(message, symbol):
        start, freq = enc_statfun(symbol)
        return vrans.push(message, start, freq, precision),

    def pop(message):
        cf, pop_fun = vrans.pop(message, precision)
        symbol = dec_statfun(cf)
        start, freq = enc_statfun(symbol)
        assert np.all(start <= cf) and np.all(cf < start + freq)
        return pop_fun(start, freq), symbol
    return Codec(push, pop)

def repeat(codec, n):
    """
    Repeat codec n times.

    Assumes that symbols is a list with len(symbols) == n.
    """
    return from_iterable(itertools.repeat(codec, n))

def from_iterable(iterable):
    """
    Allows defining a codec in terms of a sequence of codecs. Each element of
    the argument `iterable` must be a codec. The codecs should be arranged in
    the order in which data will be *decoded* (popped).
    """
    iterable = list(iterable)
    def gen_fun():
        return (x for x in iterable)
    return from_generator(gen_fun)

def from_generator(generator_fun):
    """
    This allows defining a codec using a Python generator function. The
    generator must `yield` a sequence of codecs, in the order in which data
    should be decoded. We can easily implement a 'serial' codec in terms of
    `from_generator`:

    >>> def serial(codecs):
    >>>     def gen_fun():
    ...         for codec in codecs:
    ...             yield codec
    ...     return from_generator(gen_fun)

    The symbols resulting from decoding can be used by the generator, by
    assigning the result of the yield expression, for example, we might firstly
    decode the precision of a Uniformly distributed symbol, then decode the
    symbol itself:

    >>> def gen_fun():
    ...     prec = (yield cs.Uniform(16))
    ...     yield cs.Uniform(prec)
    >>>
    >>> dependent_codec = from_generator(gen_fun)
    """
    def safe_next(generator):
        return safe_send(generator, None)

    def safe_send(generator, s):
        try: n = generator.send(s)
        except StopIteration: return True, None
        return False, n

    def push(message, result):
        g = generator_fun()
        codec_stack = []
        done, codec = safe_next(g)
        for symbol in result:
            assert not done
            codec_stack.append(codec)
            done, codec = safe_send(g, symbol)
        for codec, symbol in reversed(list(zip(codec_stack, result))):
            message, = codec.push(message, symbol)
        return message,

    def pop(message):
        g = generator_fun()
        result = []
        done, codec = safe_next(g)
        while not done:
            message, symbol = codec.pop(message)
            result.append(symbol)
            done, codec = safe_send(g, symbol)
        return message, result

    return Codec(push, pop)

def substack(codec, view_fun):
    """
    Apply a codec on a subset of a message head.

    view_fun should be a function: head -> subhead, for example
    view_fun = lambda head: head[0]
    to run the codec on only the first element of the head
    """
    def push(message, data, *context):
        head, tail = message
        subhead, update = util.view_update(head, view_fun)
        (subhead, tail), *context = codec.push((subhead, tail), data, *context)
        return ((update(subhead), tail), *context)

    def pop(message, *context):
        head, tail = message
        subhead, update = util.view_update(head, view_fun)
        (subhead, tail), data, *context = codec.pop((subhead, tail), *context)
        return ((update(subhead), tail), data, *context)
    return Codec(push, pop)

def parallel(codecs, view_funs):
    """
    Run a number of independent codecs on different substacks. This could be
    executed in parallel, but when running in the Python interpreter will run
    in series.

    Assumes data is a list, arranged the same way as codecs and view_funs.
    """
    codecs = [substack(codec, view_fun)
              for codec, view_fun in zip(codecs, view_funs)]
    def push(message, symbols):
        assert len(symbols) == len(codecs)
        for codec, symbol in reversed(list(zip(codecs, symbols))):
            message, = codec.push(message, symbol)
        return message,
    def pop(message):
        symbols = []
        for codec in codecs:
            message, symbol = codec.pop(message)
            symbols.append(symbol)
        assert len(symbols) == len(codecs)
        return message, symbols
    return Codec(push, pop)

def shape(message):
    """Get the shape of the message head(s)"""
    head, _ = message
    def _shape(head):
        if type(head) is tuple:
            return tuple(_shape(h) for h in head)
        else:
            return np.shape(head)
    return _shape(head)

def is_empty(message):
    """Check if message is empty.
    Useful for decoding something of unknown length"""
    return (not message[1]) and np.all(message[0] == vrans.rans_l)

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
    # TODO: special case this in rans.py
    return NonUniform(_uniform_enc_statfun, _uniform_dec_statfun, precision)

def BigUniform(precision):
    """
    Uniform codec allowing precision > 24. Assumes symbols have dtype uint64.
    """
    def push(message, symbol):
        for lower in [0, 16, 32, 48]:
            s = (symbol >> lower) & ((1 << 16) - 1)
            diff = np.where(precision >= lower, precision - lower, 0)
            p = np.minimum(diff, 16)
            message, = Uniform(p).push(message, s)
        return message,

    def pop(message):
        symbol = 0
        for lower in [48, 32, 16, 0]:
            diff = np.where(precision >= lower, precision - lower, 0)
            p = np.minimum(diff, 16)
            message, s = Uniform(p).pop(message)
            symbol = (symbol << 16) | s
        return message, symbol
    return Codec(push, pop)

def _discretize(cdf, ppf, low, high, bin_prec, coding_prec):
    """
    Utility function for forming a codec given a (continuous) cdf and its
    inverse. Assumes that

        grad(cdf) >= 2 ** (bin_prec - coding_prec) / (high - low)

    so that all intervals end up with non-zero mass.
    """
    def cdf_(idx):
        x_low = low + (high - low) * idx / (1 << bin_prec)
        return np.where(
            idx >= 0, _nearest_int((1 << coding_prec) * cdf(x_low)), 0)
    enc_statfun = _cdf_to_enc_statfun(cdf_)
    def ppf_(cf):
        x_max = ppf((cf + .5) / (1 << coding_prec))
        return np.uint64(
            np.floor((1 << bin_prec) * (x_max - low) / (high - low)))
    return NonUniform(enc_statfun, ppf_, coding_prec)

def _benford_high_bits(data_prec, prec):
    def cdf(s):
        return ((np.log2((1 << data_prec) + s) - data_prec)
                / (np.log2(1 << (data_prec + 1)) - data_prec))

    def ppf(cf):
        return 2 ** data_prec * (
            2 ** (cf * (np.log2(1 << (data_prec + 1)) - data_prec)) - 1)
    return _discretize(cdf, ppf, 0, 1 << data_prec, data_prec, prec)

def Benford64():
    """
    Simple self-delimiting code for numbers x with

        2 ** 31 <= x < 2 ** 63

    with log(x) approximately uniformly distributed. Useful for coding
    vectorans stack heads.
    """
    length_push, length_pop = Uniform(5)
    # x_higher_push, x_higher_pop = _benford_high_bits(8, 16)
    def push(message, x):
        x_len = np.uint64(np.log2(x))
        x = x & ((1 << x_len) - 1)
        message, = _benford_high_bits(4, 16).push(message, x >> (x_len - 4))
        message, = BigUniform(x_len - 4).push(message, x & ((1 << (x_len - 4)) - 1))
        message, = length_push(message, x_len - 31)
        return message,

    def pop(message):
        message, x_len = length_pop(message)
        x_len = x_len + 31
        message, x_low = BigUniform(x_len - 4).pop(message)
        message, x_high = _benford_high_bits(4, 16).pop(message)
        return message, (1 << x_len) | (x_high << (x_len - 4)) | x_low
    return Codec(push, pop)
Benford64 = Benford64()

def flatten(message):
    """
    Flatten a message head and tail into a 1d array. Use this when finished
    coding to map to a message representation which can easily be saved to
    disk.

    If the message head is non-scalar it will be efficiently flattened by
    coding elements as if they were data.
    """
    return vrans.flatten(reshape_head(message, (1,)))

def unflatten(arr, shape):
    """
    Unflatten a 1d array, into a vrans message with desired shape. This is the
    inverse of flatten.
    """
    return reshape_head(vrans.unflatten(arr, (1,)), shape)

def _fold_sizes(small, big):
    sizes = [small]
    while small != big:
        small = 2 * small if 2 * small <= big else big
        sizes.append(small)
    return sizes

_fold_codec = lambda diff: substack(Benford64, lambda head: head[:diff])

def _fold_codecs(sizes):
    return [_fold_codec(diff) for diff in np.subtract(sizes[1:], sizes[:-1])]

def _resize_head_1d(message, size):
    head, tail = message
    sizes = _fold_sizes(*sorted((size, np.size(head))))
    codecs = _fold_codecs(sizes)
    if size < np.size(head):
        for (push, _), new_size in reversed(list(zip(codecs, sizes[:-1]))):
            (head, tail), = push((head[:new_size], tail), head[new_size:])
    elif size > np.size(head):
        for _, pop in codecs:
            (head, tail), head_ex = pop((head, tail))
            head = np.concatenate([head, head_ex])
    return head, tail

def reshape_head(message, shape):
    """
    Reshape the head of a message. Note that growing the head uses up
    information from the message and will fail if the message is empty.
    """
    head, tail = message
    if head.shape == shape:
        return message
    message = (np.ravel(head), tail)
    head, tail = _resize_head_1d(message, size=np.prod(shape))
    return np.reshape(head, shape), tail

def random_message(flat_len, shape, rng=np.random):
    """Generate a random vrans stack."""
    arr = rng.randint(1 << 32, size=flat_len, dtype='uint32')
    return unflatten(arr, shape)

def Bernoulli(p, prec):
    """Codec for Bernoulli distributed data"""
    onemp = np.clip(_nearest_int((1 - p) * (1 << prec)), 1, (1 << prec) - 1)
    enc_statfun = _cdf_to_enc_statfun(
        lambda s: np.choose(np.int64(s), [0, onemp, 1 << prec]))
    dec_statfun = lambda cf: np.uint64(cf >= onemp)
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
    """
    Codec for categorical distributed data.
    Assume that the last dim of p contains the probability vectors,
    i.e. np.sum(p, axis=-1) == ones
    """
    cumulative_buckets = _cumulative_buckets_from_probs(p, prec)
    enc_statfun = _cdf_to_enc_statfun(_cdf_from_cumulative_buckets(cumulative_buckets))
    dec_statfun = _ppf_from_cumulative_buckets(cumulative_buckets)
    return NonUniform(enc_statfun, dec_statfun, prec)

# Inverse of np.diff
def _undiff(x):
    return np.concatenate([np.zeros_like(x, shape=(*x.shape[:-1], 1)),
                           np.cumsum(x, -1)], -1)

def CategoricalNew(weights, prec):
    """
    Assume that the last dim of weights contains the unnormalized
    probability vectors, so p = weights / np.sum(weights, axis=-1).
    """
    cumweights = _undiff(weights)
    cumfreqs = _nearest_int((1 << prec) * (cumweights / cumweights[..., -1:]))
    def enc_statfun(x):
        lower = np.take_along_axis(cumfreqs, x[..., None], -1)[..., 0]
        upper = np.take_along_axis(cumfreqs, x[..., None] + 1, -1)[..., 0]
        return lower, upper - lower
    def dec_statfun(cf):
        # One could speed this up for large alphabets by
        #   (a) Using vectorized binary search, not available in numpy
        #   (b) Using the alias method
        return np.argmin(cumfreqs <= cf[..., None], axis=-1) - 1
    return NonUniform(enc_statfun, dec_statfun, prec)

def Logistic_UnifBins(
        means, log_scales, coding_prec, bin_prec, bin_lb, bin_ub):
    """
    Codec for logistic distributed data.

    The discretization is assumed to be uniform (evenly spaced) between bin_lb
    and bin_ub.
    """
    bin_range = bin_ub - bin_lb
    def cdf(x):
        cdf_min = (x - bin_lb) / bin_range * 2 ** (bin_prec - coding_prec)
        cdf_max = 1 + (x - bin_ub) / bin_range * 2 ** (bin_prec - coding_prec)
        return np.clip(
            expit((x - means) / np.exp(log_scales)), cdf_min, cdf_max)

    def ppf(cf):
        ppf_max = bin_lb + cf * bin_range * 2 ** (coding_prec - bin_prec)
        ppf_min = bin_ub + (cf - 1) * bin_range * 2 ** (coding_prec - bin_prec)
        return np.clip(
            np.exp(log_scales) * logit(cf) + means, ppf_min, ppf_max)
    return _discretize(cdf, ppf, bin_lb, bin_ub, bin_prec, coding_prec)

def _create_logistic_mixture_buckets(means, log_scales, logit_probs, coding_prec, bin_prec,
                                     bin_lb, bin_ub):
    inv_stdv = np.exp(-log_scales)
    buckets = np.linspace(bin_lb, bin_ub, (1 << bin_prec)+1)
    buckets = np.broadcast_to(buckets, means.shape + ((1 << bin_prec)+1,))
    cdfs = inv_stdv[..., np.newaxis] * (buckets - means[..., np.newaxis])
    cdfs[..., 0] = -np.inf
    cdfs[..., -1] = np.inf
    cdfs = sigmoid(cdfs)
    prob_cpts = cdfs[..., 1:] - cdfs[..., :-1]
    mixture_probs = util.softmax(logit_probs, axis=1)
    probs = np.sum(prob_cpts * mixture_probs[..., np.newaxis], axis=1)
    return _cumulative_buckets_from_probs(probs, coding_prec)

def LogisticMixture_UnifBins(means, log_scales, logit_probs, coding_prec, bin_prec, bin_lb, bin_ub):
    """
    Codec for data from a mixture of logistic distributions.

    The discretization is assumed to be uniform between bin_lb and bin_ub.
    logit_probs are the mixture weights as logits.
    """
    cumulative_buckets = _create_logistic_mixture_buckets(means, log_scales, logit_probs,
                                                          coding_prec, bin_prec, bin_lb, bin_ub)
    enc_statfun = _cdf_to_enc_statfun(_cdf_from_cumulative_buckets(cumulative_buckets))
    dec_statfun = _ppf_from_cumulative_buckets(cumulative_buckets)
    return NonUniform(enc_statfun, dec_statfun, coding_prec)

@lru_cache()
def std_gaussian_buckets(precision):
    """
    Return the endpoints of buckets partitioning the domain of the prior. Each
    bucket has mass 1 / (1 << precision) under the prior.
    """
    return norm.ppf(np.linspace(0, 1, (1 << precision) + 1))

@lru_cache()
def std_gaussian_centres(precision):
    """
    Return the medians of buckets partitioning the domain of the prior. Each
    bucket has mass 1 / (1 << precision) under the prior.
    """
    return norm.ppf((np.arange(1 << precision) + 0.5) / (1 << precision))

def _gaussian_cdf(mean, stdd, prior_prec, post_prec):
    def cdf(idx):
        x = std_gaussian_buckets(prior_prec)[idx]
        return _nearest_int(norm.cdf(x, mean, stdd) * (1 << post_prec))
    return cdf

def _gaussian_ppf(mean, stdd, prior_prec, post_prec):
    cdf = _gaussian_cdf(mean, stdd, prior_prec, post_prec)
    def ppf(cf):
        x = norm.ppf((cf + 0.5) / (1 << post_prec), mean, stdd)
        # Binary search is faster than using the actual gaussian cdf for the
        # precisions we typically use, however the cdf is O(1) whereas search
        # is O(precision), so for high precision cdf will be faster.
        idxs = np.uint64(np.digitize(x, std_gaussian_buckets(prior_prec)) - 1)
        # This loop works around an issue which is extremely rare when we use
        # float64 everywhere but is common if we work with float32: due to the
        # finite precision of floating point arithmetic, norm.[cdf,ppf] are not
        # perfectly inverse to each other.
        while not np.all((cdf(idxs) <= cf) & (cf < cdf(idxs + 1))):
            idxs = np.select(
                [cf < cdf(idxs), cf >= cdf(idxs + 1)],
                [idxs - 1,       idxs + 1           ], idxs)
        return idxs
    return ppf

def DiagGaussian_StdBins(mean, stdd, coding_prec, bin_prec):
    """
    Codec for data from a diagonal Gaussian with bins that have equal mass under
    a standard (0, I) Gaussian
    """
    enc_statfun = _cdf_to_enc_statfun(
        _gaussian_cdf(mean, stdd, bin_prec, coding_prec))
    dec_statfun = _gaussian_ppf(mean, stdd, bin_prec, coding_prec)
    return NonUniform(enc_statfun, dec_statfun, coding_prec)

def DiagGaussian_GaussianBins(mean, stdd, bin_mean, bin_stdd, coding_prec, bin_prec):
    """
    Codec for data from a diagonal Gaussian with bins that have equal mass under
    a different diagonal Gaussian
    """
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

def DiagGaussian_UnifBins(mean, stdd, bin_min, bin_max, coding_prec, n_bins, rebalanced=True):
    """
    Codec for data from a diagonal Gaussian with uniform bins.
    rebalanced=True will ensure no zero frequencies, but is slower.
    """
    if rebalanced:
        bins = np.linspace(bin_min, bin_max, n_bins)
        bins = np.broadcast_to(np.moveaxis(bins, 0, -1), mean.shape + (n_bins,))
        cdfs = norm.cdf(bins, mean[..., np.newaxis], stdd[..., np.newaxis])
        cdfs[..., 0] = 0
        cdfs[..., -1] = 1
        pmfs = cdfs[..., 1:] - cdfs[..., :-1]
        buckets = _cumulative_buckets_from_probs(pmfs, coding_prec)
        enc_statfun = _cdf_to_enc_statfun(_cdf_from_cumulative_buckets(buckets))
        dec_statfun = _ppf_from_cumulative_buckets(buckets)
    else:
        bin_width = (bin_max - bin_min)/float(n_bins)
        def cdf(idx):
            bin_ub = bin_min + idx * bin_width
            return _nearest_int(norm.cdf(bin_ub, mean, stdd) * (1 << coding_prec))
        def ppf(cf):
            x_max = norm.ppf((cf + 0.5) / (1 << coding_prec), mean, stdd)
            bin_idx = np.floor((x_max - bin_min) / bin_width)
            return np.uint64(np.minimum(n_bins-1, bin_idx))
        enc_statfun = _cdf_to_enc_statfun(cdf)
        dec_statfun = ppf
    return NonUniform(enc_statfun, dec_statfun, coding_prec)

def AutoRegressive(param_fn, data_shape, params_shape, elem_idxs, elem_codec):
    """
    Codec for data from distributions which are calculated autoregressively.
    That is, the data can be partitioned into n elements such that the
    distribution/codec for an element is only known when all previous
    elements are known. This is does not affect the push step, but does
    affect the pop step, which must be done in sequence (so is slower).

    elem_param_fn maps data to the params for the respective codecs.
    elem_idxs defines the ordering over elements within data.

    We assume that the indices within elem_idxs can also be used to index
    the params from elem_param_fn. These indexed params are then used in
    the elem_codec to actually code each element.
    """
    def push(message, data, all_params=None):
        if not all_params:
            all_params = param_fn(data)
        for idx in reversed(elem_idxs):
            elem_params = all_params[idx]
            elem_push, _ = elem_codec(elem_params, idx)
            message, = elem_push(message, data[idx].astype('uint64'))
        return message,

    def pop(message):
        data = np.zeros(data_shape, dtype=np.uint64)
        all_params = np.zeros(params_shape, dtype=np.float32)
        for idx in elem_idxs:
            all_params = param_fn(data, all_params, idx)
            elem_params = all_params[idx]
            _, elem_pop = elem_codec(elem_params, idx)
            message, elem = elem_pop(message)
            data[idx] = elem
        return message, data
    return Codec(push, pop)

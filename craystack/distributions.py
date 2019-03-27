import numpy as np
from scipy.stats import norm

from craystack.core import NonUniform
import craystack.util as util
import craystack.vectorans as vrans
from scipy.special import expit as sigmoid


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
    # TODO: a more efficient system than single stack
    head, tail = x
    flat_head = np.ravel(head)
    append, _ = Benford64
    single_stack = vrans.x_init(1)[0], tail
    for el in flat_head:
        single_stack = append(single_stack, np.array([el]).astype('uint64'))
    return vrans.flatten(single_stack)

def unflatten_benford(arr, shape):
    size = np.prod(shape)
    single_stack = vrans.unflatten(arr, (1,))
    _, pop = Benford64
    flat_head = []
    for _ in range(size):
        single_stack, elem = pop(single_stack)
        flat_head.append(elem)
    return np.array(flat_head[::-1]).reshape(shape), single_stack[1]

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

def Logistic(mean, log_scale, coding_prec, bin_prec=8):
    cumulative_buckets = _create_logistic_buckets(mean, log_scale, coding_prec, bin_prec)
    enc_statfun = _cdf_to_enc_statfun(_cdf_from_cumulative_buckets(cumulative_buckets))
    dec_statfun = _ppf_from_cumulative_buckets(cumulative_buckets)
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

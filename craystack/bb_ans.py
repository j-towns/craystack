import numpy as np
from scipy.stats import norm
import craystack.core as cs


def BBANS(prior, likelihood, posterior):
    """
    This codec is for data modelled with a latent variable model as described
    in the paper 'Practical Lossless Compression with Latent Variable Models'
    currently under review for ICLR '19.

       latent        observed
      variable         data

        ( z ) ------> ( x )

    This assumes data x is modelled via a model which includes a latent
    variable. The model has a prior p(z), likelihood p(x | z) and (possibly
    approximate) posterior q(z | x). See the paper for more details.
    """
    prior_append, prior_pop = prior

    def append(message, data):
        _, posterior_pop = posterior(data)
        message, latent = posterior_pop(message)
        likelihood_append, _ = likelihood(latent)
        message = likelihood_append(message, data)
        message = prior_append(message, latent)
        return message

    def pop(message):
        message, latent = prior_pop(message)
        _, likelihood_pop = likelihood(latent)
        message, data = likelihood_pop(message)
        posterior_append, _ = posterior(data)
        message = posterior_append(message, latent)
        return message, data
    return append, pop

def VAE(gen_net, rec_net, obs_codec, prior_prec, latent_prec):
    """
    This codec uses the BB-ANS algorithm to code data which is distributed
    according to a variational auto-encoder (VAE) model. It is assumed that the
    VAE uses an isotropic Gaussian prior and diagonal Gaussian for its
    posterior.
    """
    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    prior = cs.substack(cs.Uniform(prior_prec), z_view)

    def likelihood(latent_idxs):
        z = std_gaussian_centres(prior_prec)[latent_idxs]
        return cs.substack(obs_codec(gen_net(z)), x_view)

    def posterior(data):
        post_mean, post_stdd = rec_net(data)
        return cs.substack(_DiagGaussianLatent(
            post_mean, post_stdd, prior_prec, latent_prec), z_view)
    return BBANS(prior, likelihood, posterior)

def TwoLayerVAE(rec_net1, rec_net2,
                prior1_codec, obs_codec,
                prior_prec, latent_prec):
    """
    rec_net1 outputs params for q(z1|x)
    rec_net2 outputs params for q(z2|x)
    prior1_codec is to code z1 by p(z1|z2)
    obs_codec is to code x by p(x|z1)"""
    z1_view_prior = lambda head: head[0]
    z1_view_post = lambda head: head[1]
    z2_view = lambda head: head[2]
    x_view = lambda head: head[3]

    prior_z2_append, prior_z2_pop = cs.substack(cs.Uniform(prior_prec), z2_view)

    def prior_append(message, latents):
        z1, z2 = latents
        z1_vals = std_gaussian_centres(prior_prec)[z1]
        z2_vals = std_gaussian_centres(prior_prec)[z2]
        prior_z1_append, _ = cs.substack(prior1_codec(z2_vals), z1_view_prior)
        message = prior_z1_append(message, z1_vals)
        message = prior_z2_append(message, z2)
        return message

    def prior_pop(message):
        message, z2 = prior_z2_pop(message)
        _, prior_z1_pop = cs.substack(prior1_codec(z2), z1_view_prior)
        message, z1 = prior_z1_pop(message)
        return message, (z1, z2)

    def likelihood(latents):
        """p(x|z1)"""
        z1, _ = latents
        z1_vals = std_gaussian_centres(prior_prec)[z1]
        return cs.substack(obs_codec(z1_vals), x_view)

    def posterior(data):
        """
        q(z1|x), q(z2|x)
        We assume the data doesn't need looking up, as bucket indices are
        the same as the actual data (integers in [0,1,...,n], i.e. pixel
        intensities)
        """
        mu1, sig1, h = rec_net1(data)
        mu2, sig2 = rec_net2(h)

        post_z1_append, post_z1_pop = cs.substack(_DiagGaussianLatent(
            mu1, sig1, prior_prec, latent_prec), z1_view_post)
        post_z2_append, post_z2_pop = cs.substack(_DiagGaussianLatent(
            mu2, sig2, prior_prec, latent_prec), z2_view)

        def posterior_append(message, latents):
            z1, z2 = latents
            message = post_z1_append(message, z1)
            message = post_z2_append(message, z2)
            return message

        def posterior_pop(message):
            message, z2 = post_z2_pop(message)
            message, z1 = post_z1_pop(message)
            return message, (z1, z2)

        return posterior_append, posterior_pop

    return BBANS((prior_append, prior_pop), likelihood, posterior)

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
        return cs._nearest_int(norm.cdf(x, mean, stdd) * (1 << post_prec))
    return cdf

def _gaussian_latent_ppf(mean, stdd, prior_prec, post_prec):
    def ppf(cf):
        x = norm.ppf((cf + 0.5) / (1 << post_prec), mean, stdd)
        # Binary search is faster than using the actual gaussian cdf for the
        # precisions we typically use, however the cdf is O(1) whereas search
        # is O(precision), so for high precision cdf will be faster.
        return np.uint64(np.digitize(x, std_gaussian_buckets(prior_prec)) - 1)
    return ppf

def _DiagGaussianLatent(mean, stdd, prior_prec, post_prec):
    enc_statfun = cs._cdf_to_enc_statfun(
        _gaussian_latent_cdf(mean, stdd, prior_prec, post_prec))
    dec_statfun = _gaussian_latent_ppf(mean, stdd, prior_prec, post_prec)
    return cs.NonUniform(enc_statfun, dec_statfun, post_prec)

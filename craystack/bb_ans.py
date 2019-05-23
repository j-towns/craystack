from craystack.codecs import substack, Uniform, \
    std_gaussian_centres, DiagGaussian_StdBins, Codec


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
    prior_push, prior_pop = prior

    def push(message, data):
        _, posterior_pop = posterior(data)
        message, latent = posterior_pop(message)
        likelihood_push, _ = likelihood(latent)
        message = likelihood_push(message, data)
        message = prior_push(message, latent)
        return message

    def pop(message):
        message, latent = prior_pop(message)
        likelihood_pop = likelihood(latent).pop
        message, data = likelihood_pop(message)
        posterior_push = posterior(data).push
        message = posterior_push(message, latent)
        return message, data
    return Codec(push, pop)


def VAE(gen_net, rec_net, obs_codec, prior_prec, latent_prec):
    """
    This codec uses the BB-ANS algorithm to code data which is distributed
    according to a variational auto-encoder (VAE) model. It is assumed that the
    VAE uses an isotropic Gaussian prior and diagonal Gaussian for its
    posterior.
    """
    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    prior = substack(Uniform(prior_prec), z_view)

    def likelihood(latent_idxs):
        z = std_gaussian_centres(prior_prec)[latent_idxs]
        return substack(obs_codec(gen_net(z)), x_view)

    def posterior(data):
        post_mean, post_stdd = rec_net(data)
        return substack(DiagGaussian_StdBins(
            post_mean, post_stdd, latent_prec, prior_prec), z_view)
    return BBANS(prior, likelihood, posterior)

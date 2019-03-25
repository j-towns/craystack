import craystack.core as cs
from craystack.distributions import Uniform, DiagGaussianLatent, \
    std_gaussian_centres, DiagGaussianLatentStdBins


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

    prior = cs.substack(Uniform(prior_prec), z_view)

    def likelihood(latent_idxs):
        z = std_gaussian_centres(prior_prec)[latent_idxs]
        return cs.substack(obs_codec(gen_net(z)), x_view)

    def posterior(data):
        post_mean, post_stdd = rec_net(data)
        return cs.substack(DiagGaussianLatentStdBins(
            post_mean, post_stdd, latent_prec, prior_prec), z_view)
    return BBANS(prior, likelihood, posterior)

def ResNetVAE(up_pass, rec_net_top, rec_nets, gen_net_top, gen_nets, obs_codec,
              prior_prec, latent_prec):
    """
    Codec for a ResNetVAE.
    Assume that the posterior is bidirectional -
    i.e. has a deterministic upper pass but top down sampling.
    Further assume that all latent conditionals are factorised Gaussians,
    both in the generative network p(z_n|z_{n-1})
    and in the inference network q(z_n|x, z_{n-1})

    Assume that everything is ordered bottom up
    """
    z_view = lambda head: head[0]
    x_view = lambda head: head[1]

    prior_codec = cs.substack(Uniform(prior_prec), z_view)

    def prior_append(message, latents):
        # append bottom-up
        append, _ = prior_codec
        latents, _ = latents
        for latent in latents:
            latent, _ = latent
            message = append(message, latent)
        return message

    def prior_pop(message):
        # pop top-down
        (prior_mean, prior_stdd), h_gen = gen_net_top()
        _, pop = prior_codec
        message, latent = pop(message)
        latents = [(latent, (prior_mean, prior_stdd))]
        for gen_net in reversed(gen_nets):
            previous_latent_val = prior_mean + std_gaussian_centres(prior_prec)[latent] * prior_stdd
            (prior_mean, prior_stdd), h_gen = gen_net(h_gen, previous_latent_val)
            message, latent = pop(message)
            latents.append((latent, (prior_mean, prior_stdd)))
        return message, (latents[::-1], h_gen)

    def posterior(data):
        # run deterministic upper-pass
        # TODO: The data dependency is currently handled inside the Resnet model - move outside
        contexts = up_pass()

        def posterior_append(message, latents):
            # first run the model top-down to get the params and latent vals
            latents, _ = latents

            (post_mean, post_stdd), h_rec = rec_net_top(contexts[-1])
            post_params = [(post_mean, post_stdd)]

            for rec_net, latent, context in reversed(list(zip(rec_nets, latents[1:], contexts[:-1]))):
                previous_latent, (prior_mean, prior_stdd) = latent
                previous_latent_val = prior_mean + \
                                      std_gaussian_centres(prior_prec)[previous_latent] * prior_stdd

                (post_mean, post_stdd), h_rec = rec_net(h_rec, previous_latent_val, context)
                post_params.append((post_mean, post_stdd))

            # now append bottom up
            for latent, post_param in zip(latents, reversed(post_params)):
                latent, (prior_mean, prior_stdd) = latent
                post_mean, post_stdd = post_param
                append, _ = cs.substack(DiagGaussianLatent(post_mean, post_stdd,
                                                           prior_mean, prior_stdd,
                                                           latent_prec, prior_prec),
                                        z_view)
                message = append(message, latent)
            return message

        def posterior_pop(message):
            # pop top-down
            (post_mean, post_stdd), h_rec = rec_net_top(contexts[-1])
            (prior_mean, prior_stdd), h_gen = gen_net_top()
            _, pop = cs.substack(DiagGaussianLatent(post_mean, post_stdd,
                                                    prior_mean, prior_stdd,
                                                    latent_prec, prior_prec),
                                 z_view)
            message, latent = pop(message)
            latents = [(latent, (prior_mean, prior_stdd))]
            for rec_net, gen_net, context in reversed(list(zip(rec_nets, gen_nets, contexts[:-1]))):
                previous_latent_val = prior_mean + \
                                      std_gaussian_centres(prior_prec)[latents[-1][0]] * prior_stdd

                (post_mean, post_stdd), h_rec = rec_net(h_rec, previous_latent_val, context)
                (prior_mean, prior_stdd), h_gen = gen_net(h_gen, previous_latent_val)
                _, pop = cs.substack(DiagGaussianLatent(post_mean, post_stdd,
                                                        prior_mean, prior_stdd,
                                                        latent_prec, prior_prec),
                                     z_view)
                message, latent = pop(message)
                latents.append((latent, (prior_mean, prior_stdd)))
            return message, (latents[::-1], h_gen)

        return posterior_append, posterior_pop

    def likelihood(latents):
        # get the z1 vals to condition on
        latents, h = latents
        z1_idxs, (prior_mean, prior_stdd) = latents[0]
        z1_vals = prior_mean + std_gaussian_centres(prior_prec)[z1_idxs] * prior_stdd
        return cs.substack(obs_codec(h, z1_vals), x_view)

    return BBANS((prior_append, prior_pop), likelihood, posterior)

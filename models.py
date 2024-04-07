import jax.numpy as jnp
import jax.scipy.stats as stats
from typing import NamedTuple, Mapping
from tensorflow_probability.substrates import jax as tfp
import jax

bijectors = tfp.bijectors
sigmoid_transform = bijectors.IteratedSigmoidCentered()


class Model(NamedTuple):
    dim: int
    log_prior: Mapping
    log_likelyhood: Mapping
    t_inv_map: Mapping
    jac_t_inv_map: Mapping
    log_det_jac_t_inv_map: Mapping


def LinearModel(X, y, a_zero, b_zero):
    dim = X.shape[1] + 1

    def t_inv_map(param):
        return param[: dim - 1], jnp.exp(param[dim - 1])

    def log_prior(param):
        w, sigma = t_inv_map(param)
        log_w_proba = jnp.sum(stats.norm.logpdf(w, 0, sigma))
        log_sigma_proba = stats.gamma.logpdf(sigma / a_zero, b_zero) - jnp.log(a_zero)
        return log_w_proba + log_sigma_proba

    def log_likelyhood(param):
        w, sigma = t_inv_map(param)
        preds = X @ w
        proba = jnp.sum(stats.norm.logpdf(y, preds, sigma))
        return proba

    def log_det_jac_t_inv_map(param):
        return jnp.sum(jnp.log(jnp.abs(jnp.diagonal(jac_t_inv_map(param)))))

    def jac_t_inv_map(param):
        to_return = jnp.eye(dim)
        to_return = to_return.at[dim - 1, dim - 1].set(jnp.exp(param[dim - 1]))
        return to_return

    return Model(
        dim=dim,
        log_prior=log_prior,
        log_likelyhood=log_likelyhood,
        t_inv_map=t_inv_map,
        jac_t_inv_map=jac_t_inv_map,
        log_det_jac_t_inv_map=log_det_jac_t_inv_map,
    )


def NMF_Model_PoissonGamma(data, rank, gamma_prior_shape, gamma_prior_scale):
    num_samples, num_dims = data.shape
    dim = num_samples * rank + rank * num_dims

    def t_inv_map(params):
        theta, beta = jnp.split(params, [num_samples * rank])
        theta = jnp.reshape(theta, (num_samples, rank))
        beta = jnp.reshape(beta, (rank, num_dims))
        theta = jnp.exp(theta + 1)
        beta = jnp.exp(beta + 1)
        return theta, beta

    def jac_t_inv_map(params):
        return jnp.diag(jnp.exp(params + 1))

    def log_det_jac_t_inv_map(params):
        return jnp.sum(jnp.log(jnp.exp(params + 1)))

    def log_prior(params):
        theta, beta = t_inv_map(params)
        theta_prior = stats.gamma.logpdf(
            theta / gamma_prior_scale, gamma_prior_shape
        ) - jnp.log(gamma_prior_scale)
        beta_prior = stats.gamma.logpdf(
            beta / gamma_prior_scale, gamma_prior_shape
        ) - jnp.log(gamma_prior_scale)
        return jnp.sum(theta_prior) + jnp.sum(beta_prior)

    def log_likelyhood(params):
        theta, beta = t_inv_map(params)
        reconst = theta @ beta
        log_like = stats.poisson.logpmf(data, reconst)
        return jnp.sum(log_like)

    return Model(
        dim=dim,
        log_prior=log_prior,
        log_likelyhood=log_likelyhood,
        t_inv_map=t_inv_map,
        jac_t_inv_map=jac_t_inv_map,
        log_det_jac_t_inv_map=log_det_jac_t_inv_map,
    )


def new_NMF_Model_PoissonDirExp(data, rank, dir_prior=1, exp_prior=4):
    num_samples, num_dims = data.shape
    dim = num_samples * (rank - 1) + rank * num_dims

    def t_inv_map(params):
        theta, beta = jnp.split(params, [(num_samples) * (rank - 1)])
        theta = jnp.reshape(theta, (num_samples, rank - 1))
        beta = jnp.reshape(beta, (rank, num_dims))
        theta = sigmoid_transform.forward(theta)
        beta = jnp.exp(beta + 1)
        return theta, beta

    def jac_t_inv_map(params):
        raise NotImplemented

    def log_det_jac_t_inv_map(params):
        theta, beta = jnp.split(params, [(num_samples - 1) * rank])
        theta = jnp.reshape(theta, (num_samples - 1, rank))
        log_det = jnp.sum(sigmoid_transform.forward_log_det_jacobian(theta))
        det = jnp.sum(jnp.log(jnp.exp(beta + 1)))
        return log_det + det

    def log_prior(params):
        theta, beta = t_inv_map(params)

        theta_prior = jnp.log(
            stats.dirichlet.pdf(theta.T, jnp.full(shape=(rank,), fill_value=dir_prior))
        )
        beta_prior = jnp.sum(stats.expon.logpdf(beta, scale=exp_prior))
        return jnp.sum(theta_prior) + jnp.sum(beta_prior)

    def log_likelyhood(params):
        theta, beta = t_inv_map(params)
        reconst = theta @ beta
        log_like = stats.poisson.logpmf(data, reconst)
        return jnp.sum(log_like)

    return Model(
        dim=dim,
        log_prior=log_prior,
        log_likelyhood=log_likelyhood,
        t_inv_map=t_inv_map,
        jac_t_inv_map=jac_t_inv_map,
        log_det_jac_t_inv_map=log_det_jac_t_inv_map,
    )


def old_NMF_Model_PoissonDirExp(data, rank, dir_prior=1, exp_prior=4):
    num_samples, num_dims = data.shape
    dim = (num_samples - 1) * rank + rank * num_dims

    def t_inv_map(params):
        theta, beta = jnp.split(params, [(num_samples - 1) * rank])
        theta = jnp.reshape(theta, (num_samples - 1, rank))
        beta = jnp.reshape(beta, (rank, num_dims))
        theta = sigmoid_transform.forward(theta.T).T
        beta = jnp.exp(beta + 1)
        return theta, beta

    def jac_t_inv_map(params):
        raise NotImplemented

    def log_det_jac_t_inv_map(params):
        theta, beta = jnp.split(params, [(num_samples - 1) * rank])
        theta = jnp.reshape(theta, (num_samples - 1, rank))
        log_det = jnp.sum(sigmoid_transform.forward_log_det_jacobian(theta.T))
        det = jnp.sum(jnp.log(jnp.exp(beta + 1)))
        return log_det + det

    def log_prior(params):
        theta, beta = t_inv_map(params)

        theta_prior = jnp.log(
            stats.dirichlet.pdf(
                theta, jnp.full(shape=(num_samples,), fill_value=dir_prior)
            )
        )
        beta_prior = jnp.sum(stats.expon.logpdf(beta, scale=exp_prior))
        return jnp.sum(theta_prior) + jnp.sum(beta_prior)

    def log_likelyhood(params):
        theta, beta = t_inv_map(params)
        reconst = theta @ beta
        log_like = stats.poisson.logpmf(data, reconst)
        return jnp.sum(log_like)

    return Model(
        dim=dim,
        log_prior=log_prior,
        log_likelyhood=log_likelyhood,
        t_inv_map=t_inv_map,
        jac_t_inv_map=jac_t_inv_map,
        log_det_jac_t_inv_map=log_det_jac_t_inv_map,
    )


def HLR_Model(data, beta_prior=100, alpha_prior=1):
    dim = 34

    def jac_t_inv_map(params):
        raise NotImplementedError("methods not implemented")

    def log_det_jac_t_inv_map(params):
        return 0.0

    def t_inv_map(params):
        beta = params[:5]
        alpha_age = params[5:9]
        alpha_regions = params[9:14]
        alpha_edu = params[14:18]
        alpha_age_edu = params[18:34]
        return beta, alpha_age, alpha_regions, alpha_edu, alpha_age_edu

    def log_likelyhood(params):
        beta, alpha_age, alpha_regions, alpha_edu, alpha_age_edu = t_inv_map(params)
        regression = (
            beta[0]
            + data["female"] * beta[1]
            + data["black"] * beta[2]
            + data["female_black"] * beta[3]
            + data["prev"] * beta[4]
        )
        prob_1 = (
            regression
            + alpha_age[data["age"]]
            + alpha_edu[data["edu"]]
            + alpha_regions[data["regions"]]
            + alpha_age_edu[data["age_edu"]]
            + beta[0]
        )
        prob1 = jax.nn.sigmoid(prob_1)
        likelyhood = prob1 * (2 * data["y"] - 1) + 1 - data["y"]
        return jnp.sum(jnp.log(likelyhood))

    def log_prior(params):
        beta, alpha_age, alpha_regions, alpha_edu, alpha_age_edu = t_inv_map(params)
        alpha = jnp.hstack((alpha_age, alpha_regions, alpha_edu, alpha_age_edu))
        log_prior = jnp.sum(stats.norm.logpdf(beta, scale=beta_prior)) + jnp.sum(
            stats.norm.logpdf(alpha, scale=alpha_prior)
        )
        return log_prior

    return Model(
        dim=dim,
        t_inv_map=t_inv_map,
        log_prior=log_prior,
        log_likelyhood=log_likelyhood,
        log_det_jac_t_inv_map=log_det_jac_t_inv_map,
        jac_t_inv_map=jac_t_inv_map,
    )

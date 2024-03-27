import jax.numpy as jnp
import jax
import jax.random as jrandom
from tensorflow_probability.substrates import jax as tfp

bijectors = tfp.bijectors
sigmoid_transform = bijectors.IteratedSigmoidCentered()
from models import Model


def mean_field_obj(param, sample, model):
    mu = param["mu"]
    sigma = param["sigma"]
    sample = sample * sigma + mu
    log_likelyhood = model.log_likelyhood(sample)
    log_prior = model.log_prior(sample)
    log_det_jac = jnp.abs(model.log_det_jac_t_inv_map(sample))
    entropy = 0.5 * jnp.sum(jnp.log((2 * jnp.pi * jnp.e) * sigma**2))
    return log_likelyhood + log_prior + log_det_jac + entropy


mean_field_grad = jax.grad(mean_field_obj, argnums=0)
mean_field_grad_val = jax.value_and_grad(mean_field_obj, argnums=0)
v_mean_field_grad_val = jax.vmap(
    mean_field_grad_val,
    in_axes=(None, 0, None),
    out_axes=(0, {"mu": 0, "sigma": 0}),
)


@jax.jit
def adaptive_step_size(iter, s_k, grads, stepsize=0.5, momentum=0.1, tau=1):
    s_kplus = jax.tree_map(
        lambda x, y: momentum * x**2 + (1 - momentum) * y, grads, s_k
    )
    lead_const = stepsize * (iter + 1) ** (-0.5 + 1e-6)
    rho_k = jax.tree_map(lambda x: lead_const / (tau + x), s_kplus)
    return rho_k, s_kplus


@jax.jit
def sgd_update_params(params, new_params, step_size):
    return jax.tree_map(lambda x, y, z: x + z * y, params, new_params, step_size)


class mean_field_advi:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.params = {
            "mu": jnp.zeros((self.model.dim,)) * 1.0,
            "sigma": jnp.ones((self.model.dim,)) * 1.0,
        }
        self.old_params_grad = jax.tree_map(lambda x: jnp.zeros_like(x), self.params)
        self.obj_fun = mean_field_obj
        self.grad_fun = jax.grad(mean_field_grad)

    def sgd_update_params(self, params, new_params, step_size):
        return jax.tree_map(lambda x, y, z: x + z * y, params, new_params, step_size)

    def run_advi(
        self,
        key,
        num_sample,
        num_iter,
        learning_rate,
        print_every=100,
        alpha=0.1,
        adaptive=False,
    ):
        loss_val = []
        jit_wrapper = lambda x, y: v_mean_field_grad_val(x, y, self.model)
        val_grad = jax.jit(jit_wrapper)
        for i in range(num_iter):
            key, _ = jrandom.split(key)
            samples = jrandom.normal(key, shape=(num_sample, self.model.dim))
            vals, grads = val_grad(self.params, samples)

            loss_val.append(jnp.mean(vals))
            mean_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)
            if adaptive:
                if i == 0:
                    s_k = jax.tree_map(lambda x: x**2, mean_grad)
                step_size, s_k = adaptive_step_size(
                    i, s_k, mean_grad, stepsize=learning_rate, momentum=alpha
                )

            step_size = jax.tree_map(
                lambda x: learning_rate * jnp.ones(x.shape), self.params
            )
            self.params = sgd_update_params(self.params, mean_grad, step_size)

            if i % print_every == 0:
                print(loss_val[-1])
                # break
        return loss_val

    def sample(self, key):
        sample = (
            jrandom.normal(key, shape=(self.model.dim,)) * self.params["sigma"]
            + self.params["mu"]
        )
        return self.model.t_inv_map(sample)

    def sample_advi(self, key):
        r_norm = jrandom.normal(key, shape=(dim, ))
        return r_norm*self.params['sigma']+self.params['mu']
"""Learned-drift MALA: eqs. 8–10 and Algorithm 2. q is the Gaussian, not Q_phi."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ntns.mlp import mlp_apply

T_STAR = 0.5


def mala_proposal(x, velocity, eps, xi):
    """eq. 8: x' = x + ε v_phi(x, t*) + sqrt(2ε) ξ,  ξ ~ N(0, I)."""
    eps = jnp.asarray(eps, dtype=x.dtype)
    return x + eps * velocity + jnp.sqrt(2.0 * eps) * xi


def log_gaussian_transition(x_to, x_from, velocity_from, eps):
    """log q(x_to | x_from) = log N(x_to; x_from + ε v(x_from), 2ε I)."""
    eps = jnp.asarray(eps, dtype=x_to.dtype)
    mean = x_from + eps * velocity_from
    var = 2.0 * eps
    dim = x_to.shape[-1]
    diff2 = jnp.sum((x_to - mean) ** 2, axis=-1)
    return -0.5 * dim * jnp.log(2.0 * jnp.pi * var) - 0.5 * diff2 / var


def mh_log_ratio(x, x_prop, v_x, v_prop, log_pi_x, log_pi_prop, eps):
    """log[ π(x') q(x|x') / (π(x) q(x'|x)) ]. Reverse q uses v at x'."""
    log_q_fwd = log_gaussian_transition(x_prop, x, v_x, eps)
    log_q_rev = log_gaussian_transition(x, x_prop, v_prop, eps)
    return log_pi_prop - log_pi_x + log_q_rev - log_q_fwd


def decide_accept(u, log_alpha, e_prop, e_star):
    """Alg. 2: accept iff u < min(1, exp(log_alpha)) AND E(x') < E*."""
    log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)
    mh_ok = jnp.log(u) < jnp.minimum(log_alpha, 0.0)
    return mh_ok & (e_prop < e_star)


def mala_step(key, x, params, eps, e_star, energy_fn, log_prior_fn, t_star=T_STAR):
    """One learned-drift MALA step on a batch x of shape (k, dim)."""
    k_xi, k_u = jax.random.split(key)
    t_star = jnp.asarray(t_star, dtype=x.dtype)
    v_x = mlp_apply(params, x, t_star)
    xi = jax.random.normal(k_xi, x.shape, dtype=x.dtype)
    x_prop = mala_proposal(x, v_x, eps, xi)
    v_prop = mlp_apply(params, x_prop, t_star)
    e_prop = energy_fn(x_prop)
    log_alpha = mh_log_ratio(
        x,
        x_prop,
        v_x,
        v_prop,
        log_prior_fn(x),
        log_prior_fn(x_prop),
        eps,
    )
    u = jax.random.uniform(k_u, (x.shape[0],), dtype=x.dtype)
    acc = decide_accept(u, log_alpha, e_prop, e_star)
    x_new = jnp.where(acc[:, None], x_prop, x)
    return x_new, acc


def mutate(
    key,
    x,
    params,
    eps,
    e_star,
    energy_fn,
    log_prior_fn,
    T: int,
    t_star=T_STAR,
):
    """T steps of KE* (Alg. 2). Returns final x and mean accept rate."""

    def body(x, key):
        x, acc = mala_step(
            key, x, params, eps, e_star, energy_fn, log_prior_fn, t_star
        )
        return x, acc.astype(x.dtype)

    keys = jax.random.split(key, T)
    x, accs = jax.lax.scan(body, x, keys)
    return x, jnp.mean(accs)


def adapt_log_eps(
    log_eps,
    mean_accept,
    gamma: float = 0.5,
    alpha_target: float = 0.3,
    kappa: float = 0.0,
    iteration: int = 1,
):
    """eq. 17 with κ=0, γ=0.5, α_target=0.3: log ε ← log ε + γ_i (ā - α)."""
    i = jnp.maximum(jnp.asarray(iteration, dtype=log_eps.dtype), 1.0)
    gain = gamma / (i ** kappa)
    return log_eps + gain * (mean_accept - alpha_target)

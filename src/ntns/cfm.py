"""Conditional flow matching, eq. 5: linear interpolant, independent coupling."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ntns.mlp import mlp_apply


def interpolant(x0, x1, t):
    """x_t = t x1 + (1-t) x0. t scalar or shape matching x0 without the last axis."""
    t = jnp.asarray(t, dtype=x0.dtype)
    if t.ndim == x0.ndim - 1:
        t = t[..., None]
    return t * x1 + (1.0 - t) * x0


def target_velocity(x0, x1):
    """Regression target x1 - x0 (independent coupling, linear interpolant)."""
    return x1 - x0


def cfm_loss(params, x0, x1, t):
    """L_CFM = mean ||v_phi(x_t, t) - (x1 - x0)||^2."""
    xt = interpolant(x0, x1, t)
    v = mlp_apply(params, xt, t)
    err = v - target_velocity(x0, x1)
    return jnp.mean(jnp.sum(err * err, axis=-1))


def fit_mean_subtracted_mvn(x, ridge: float = 1e-4):
    """Empirical mean and covariance of mean-subtracted live points, plus ridge."""
    mu = jnp.mean(x, axis=0)
    xc = x - mu
    n = x.shape[0]
    denom = jnp.maximum(n - 1.0, 1.0)
    cov = (xc.T @ xc) / denom
    cov = cov + ridge * jnp.eye(x.shape[-1], dtype=x.dtype)
    return mu, cov


def sample_mvn(key, mean, cov, n: int):
    L = jnp.linalg.cholesky(cov)
    z = jax.random.normal(key, (n,) + mean.shape, dtype=mean.dtype)
    return mean + z @ L.T


def _adam_init(params):
    zeros = jax.tree.map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def _adam_step(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = jax.tree.map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = jax.tree.map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = jax.tree.map(lambda m: m / (1.0 - b1 ** t), m)
    vhat = jax.tree.map(lambda v: v / (1.0 - b2 ** t), v)
    params = jax.tree.map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        mhat,
        vhat,
    )
    return params, {"m": m, "v": v, "t": t}


def train_cfm(
    key,
    params,
    live,
    n_steps: int,
    batch_size: int | None = None,
    lr: float = 1e-2,
    opt_state=None,
    ridge: float = 1e-4,
):
    """Warm-start CFM on current survivors. p0 = N(mu, Sigma) fitted to live."""
    mu, cov = fit_mean_subtracted_mvn(live, ridge=ridge)
    n = int(live.shape[0])
    if batch_size is None:
        batch_size = n
    batch_size = min(int(batch_size), n)
    if opt_state is None:
        opt_state = _adam_init(params)

    def step(carry, key):
        params, opt_state = carry
        k0, k1, k2 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (batch_size,), 0, n)
        x1 = live[idx]
        x0 = sample_mvn(k0, mu, cov, batch_size)
        t = jax.random.uniform(k2, (batch_size,), dtype=live.dtype)

        def loss_fn(p):
            return cfm_loss(p, x0, x1, t)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        params, opt_state = _adam_step(params, grads, opt_state, lr)
        return (params, opt_state), loss

    if n_steps <= 0:
        return params, opt_state, mu, cov
    keys = jax.random.split(key, n_steps)
    (params, opt_state), _ = jax.lax.scan(step, (params, opt_state), keys)
    return params, opt_state, mu, cov

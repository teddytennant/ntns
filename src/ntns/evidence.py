"""Constrained-prior volumes and Z(β) quadrature, eqs. 2–4."""

from __future__ import annotations

import jax.numpy as jnp
import jax.nn


def shrinkage_step(X, n_live):
    """Exact order-statistic mean: X' = X * n/(n+1) after deleting one of n live."""
    n_live = jnp.asarray(n_live, dtype=X.dtype)
    return X * (n_live / (n_live + 1.0))


def partition_function(energies, X_before, X_after, beta):
    """eq. 4: Z(β) = sum_i exp(-β E_i) (X_{i-1} - X_i)."""
    weights = jnp.exp(-beta * energies) * (X_before - X_after)
    return jnp.sum(weights)


def log_partition_function(energies, X_before, X_after, beta):
    """log of eq. 4 via log-sum-exp."""
    dX = jnp.maximum(X_before - X_after, 0.0)
    log_w = -beta * energies + jnp.log(dX)
    finite = jnp.isfinite(log_w) & (dX > 0)
    log_w = jnp.where(finite, log_w, -jnp.inf)
    return jax.nn.logsumexp(log_w)


def log_z_live(live_energies, remaining_X, beta):
    """Remaining live contribution: (X / m) sum_j exp(-β E_j)."""
    m = live_energies.shape[0]
    log_mean = jax.nn.logsumexp(-beta * live_energies) - jnp.log(m)
    log_X = jnp.log(jnp.maximum(remaining_X, 0.0))
    return log_X + log_mean

"""Tiny time-conditioned MLP velocity (SiLU). Not an EGNN."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def init_mlp(key, dim: int, hidden: int = 32, n_hidden: int = 2):
    """Params for concat([x, t]) -> hidden^n_hidden -> dim. He init for SiLU."""
    sizes = [dim + 1] + [hidden] * n_hidden + [dim]
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, d_in, d_out in zip(keys, sizes[:-1], sizes[1:]):
        w = jax.random.normal(k, (d_in, d_out)) * jnp.sqrt(2.0 / d_in)
        b = jnp.zeros((d_out,))
        params.append((w, b))
    return params


def mlp_apply(params, x, t):
    """v_phi(x, t). x: (..., dim), t: scalar or broadcastable to x.shape[:-1]."""
    t = jnp.asarray(t, dtype=x.dtype)
    t = jnp.broadcast_to(t, x.shape[:-1])[..., None]
    h = jnp.concatenate([x, t], axis=-1)
    last = len(params) - 1
    for i, (w, b) in enumerate(params):
        h = h @ w + b
        if i < last:
            h = jax.nn.silu(h)
    return h

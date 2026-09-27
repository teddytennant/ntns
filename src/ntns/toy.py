"""Toy prior and energies. Not Lennard-Jones."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def gaussian_energy(x):
    """E(x) = 0.5 ||x||^2. Analytic Z(β) under N(0,I) prior is (1+β)^{-d/2}."""
    return 0.5 * jnp.sum(x * x, axis=-1)


def double_well_energy(x):
    """1D/first-coord double well (x0^2 - 1)^2 plus harmonic remainder."""
    x0 = x[..., 0]
    rest = x[..., 1:]
    return (x0 * x0 - 1.0) ** 2 + 0.5 * jnp.sum(rest * rest, axis=-1)


def log_standard_normal(x):
    """log π(x) for isotropic N(0, I), up to a constant kept for MH ratios."""
    dim = x.shape[-1]
    return -0.5 * jnp.sum(x * x, axis=-1) - 0.5 * dim * jnp.log(2.0 * jnp.pi)


def sample_standard_normal(key, n, dim):
    return jax.random.normal(key, (n, dim))


def analytic_log_z_gaussian(beta, dim: int):
    """log ∫ exp(-β * 0.5||x||^2) N(x; 0, I) dx = -(d/2) log(1+β)."""
    return -0.5 * dim * jnp.log1p(beta)

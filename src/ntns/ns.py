"""Nested sampling outer loop, Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from ntns.cfm import train_cfm
from ntns.evidence import log_partition_function, log_z_live, shrinkage_step
from ntns.mala import T_STAR, adapt_log_eps, mutate
from ntns.mlp import init_mlp


@dataclass
class NSResult:
    dead_energies: jnp.ndarray
    X_before: jnp.ndarray
    X_after: jnp.ndarray
    live_points: jnp.ndarray
    live_energies: jnp.ndarray
    remaining_X: float
    e_star: float
    log_eps: float
    params: object
    n_iterations: int
    mean_accepts: jnp.ndarray

    def log_Z(self, beta):
        if self.dead_energies.size == 0:
            return jnp.array(-jnp.inf)
        return log_partition_function(
            self.dead_energies, self.X_before, self.X_after, beta
        )

    def log_Z_live(self, beta):
        return log_z_live(self.live_energies, self.remaining_X, beta)

    def log_Z_total(self, beta):
        return jnp.logaddexp(self.log_Z(beta), self.log_Z_live(beta))


def nested_sample(
    key,
    energy_fn,
    log_prior_fn,
    sample_prior_fn,
    dim: int,
    m: int = 32,
    k: int = 1,
    T: int = 8,
    max_iterations: int = 20,
    delta: float = -3.0,
    cfm_steps: int = 16,
    cfm_lr: float = 1e-2,
    hidden: int = 32,
    n_hidden: int = 2,
    eps0: float = 0.1,
    gamma: float = 0.5,
    kappa: float = 0.0,
    alpha_target: float = 0.3,
    t_star: float = T_STAR,
    n_init: int | None = None,
    beta_ref: float = 1.0,
    batch_size: int | None = None,
):
    """Algorithm 1. Stops at max_iterations or log Z_live - log Z < δ."""
    if k < 1 or k >= m:
        raise ValueError("need 1 <= k < m")
    if n_init is None:
        n_init = m

    key, k_init, k_mlp = jax.random.split(key, 3)
    x = sample_prior_fn(k_init, n_init)
    e = energy_fn(x)
    if n_init > m:
        idx = jnp.argsort(e)[:m]
        x, e = x[idx], e[idx]

    params = init_mlp(k_mlp, dim, hidden=hidden, n_hidden=n_hidden)
    opt_state = None
    log_eps = jnp.array(jnp.log(eps0), dtype=x.dtype)
    X = jnp.array(1.0, dtype=x.dtype)

    dead_E, dead_Xb, dead_Xa = [], [], []
    mean_accepts = []
    e_star = jnp.inf
    n_iters = 0

    for i in range(1, max_iterations + 1):
        n_live = int(x.shape[0])
        order = jnp.argsort(e)
        deleted = order[n_live - k :][::-1]
        survivors = order[: n_live - k]
        e_star = jnp.min(e[deleted])

        n = n_live
        X_cur = X
        for j in range(k):
            X_next = shrinkage_step(X_cur, n)
            dead_E.append(e[deleted[j]])
            dead_Xb.append(X_cur)
            dead_Xa.append(X_next)
            X_cur = X_next
            n -= 1
        X = X_cur

        surv_x = x[survivors]
        surv_e = e[survivors]
        # Ties at the contour (cloned parents) can leave a survivor with E == min(deleted).
        # The kernel constraint is strict, E < E*, so bump E* by 1 ulp when needed.
        max_surv = jnp.max(surv_e)
        e_star = jnp.where(
            max_surv < e_star, e_star, jnp.nextafter(max_surv, jnp.asarray(jnp.inf, dtype=max_surv.dtype))
        )

        key, k_cfm, k_par, k_mut = jax.random.split(key, 4)
        params, opt_state, _, _ = train_cfm(
            k_cfm,
            params,
            surv_x,
            n_steps=cfm_steps,
            batch_size=batch_size,
            lr=cfm_lr,
            opt_state=opt_state,
        )

        n_surv = n_live - k
        if k <= n_surv:
            parent_idx = jax.random.choice(k_par, n_surv, shape=(k,), replace=False)
        else:
            parent_idx = jax.random.randint(k_par, (k,), 0, n_surv)
        parents = surv_x[parent_idx]
        eps = jnp.exp(log_eps)
        new_x, mean_acc = mutate(
            k_mut,
            parents,
            params,
            eps,
            e_star,
            energy_fn,
            log_prior_fn,
            T,
            t_star=t_star,
        )
        mean_accepts.append(mean_acc)
        log_eps = adapt_log_eps(
            log_eps,
            mean_acc,
            gamma=gamma,
            alpha_target=alpha_target,
            kappa=kappa,
            iteration=i,
        )
        log_eps = jnp.clip(log_eps, -8.0, 4.0)

        new_e = energy_fn(new_x)
        x = jnp.concatenate([surv_x, new_x], axis=0)
        e = jnp.concatenate([surv_e, new_e], axis=0)
        n_iters = i

        E_dead = jnp.stack(dead_E)
        Xb = jnp.stack(dead_Xb)
        Xa = jnp.stack(dead_Xa)
        log_Z = log_partition_function(E_dead, Xb, Xa, beta_ref)
        log_Zl = log_z_live(e, X, beta_ref)
        if (log_Zl - log_Z) < delta:
            break

    return NSResult(
        dead_energies=jnp.stack(dead_E) if dead_E else jnp.zeros((0,), dtype=x.dtype),
        X_before=jnp.stack(dead_Xb) if dead_Xb else jnp.zeros((0,), dtype=x.dtype),
        X_after=jnp.stack(dead_Xa) if dead_Xa else jnp.zeros((0,), dtype=x.dtype),
        live_points=x,
        live_energies=e,
        remaining_X=float(X),
        e_star=float(e_star),
        log_eps=float(log_eps),
        params=params,
        n_iterations=n_iters,
        mean_accepts=jnp.stack(mean_accepts) if mean_accepts else jnp.zeros((0,)),
    )

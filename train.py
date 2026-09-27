"""CPU/GPU smoke: 2D Gaussian energy, writes results.json."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from ntns import (
    analytic_log_z_gaussian,
    gaussian_energy,
    log_standard_normal,
    nested_sample,
    sample_standard_normal,
)


def main():
    dim = 2
    key = jax.random.PRNGKey(0)

    def sample_prior(k, n):
        return sample_standard_normal(k, n, dim)

    result = nested_sample(
        key,
        energy_fn=gaussian_energy,
        log_prior_fn=log_standard_normal,
        sample_prior_fn=sample_prior,
        dim=dim,
        m=32,
        k=4,
        T=6,
        max_iterations=12,
        delta=-3.0,
        cfm_steps=8,
        cfm_lr=1e-2,
        hidden=16,
        n_hidden=2,
        eps0=0.1,
    )

    betas = [0.5, 1.0, 2.0]
    logZ = {str(b): float(result.log_Z(b)) for b in betas}
    logZ_total = {str(b): float(result.log_Z_total(b)) for b in betas}
    analytic = {str(b): float(analytic_log_z_gaussian(b, dim)) for b in betas}
    live_ok = bool(jnp.all(result.live_energies < result.e_star))

    payload = {
        "dim": dim,
        "m": 32,
        "k": 4,
        "T": 6,
        "n_iterations": result.n_iterations,
        "n_dead": int(result.dead_energies.shape[0]),
        "remaining_X": result.remaining_X,
        "e_star": result.e_star,
        "log_eps": result.log_eps,
        "live_all_below_Estar": live_ok,
        "logZ_dead": logZ,
        "logZ_total": logZ_total,
        "analytic_logZ": analytic,
        "mean_accept_last": float(result.mean_accepts[-1])
        if result.mean_accepts.size
        else None,
    }
    out = Path("results.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"NTNS 2D Gaussian: iters={result.n_iterations} "
        f"logZ(1)={logZ['1.0']:.3f} analytic={analytic['1.0']:.3f} "
        f"E*={result.e_star:.3f} wrote {out}"
    )


if __name__ == "__main__":
    main()

Neural Transport Nested Sampling (https://arxiv.org/abs/2609.29413) as a JAX kernel on a toy Gaussian energy: nested sampling (Alg. 1) with a conditional-flow-matching velocity (eq. 5) as the drift of a Metropolis-adjusted Langevin proposal (eqs. 8–10, Alg. 2). Prior-volume shrinkage uses the exact order-statistic mean \(X_i = X_{i-1}\, n/(n+1)\) at the current live count \(n\) (so \(k=1\) gives \(X_i=(m/(m+1))^i\)), not the log-mean \(\exp(-i/m)\). The CFM base \(p_0\) is \(\mathcal{N}(\mu,\Sigma)\) with \(\mu,\Sigma\) the empirical mean and mean-subtracted covariance of the current survivors.

Install: `pip install -e .`

Run: `pytest` and `python train.py`

What does not match the paper: no EGNN, no LJ-13/LJ-55, tiny MLP velocity, live count and T far below Table 2, no DW-4 molecular benchmark.

"""Bayesian hyperparameter optimisation: Gaussian-process surrogate + Expected Improvement (Snoek et al., 2012).

Implemented with scikit-learn/scipy only, so it runs identically on Colab and locally without extra packages.
Loop: evaluate n_init quasi-random (Sobol) configurations → fit a GP to (config → CV score) → choose the next config
that maximises Expected Improvement over the best score so far → evaluate it → repeat. Every trial is appended to a CSV,
so an interrupted run resumes where it stopped.
"""
import os, time, json, math
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel


class Space:
    """Each dimension: (name, low, high, kind) with kind in {"log", "lin", "int", "logint"}. Optimiser works in [0,1]^d."""
    def __init__(self, dims):
        self.dims = dims

    def to_params(self, u):
        out = {}
        for (name, lo, hi, kind), x in zip(self.dims, u):
            if kind in ("log", "logint"):
                v = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo)))
            else:
                v = lo + x * (hi - lo)
            out[name] = int(round(v)) if kind in ("int", "logint") else float(v)
        return out

    @property
    def d(self):
        return len(self.dims)


def expected_improvement(mu, sigma, best, xi=0.01):
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - best - xi) / sigma
    return (mu - best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def optimise(objective, space, n_trials, n_init, log_path, seed=42, n_candidates=20000, xi=0.01):
    """objective(params) -> dict with key 'score' (to MAXIMISE) plus anything to log. Resumes from log_path if present."""
    rng = np.random.default_rng(seed)
    done = pd.read_csv(log_path) if os.path.isfile(log_path) else pd.DataFrame()
    U = [json.loads(u) for u in done["u"]] if len(done) else []
    Y = list(done["score"]) if len(done) else []
    sobol = qmc.Sobol(d=space.d, scramble=True, seed=seed).random(max(n_init, 1))
    for t in range(len(U), n_trials):
        if t < n_init:
            u = sobol[t]; how = "sobol-init"
        else:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(space.d, 0.3), length_scale_bounds=(1e-2, 10.0), nu=2.5) \
                     + WhiteKernel(1e-4, (1e-8, 1e-1))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=seed + t)
            gp.fit(np.array(U), np.array(Y))
            cand = qmc.Sobol(d=space.d, scramble=True, seed=seed + 1000 + t).random(n_candidates)
            mu, sd = gp.predict(cand, return_std=True)
            ei = expected_improvement(mu, sd, max(Y), xi)
            u = cand[int(np.argmax(ei))]; how = "gp-ei"
        params = space.to_params(u)
        t0 = time.time(); res = objective(params); dt = time.time() - t0
        U.append(list(map(float, u))); Y.append(res["score"])
        row = {"trial": t, "how": how, **params, **res, "seconds": dt, "u": json.dumps(list(map(float, u)))}
        pd.DataFrame([row]).to_csv(log_path, mode="a", header=not os.path.isfile(log_path), index=False)
        print(f"trial {t:2d} [{how}] score {res['score']:.4f} (best {max(Y):.4f}) {dt:.0f}s  {params}", flush=True)
    return pd.read_csv(log_path)

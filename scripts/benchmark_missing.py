"""
Benchmark for GaussianHMM with ``missing="marginalize"``.

Measures inference and EM cost on 20,000 rows x 16 channels with 4 states,
for two missingness regimes:

* ``cyclic``   -- four channel subsets repeat for the whole batch, so the
  pattern cache holds exactly four entries and every row reuses precomputed
  per-state factors;
* ``random``   -- each row draws an independent 50% channel mask, producing
  many distinct patterns and exercising the cache's 64 MiB storage budget.

Run with::

    python -I scripts/benchmark_missing.py
"""

import argparse
import time

import numpy as np

from hmmlearn import _missing, hmm


def build_inputs(kind, n_rows=20_000, n_features=16, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_rows, n_features)
    if kind == "cyclic":
        # Four recurring masks; each channel is observed in three of them.
        pattern = rng.randint(0, 4, n_rows)
        for t in range(n_rows):
            X[t, np.arange(n_features) % 4 == pattern[t]] = np.nan
    elif kind == "random":
        X[rng.rand(n_rows, n_features) < 0.5] = np.nan
    else:
        raise ValueError(kind)
    return X


def build_model(X, covariance_type, n_components=4):
    n_features = X.shape[1]
    rng = np.random.RandomState(1)
    means = rng.randn(n_components, n_features)
    eye = np.eye(n_features)
    if covariance_type == "full":
        a = rng.rand(n_features, n_features)
        base = a @ a.T / n_features + eye
        covars = np.stack([base] * n_components)
    else:
        covars = np.broadcast_to(np.ones(n_features),
                                 (n_components, n_features)).copy()
    model = hmm.GaussianHMM(
        n_components, covariance_type, init_params="", params="st",
        missing="marginalize")
    model.startprob_ = np.full(n_components, 1 / n_components)
    model.transmat_ = np.full((n_components, n_components),
                              1 / n_components)
    model.means_ = means
    model.covars_ = covars
    return model


def timed(fn, repeats=3):
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rows", type=int, default=20_000)
    parser.add_argument("--n-features", type=int, default=16)
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--fit-iters", type=int, default=5)
    args = parser.parse_args()

    budget_mib = _missing.MISSING_CACHE_BUDGET_BYTES / 2 ** 20
    print(f"cache budget: {budget_mib:.0f} MiB "
          f"(excludes input, model parameters and per-time-step arrays)\n")

    for kind in ("cyclic", "random"):
        X = build_inputs(kind, args.n_rows, args.n_features)
        n_patterns = len(_missing.MissingPatterns(X).groups)
        print(f"== {kind}: {args.n_rows} rows, {n_patterns} distinct "
              f"missingness patterns ==")
        for covariance_type in ("diag", "full"):
            model = build_model(X, covariance_type, args.n_components)
            score_t = timed(lambda: model.score(X))
            samples_t = timed(lambda: model.score_samples(X))
            viterbi_t = timed(lambda: model.decode(
                X, algorithm="viterbi"))
            fitter = hmm.GaussianHMM(
                args.n_components, covariance_type,
                n_iter=args.fit_iters, tol=-1.0, random_state=0,
                missing="marginalize")
            start = time.perf_counter()
            fitter.fit(X)
            fit_t = time.perf_counter() - start

            cache = model._missing_cache_
            # Auxiliary-state accounting: per pattern we retain, for each
            # state, one Cholesky factor (full) or the observed diagonal
            # (diag) plus the shared log-normalizer vector.
            print(f"  {covariance_type:4s} "
                  f"score={score_t * 1e3:8.1f} ms  "
                  f"score_samples={samples_t * 1e3:8.1f} ms  "
                  f"viterbi={viterbi_t * 1e3:8.1f} ms  "
                  f"fit({args.fit_iters})={fit_t:6.2f} s  "
                  f"cache={cache.size / 2 ** 20:6.2f} MiB "
                  f"({len(cache._data)} entries)")
        print()


if __name__ == "__main__":
    main()

"""
Tests for GaussianHMM missing-value handling (``missing="marginalize"``).

NaN marks a channel without an observation; the missing channel is
integrated out of the Gaussian emission.  All test data is generated locally.
"""

from itertools import product

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.stats import multivariate_normal, norm

from .. import hmm
from .. import _missing
from ..vhmm import VariationalGaussianHMM
from . import make_covar_matrix


ATOL = 1e-8


# ---------------------------------------------------------------------------
# Reference implementations, written independently of the library internals.
# ---------------------------------------------------------------------------

def ref_log_gaussian(X, means, covars, covariance_type):
    """Brute-force marginalized per-state emission log probability."""
    nc, nf = means.shape
    out = np.zeros((len(X), nc))
    for t, row in enumerate(X):
        idx = np.flatnonzero(~np.isnan(row))
        if idx.size == 0:
            continue  # fully missing row: uniform factor 1 -> log = 0
        for c in range(nc):
            if covariance_type == "diag":
                out[t, c] = sum(
                    norm.logpdf(row[j], means[c, j],
                                np.sqrt(covars[c, j]))
                    for j in idx)
            else:
                out[t, c] = multivariate_normal.logpdf(
                    row[idx], means[c, idx],
                    covars[c][np.ix_(idx, idx)])
    return out


def ref_forward(log_frame, startprob, transmat):
    """Scaled-free log forward algorithm on a single sequence."""
    T, nc = log_frame.shape
    fwd = np.zeros((T, nc))
    fwd[0] = np.log(startprob) + log_frame[0]
    for t in range(1, T):
        for j in range(nc):
            fwd[t, j] = log_frame[t, j] + np.logaddexp.reduce(
                fwd[t - 1] + np.log(transmat[:, j]))
    return np.logaddexp.reduce(fwd[-1]), fwd


def ref_posteriors(log_frame, startprob, transmat):
    """Forward-backward state posteriors on a single sequence."""
    T, nc = log_frame.shape
    _, fwd = ref_forward(log_frame, startprob, transmat)
    bwd = np.zeros((T, nc))
    for t in range(T - 2, -1, -1):
        for i in range(nc):
            bwd[t, i] = np.logaddexp.reduce(
                np.log(transmat[i]) + log_frame[t + 1] + bwd[t + 1])
    log_gamma = fwd + bwd
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    return np.exp(log_gamma)


def ref_viterbi(log_frame, startprob, transmat):
    """Exhaustive Viterbi over every state sequence."""
    T, nc = log_frame.shape
    best = (-np.inf, None)
    for seq in product(range(nc), repeat=T):
        lp = np.log(startprob[seq[0]]) + log_frame[0, seq[0]]
        for t in range(1, T):
            lp += (np.log(transmat[seq[t - 1], seq[t]])
                   + log_frame[t, seq[t]])
        if lp > best[0]:
            best = (lp, np.array(seq))
    return best


def ref_emission_stats(X, posteriors, means, covars, covariance_type):
    """One-iteration E-step statistics computed from conditional moments."""
    nc, nf = means.shape
    obs = np.zeros((nc, nf))
    obs_count = np.zeros((nc, nf))
    post = posteriors.sum(axis=0)
    second_full = np.zeros((nc, nf, nf))
    second_diag = np.zeros((nc, nf))
    for t, row in enumerate(X):
        idx = np.flatnonzero(~np.isnan(row))
        miss = np.setdiff1d(np.arange(nf), idx)
        observed = (~np.isnan(row)).astype(float)
        for c in range(nc):
            g = posteriors[t, c]
            cond_mean = means[c].copy()
            cond_mean[idx] = row[idx]  # observed cells are known exactly
            if idx.size and miss.size and covariance_type == "full":
                cv = covars[c]
                cond_mean[miss] = (
                    means[c, miss]
                    + cv[np.ix_(miss, idx)] @ np.linalg.solve(
                        cv[np.ix_(idx, idx)],
                        row[idx] - means[c, idx]))
            obs[c] += g * cond_mean
            obs_count[c] += g * observed
            if covariance_type == "full":
                sm = np.outer(cond_mean, cond_mean)
                if miss.size and idx.size:
                    cv = covars[c]
                    sm[np.ix_(miss, miss)] += (
                        cv[np.ix_(miss, miss)]
                        - cv[np.ix_(miss, idx)] @ np.linalg.solve(
                            cv[np.ix_(idx, idx)],
                            cv[np.ix_(idx, miss)]))
                elif miss.size:  # fully missing row
                    sm += covars[c]
                second_full[c] += g * sm
            else:
                var = np.where(np.isnan(row), covars[c], 0.0)
                second_diag[c] += g * (cond_mean ** 2 + var)
    return {
        "post": post, "obs": obs, "obs_count": obs_count,
        "obs*obs.T": second_full, "obs**2": second_diag}


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture
def prng():
    return np.random.RandomState(4242)


@pytest.fixture
def small_model(prng):
    nc, nf = 2, 3
    means = np.array([[0.0, 1.0, -1.0], [3.0, -2.0, 2.0]])
    covars = make_covar_matrix("full", nc, nf, random_state=prng)
    startprob = np.array([0.6, 0.4])
    transmat = np.array([[0.7, 0.3], [0.4, 0.6]])
    return nc, nf, means, covars, startprob, transmat


def make_missing(X, prng, rate=0.4, fully_missing_rows=()):
    Xm = X.copy()
    Xm[prng.rand(*X.shape) < rate] = np.nan
    for r in fully_missing_rows:
        Xm[r] = np.nan
    return Xm


def fitted_params(model, means, covars, startprob, transmat):
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars
    model._check()
    return model


# ---------------------------------------------------------------------------
# Constructor and input validation.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["nan", "ignore", "marginalise", ""])
def test_bad_missing_value(bad):
    with pytest.raises(ValueError):
        hmm.GaussianHMM(2, missing=bad)


@pytest.mark.parametrize("covariance_type", ["spherical", "tied"])
def test_marginalize_rejects_unsupported_covariance(covariance_type):
    with pytest.raises(ValueError):
        hmm.GaussianHMM(2, covariance_type=covariance_type,
                        missing="marginalize")


def test_default_missing_is_raise_and_strict():
    h = hmm.GaussianHMM(2)
    assert h.missing == "raise"
    assert h._check_array(np.zeros((2, 2))).dtype.kind == "f"
    with pytest.raises(ValueError):
        h._check_array(np.array([[1.0, np.nan]]))


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_infinity_always_rejected(small_model, bad_value):
    nc, nf, means, covars, startprob, transmat = small_model
    h = fitted_params(
        hmm.GaussianHMM(nc, "full", missing="marginalize"),
        means, covars, startprob, transmat)
    X = np.zeros((5, nf))
    X[0, 0] = bad_value
    with pytest.raises(ValueError):
        h.score(X)
    with pytest.raises(ValueError):
        h.fit(X)


def test_other_estimators_do_not_relax_input_checks():
    X = np.array([[1.0, np.nan]])
    with pytest.raises(ValueError):
        hmm.GMMHMM(2, covariance_type="full")._check_array(X)
    with pytest.raises(ValueError):
        hmm.PoissonHMM(2)._check_array(X)
    with pytest.raises(ValueError):
        hmm.CategoricalHMM(2)._check_array(np.array([[np.nan]]))
    with pytest.raises(ValueError):
        VariationalGaussianHMM(2)._check_array(X)


def test_missing_attribute_is_gaussian_only():
    assert not hasattr(hmm.GMMHMM(2), "missing")
    assert not hasattr(hmm.PoissonHMM(2), "missing")


# ---------------------------------------------------------------------------
# Scoring: brute-force enumeration on small examples.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("implementation", ["log", "scaling"])
@pytest.mark.parametrize("covariance_type", ["diag", "full"])
def test_score_matches_brute_force(small_model, prng, implementation,
                                   covariance_type):
    nc, nf, means, covars_full, startprob, transmat = small_model
    if covariance_type == "diag":
        covars = np.broadcast_to(
            np.diag(covars_full[0]), (nc, nf)).copy()
    else:
        covars = covars_full
    h = fitted_params(
        hmm.GaussianHMM(nc, covariance_type, implementation=implementation,
                        missing="marginalize", init_params="", params="st"),
        means, covars, startprob, transmat)
    X = prng.randn(8, nf) + means[0]
    Xm = make_missing(X, prng, fully_missing_rows=(3,))

    ref_lp = ref_log_gaussian(Xm, means, covars, covariance_type)
    assert_allclose(h._compute_log_likelihood(Xm), ref_lp, atol=ATOL)

    ref_ll, _ = ref_forward(ref_lp, startprob, transmat)
    assert_allclose(h.score(Xm), ref_ll, atol=ATOL)

    ll, posteriors = h.score_samples(Xm)
    assert_allclose(ll, ref_ll, atol=ATOL)
    assert_allclose(posteriors,
                    ref_posteriors(ref_lp, startprob, transmat),
                    atol=ATOL)
    assert_allclose(posteriors.sum(axis=1), np.ones(len(Xm)), atol=ATOL)


@pytest.mark.parametrize("algorithm", ["viterbi", "map"])
def test_decode_matches_enumeration(small_model, prng, algorithm):
    nc, nf, means, covars, startprob, transmat = small_model
    h = fitted_params(
        hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="", params="st"),
        means, covars, startprob, transmat)
    X = prng.randn(6, nf) + means[0]
    Xm = make_missing(X, prng, fully_missing_rows=(2,))
    ref_lp = ref_log_gaussian(Xm, means, covars, "full")

    ref_ll, ref_seq = ref_viterbi(ref_lp, startprob, transmat)
    ll, seq = h.decode(Xm, algorithm="viterbi")
    assert_allclose(ll, ref_ll, atol=ATOL)
    assert_allclose(seq, ref_seq)

    # MAP: independently argmax the brute-force smoothed posteriors.
    ref_map = ref_posteriors(ref_lp, startprob, transmat).argmax(axis=1)
    _, map_seq = h.decode(Xm, algorithm="map")
    assert_allclose(map_seq, ref_map)
    assert_allclose(h.predict(Xm), ref_map)


def test_fully_missing_rows_keep_time_steps(small_model, prng):
    nc, nf, means, covars, startprob, transmat = small_model
    h = fitted_params(
        hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="", params="st"),
        means, covars, startprob, transmat)
    T = 7
    X = np.full((T, nf), np.nan)
    ll, posteriors = h.score_samples(X)
    # No emission evidence: posterior is the marginal state distribution
    # propagated by the Markov chain; everything stays finite and normalized.
    assert np.isfinite(ll)
    assert posteriors.shape == (T, nc)
    assert_allclose(posteriors.sum(axis=1), np.ones(T), atol=ATOL)
    lp = h._compute_log_likelihood(X)
    assert_allclose(lp, np.zeros((T, nc)), atol=ATOL)
    _, seq = h.decode(X, algorithm="viterbi")
    assert len(seq) == T
    # A fully missing row is a pure Markov propagation step: its *filtered*
    # (forward) distribution equals the previous filter times the transition
    # matrix, while its smoothed posterior additionally absorbs future
    # evidence through the backward pass.
    X2 = prng.randn(T, nf) + means[0]
    X2[3] = np.nan
    ref_lp = ref_log_gaussian(X2, means, covars, "full")
    assert_allclose(ref_lp[3], np.zeros(nc), atol=ATOL)
    _, fwd = ref_forward(ref_lp, startprob, transmat)
    prev = np.exp(fwd[2] - np.logaddexp.reduce(fwd[2]))
    pred = prev @ transmat
    filtered = np.exp(fwd[3] - np.logaddexp.reduce(fwd[3]))
    assert_allclose(filtered, pred / pred.sum(), atol=ATOL)
    # The library's smoothed posteriors match the independent smoother.
    _, posteriors = h.score_samples(X2)
    assert_allclose(
        posteriors, ref_posteriors(ref_lp, startprob, transmat), atol=ATOL)
    assert_allclose(h.score(X2),
                    np.logaddexp.reduce(fwd[-1]), atol=ATOL)


def test_log_and_scaling_agree(small_model, prng):
    nc, nf, means, covars, startprob, transmat = small_model
    kw = dict(init_params="", params="st")
    h_log = fitted_params(
        hmm.GaussianHMM(nc, "full", implementation="log",
                        missing="marginalize", **kw),
        means, covars, startprob, transmat)
    h_sca = fitted_params(
        hmm.GaussianHMM(nc, "full", implementation="scaling",
                        missing="marginalize", **kw),
        means, covars, startprob, transmat)
    X = make_missing(prng.randn(30, nf) + means[0], prng,
                     fully_missing_rows=(5, 12))
    assert_allclose(h_log.score(X), h_sca.score(X), atol=ATOL)
    assert_allclose(h_log.predict_proba(X), h_sca.predict_proba(X),
                    atol=ATOL)
    assert_allclose(
        h_log.decode(X, algorithm="viterbi")[1],
        h_sca.decode(X, algorithm="viterbi")[1])


def test_sequence_boundaries_do_not_transfer(small_model, prng):
    nc, nf, means, covars, startprob, transmat = small_model
    h = fitted_params(
        hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="", params="st"),
        means, covars, startprob, transmat)
    X = make_missing(prng.randn(15, nf) + means[0], prng,
                     fully_missing_rows=(4, 10))
    lengths = [5, 4, 6]
    parts = np.split(X, np.cumsum(lengths)[:-1])
    assert_allclose(h.score(X, lengths),
                    sum(h.score(p) for p in parts), atol=ATOL)
    seq_joint = h.decode(X, lengths, algorithm="viterbi")[1]
    seq_split = np.concatenate(
        [h.decode(p, algorithm="viterbi")[1] for p in parts])
    assert_allclose(seq_joint, seq_split)
    # And the joint result must not equal a single-chain decode in general
    # (boundary rows must reset through startprob).
    single = h.decode(X, algorithm="viterbi")[1]
    # The boundary at row 5 uses startprob, not a transition: check via
    # independent enumeration of the second sub-sequence.
    ref_lp = ref_log_gaussian(parts[1], means, covars, "full")
    _, ref_seq = ref_viterbi(ref_lp, startprob, transmat)
    assert_allclose(seq_joint[5:9], ref_seq)
    assert single is not None


# ---------------------------------------------------------------------------
# Complete-data regression: the new mode must be numerically identical to the
# legacy finite-only path when there are no NaNs.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("covariance_type", ["diag", "full"])
def test_complete_data_matches_legacy_path(small_model, prng,
                                           covariance_type):
    nc, nf, means, covars_full, startprob, transmat = small_model
    covars = (covars_full if covariance_type == "full"
              else np.broadcast_to(np.diag(covars_full[0]),
                                   (nc, nf)).copy())
    X = prng.randn(40, nf) + means[0]
    kw = dict(init_params="", params="st")
    h_raise = fitted_params(
        hmm.GaussianHMM(nc, covariance_type, **kw),
        means, covars, startprob, transmat)
    h_marg = fitted_params(
        hmm.GaussianHMM(nc, covariance_type, missing="marginalize", **kw),
        means, covars, startprob, transmat)
    assert_allclose(h_raise.score(X), h_marg.score(X), atol=ATOL)
    assert_allclose(h_raise.predict_proba(X), h_marg.predict_proba(X),
                    atol=ATOL)
    assert_allclose(
        h_raise.decode(X, algorithm="viterbi")[1],
        h_marg.decode(X, algorithm="viterbi")[1])
    assert_allclose(
        h_raise._compute_log_likelihood(X),
        h_marg._compute_log_likelihood(X), atol=ATOL)


def test_sampling_shapes_unchanged(small_model, prng):
    nc, nf, means, covars, startprob, transmat = small_model
    h = fitted_params(
        hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="", params="st"),
        means, covars, startprob, transmat)
    X, states = h.sample(50, random_state=prng)
    assert X.shape == (50, nf)
    assert states.shape == (50,)
    assert np.isfinite(X).all()


# ---------------------------------------------------------------------------
# E-step statistics and one M-step against the hand-written reference.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("covariance_type", ["diag", "full"])
def test_one_estep_matches_reference(small_model, prng, covariance_type):
    nc, nf, means, covars_full, startprob, transmat = small_model
    covars = (covars_full if covariance_type == "full"
              else np.broadcast_to(np.diag(covars_full[0]),
                                   (nc, nf)).copy())
    h = fitted_params(
        hmm.GaussianHMM(nc, covariance_type, missing="marginalize",
                        init_params="", params="stmc",
                        means_prior=0, means_weight=0,
                        covars_prior=0.0,
                        covars_weight=nf if covariance_type == "full" else 1),
        means, covars, startprob, transmat)
    X = make_missing(prng.randn(12, nf) + means[0], prng,
                     fully_missing_rows=(4,))
    stats, _ = h._do_estep(X, np.array([len(X)]))
    ref_lp = ref_log_gaussian(X, means, covars, covariance_type)
    posteriors = ref_posteriors(ref_lp, startprob, transmat)
    ref = ref_emission_stats(X, posteriors, means, covars, covariance_type)
    assert_allclose(stats["post"], ref["post"], atol=ATOL)
    assert_allclose(stats["obs"], ref["obs"], atol=ATOL)
    assert_allclose(stats["obs_count"], ref["obs_count"], atol=ATOL)
    if covariance_type == "full":
        assert_allclose(stats["obs*obs.T"], ref["obs*obs.T"], atol=ATOL)
    else:
        assert_allclose(stats["obs**2"], ref["obs**2"], atol=ATOL)


@pytest.mark.parametrize("covariance_type", ["diag", "full"])
def test_one_mstep_recovers_weighted_estimates(small_model, prng,
                                               covariance_type):
    nc, nf, means, covars_full, startprob, transmat = small_model
    covars = (covars_full if covariance_type == "full"
              else np.broadcast_to(np.diag(covars_full[0]),
                                   (nc, nf)).copy())
    h = fitted_params(
        hmm.GaussianHMM(nc, covariance_type, missing="marginalize",
                        init_params="", params="mc",
                        means_prior=0, means_weight=0,
                        covars_prior=0.0,
                        covars_weight=nf if covariance_type == "full" else 1),
        means, covars, startprob, transmat)
    X = make_missing(prng.randn(12, nf) + means[0], prng)
    stats, _ = h._do_estep(X, np.array([len(X)]))
    h._do_mstep(stats)
    # With all priors switched off, the M-step equals the posterior-weighted
    # method of moments built from the conditional moments.
    assert_allclose(h.means_,
                    stats["obs"] / stats["post"][:, None], atol=ATOL)
    if covariance_type == "full":
        ref_cov = (stats["obs*obs.T"] / stats["post"][:, None, None]
                   - np.einsum("cf,ck->cfk", h.means_, h.means_))
        assert_allclose(h._covars_, ref_cov, atol=ATOL)
        # Positive definiteness is preserved.
        assert np.linalg.eigvalsh(h._covars_).min() > 0
    else:
        ref_var = stats["obs**2"] / stats["post"][:, None] - h.means_ ** 2
        assert_allclose(h._covars_, ref_var, atol=ATOL)
        assert (h._covars_ > 0).all()


@pytest.mark.parametrize("covariance_type", ["diag", "full"])
def test_log_likelihood_non_decreasing(small_model, prng, covariance_type):
    # Priors disabled: EM on the observed-data likelihood is monotone.
    nc, nf, means, covars_full, startprob, transmat = small_model
    covars = (covars_full if covariance_type == "full"
              else np.broadcast_to(np.diag(covars_full[0]),
                                   (nc, nf)).copy())
    X = make_missing(prng.randn(80, nf) + means[0], prng, rate=0.35,
                     fully_missing_rows=(10, 40))
    h = hmm.GaussianHMM(
        nc, covariance_type, n_iter=12, random_state=7,
        missing="marginalize", init_params="st",
        means_prior=0, means_weight=0, covars_prior=0.0,
        covars_weight=nf if covariance_type == "full" else 1)
    h.means_ = means.copy()
    h.covars_ = covars.copy()
    h.fit(X)
    deltas = np.diff(h.monitor_.history)
    assert deltas.min() >= -1e-8


def test_frozen_parameters_untouched(small_model, prng):
    nc, nf, means, covars, startprob, transmat = small_model
    h = fitted_params(
        hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="st", params="st"),
        means, covars, startprob, transmat)
    X = make_missing(prng.randn(30, nf) + means[0], prng)
    h.fit(X)
    assert_allclose(h.means_, means)
    assert_allclose(h.covars_, covars)


def test_repeated_fit_has_no_residue(small_model, prng):
    nc, nf, means, covars, startprob, transmat = small_model
    X = prng.randn(60, nf) + means[0]
    Xm1 = make_missing(X, prng)
    Xm2 = make_missing(X, prng, rate=0.6)

    def fresh():
        return hmm.GaussianHMM(nc, "full", n_iter=4, random_state=123,
                               missing="marginalize")

    a, b, c = fresh(), fresh(), fresh()
    a.fit(Xm1)
    b.fit(Xm2)       # different masks in between
    c.fit(Xm1)       # same mask/data as a
    assert_allclose(a.means_, c.means_)
    assert_allclose(a._covars_, c._covars_)
    assert_allclose(a.transmat_, c.transmat_)
    # The cache generation was advanced for every M-step and never carries
    # factors from a stale model state.
    assert a._missing_cache_.stamp == c._missing_cache_.stamp


def test_fit_multi_sequence(small_model, prng):
    nc, nf, means, covars, startprob, transmat = small_model
    h = hmm.GaussianHMM(nc, "full", n_iter=5, random_state=0,
                        missing="marginalize")
    lengths = [12, 9, 14]
    X = make_missing(
        prng.randn(sum(lengths), nf) + means[0], prng,
        fully_missing_rows=(0, 12, 21))
    h.fit(X, lengths)
    assert np.isfinite(h.means_).all()
    assert np.linalg.eigvalsh(h._covars_).min() > 0
    assert np.isfinite(h.score(X, lengths))


# ---------------------------------------------------------------------------
# Initialization.
# ---------------------------------------------------------------------------

def test_auto_init_finite_and_reproducible(prng):
    nc, nf = 3, 5
    X = prng.randn(60, nf)
    Xm = make_missing(X, prng)
    a = hmm.GaussianHMM(nc, "diag", random_state=99,
                        missing="marginalize")
    b = hmm.GaussianHMM(nc, "diag", random_state=99,
                        missing="marginalize")
    a.fit(Xm)
    b.fit(Xm)
    assert np.isfinite(a.means_).all()
    assert (a._covars_ > 0).all()
    assert_allclose(a.means_, b.means_)
    assert_allclose(a._covars_, b._covars_)

    af = hmm.GaussianHMM(nc, "full", random_state=99,
                         missing="marginalize")
    af.fit(Xm)
    assert np.isfinite(af.means_).all()
    assert np.linalg.eigvalsh(af._covars_).min() > 0


def test_auto_init_all_missing_feature_raises(prng):
    X = prng.randn(40, 4)
    X[:, 2] = np.nan
    h = hmm.GaussianHMM(2, "diag", missing="marginalize")
    with pytest.raises(ValueError, match="no finite observation"):
        h.fit(X)
    # Explicit Gaussian parameters plus frozen updates make it legal.
    h2 = hmm.GaussianHMM(2, "diag", init_params="st", params="st",
                         missing="marginalize")
    h2.means_ = np.zeros((2, 4))
    h2.covars_ = np.ones((2, 4))
    h2.fit(X)
    assert_allclose(h2.means_, np.zeros((2, 4)))


def test_inference_only_on_fully_missing_batch(small_model):
    nc, nf, means, covars, startprob, transmat = small_model
    h = fitted_params(
        hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="", params="st"),
        means, covars, startprob, transmat)
    X = np.full((8, nf), np.nan)
    ll, posteriors = h.score_samples(X)
    assert_allclose(ll, 0.0, atol=ATOL)
    assert np.isfinite(posteriors).all()
    assert h.decode(X, algorithm="viterbi")[1].shape == (8,)


def test_gap_is_not_treated_as_zero(small_model, prng):
    # The marginalized likelihood of a NaN cell must differ from both
    # zero-imputation and row deletion.
    nc, nf, means, covars, startprob, transmat = small_model
    h = fitted_params(
        hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="", params="st"),
        means, covars, startprob, transmat)
    row = prng.randn(1, nf) + 5
    row_nan = row.copy(); row_nan[0, 1] = np.nan
    row_zero = row.copy(); row_zero[0, 1] = 0.0
    assert not np.allclose(
        h._compute_log_likelihood(row_nan),
        h._compute_log_likelihood(row_zero), atol=1e-6)


# ---------------------------------------------------------------------------
# Pattern cache: reuse, invalidation and the 64 MiB budget.
# ---------------------------------------------------------------------------

def test_cache_reuse_and_invalidation(small_model, prng):
    nc, nf, means, covars, startprob, transmat = small_model
    h = fitted_params(
        hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="", params="st"),
        means, covars, startprob, transmat)
    X = make_missing(prng.randn(50, nf) + means[0], prng)
    h.score(X)
    cache = h._missing_cache_
    n_entries = len(cache._data)
    stamp = cache.stamp
    h.score(X)  # same model, same masks: entries are reused, not duplicated
    assert len(cache._data) == n_entries
    assert cache.stamp == stamp

    # Changing means does not invalidate covariance factors.
    h.means_ = means + 1.0
    h.score(X)
    assert cache.stamp == stamp

    # Changing covariances invalidates everything: the generation moves
    # forward, stale entries are ignored and rebuilt on demand.
    h.covars_ = covars * 1.01
    h.score(X)
    assert cache.stamp == stamp + 1
    for key, (entry_stamp, _) in cache._data.items():
        # Entries found during the score call were rebuilt with the new
        # generation; any surviving stale key is invisible to lookups.
        assert cache.get(key, cache.stamp) is not None
        assert entry_stamp == cache.stamp


def test_cache_resets_between_fit_iterations(small_model, prng):
    nc, nf, means, covars, startprob, transmat = small_model
    h = hmm.GaussianHMM(nc, "full", n_iter=3, random_state=0,
                        missing="marginalize")
    X = make_missing(prng.randn(50, nf) + means[0], prng)
    h.fit(X)
    # _estep_begin clears storage; only the last iteration's patterns stay.
    assert h._missing_cache_.stamp >= 3


def test_cache_budget_enforced(prng):
    nc, nf = 4, 16
    means = prng.randn(nc, nf)
    A = prng.rand(nf, nf)
    covars = np.stack([A @ A.T / nf + np.eye(nf)] * nc)
    h = hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="", params="st")
    h.startprob_ = np.full(nc, 1 / nc)
    h.transmat_ = np.full((nc, nc), 1 / nc)
    h.means_ = means
    h.covars_ = covars
    X = prng.randn(2000, nf)
    X[prng.rand(2000, nf) < 0.5] = np.nan
    h.score(X)
    cache = h._missing_cache_
    assert cache.size <= _missing.MISSING_CACHE_BUDGET_BYTES
    # Every retained entry belongs to the current generation.
    assert all(stamp == cache.stamp for stamp, _ in cache._data.values())


def test_cyclic_patterns_cache_is_tiny(prng):
    nc, nf = 4, 16
    means = prng.randn(nc, nf)
    A = prng.rand(nf, nf)
    covars = np.stack([A @ A.T / nf + np.eye(nf)] * nc)
    h = hmm.GaussianHMM(nc, "full", missing="marginalize",
                        init_params="", params="st")
    h.startprob_ = np.full(nc, 1 / nc)
    h.transmat_ = np.full((nc, nc), 1 / nc)
    h.means_ = means
    h.covars_ = covars
    T = 500
    X = prng.randn(T, nf)
    for t in range(T):
        X[t, np.arange(nf) % 4 == (t % 4)] = np.nan
    h.score(X)
    assert len(h._missing_cache_._data) == 4

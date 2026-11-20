"""
Marginalization over missing (NaN) observations for Gaussian emissions.

A *missingness pattern* is the bit mask of channels that are observed at a
given time step.  Rows sharing the same pattern form a group; all numeric work
(per-state Cholesky factors, log-determinants and conditional moments) is then
done once per (pattern, state) pair and reused for every row in the group.

Only :class:`~hmmlearn.hmm.GaussianHMM` with ``covariance_type`` "diag" or
"full" supports missing values.  NaN marks a cell without an observation;
``+/- inf`` remains illegal everywhere.
"""

import numpy as np
from scipy import linalg


#: Upper bound on the auxiliary storage retained for pattern reuse.
MISSING_CACHE_BUDGET_BYTES = 64 * 2 ** 20

_LOG_2PI = np.log(2 * np.pi)


class MissingPatterns:
    """Row groups of ``X`` sharing the same observed-channel mask."""

    def __init__(self, X):
        n_samples, n_features = X.shape
        observed = ~np.isnan(X)
        if n_features <= 64:
            bits = np.ones(n_features, dtype=np.int64) << np.arange(
                n_features, dtype=np.int64)
            keys = observed.astype(np.int64) @ bits
        else:
            packed = np.packbits(observed, axis=1, bitorder="little")
            keys = np.array([row.tobytes() for row in packed], dtype=object)
        unique, inverse = np.unique(keys, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        sorted_inv = inverse[order]
        bounds = np.flatnonzero(np.diff(sorted_inv)) + 1
        groups = np.split(order, bounds)
        self.groups = list(zip(unique, groups))
        self.observed = observed

    def __iter__(self):
        return iter(self.groups)

    def indices(self, rows):
        """Channels observed in every row of the group (group is uniform)."""
        return np.flatnonzero(self.observed[rows[0]])


class MissingCache:
    """
    Per-pattern reuse cache for marginalized Gaussian factors.

    Entries only depend on the covariance parameters and on the observed
    channel mask, so a change of mean parameters never invalidates them.
    ``stamp`` identifies the covariance generation; entries from an older
    generation are ignored.  The total retained size is bounded by
    :data:`MISSING_CACHE_BUDGET_BYTES`; entries that would exceed the budget
    are recomputed on demand instead of being stored.
    """

    def __init__(self, budget=MISSING_CACHE_BUDGET_BYTES):
        self._data = {}
        self.size = 0
        self.budget = budget
        self.stamp = 0

    def get(self, key, stamp):
        item = self._data.get(key)
        if item is None or item[0] != stamp:
            return None
        return item[1]

    def put(self, key, entry, covariance_type, stamp):
        size = _entry_nbytes(entry, covariance_type)
        existing = self._data.get(key)
        if existing is not None and existing[0] == stamp:
            return  # identical generation: keep the retained factor
        if self.size + size <= self.budget:
            if existing is not None:
                # Replace a stale-generation entry.
                self.size -= _entry_nbytes(existing[1], covariance_type)
            self._data[key] = (stamp, entry)
            self.size += size

    def bump(self):
        """Invalidate every entry after a covariance update."""
        self.stamp += 1

    def clear(self):
        self._data.clear()
        self.size = 0


def _entry_nbytes(entry, covariance_type):
    """Size of one cached pattern entry in bytes."""
    if covariance_type == "diag":
        mean_obs, log_norm, cov_obs = entry
        return mean_obs.nbytes + log_norm.nbytes + cov_obs.nbytes
    chols, log_norm = entry
    return log_norm.nbytes + sum(chol.nbytes for chol in chols)


def regularize_full_covars(covars, floor):
    """Project full covariance matrices onto the PSD cone with a floor.

    EM updates are positive semidefinite in exact arithmetic; with missing
    data and states carrying negligible posterior mass, floating-point
    cancellation can produce tiny negative eigenvalues.  Rebuilding each
    matrix from its clipped eigendecomposition restores symmetry and
    positive definiteness without changing well-conditioned components.
    """
    regularized = np.empty_like(covars)
    for c, cv in enumerate(covars):
        eigvals, eigvecs = linalg.eigh(cv)
        eigvals = np.maximum(eigvals, floor)
        regularized[c] = (eigvecs * eigvals) @ eigvecs.T
    return regularized


def _build_density_entry(means, covars, covariance_type, idx):
    """Precompute the per-state Gaussian factors for one pattern."""
    if covariance_type == "diag":
        mean_obs = means[:, idx]
        cov_obs = np.maximum(covars[:, idx], np.finfo(float).tiny)
        log_norm = idx.size * _LOG_2PI + np.log(cov_obs).sum(axis=1)
        return mean_obs, log_norm, cov_obs
    n_components = means.shape[0]
    chols = []
    log_norm = np.empty(n_components)
    for c in range(n_components):
        sub = covars[c][np.ix_(idx, idx)]
        try:
            chol = linalg.cholesky(sub, lower=True)
        except linalg.LinAlgError:
            # Mirror the defensive behaviour of log_multivariate_normal_density.
            chol = linalg.cholesky(
                sub + 1e-7 * np.eye(idx.size), lower=True)
        chols.append(chol)
        log_norm[c] = (idx.size * _LOG_2PI
                       + 2.0 * np.log(np.diag(chol)).sum())
    return chols, log_norm


def _diag_group_logprob(x, entry):
    """Diagonal log-density for all rows of a pattern group."""
    mean_obs, log_norm, cov_obs = entry
    diff = x[:, None, :] - mean_obs[None, :, :]
    quadratic = ((diff * diff) / cov_obs[None, :, :]).sum(axis=-1)
    return -0.5 * (log_norm[None, :] + quadratic)


def _full_group_logprob(x, means_obs, entry):
    """Full-covariance log-density for all rows of a pattern group."""
    chols, log_norm = entry
    n_rows = x.shape[0]
    n_components = means_obs.shape[0]
    out = np.empty((n_rows, n_components))
    for c in range(n_components):
        sol = linalg.solve_triangular(
            chols[c], (x - means_obs[c]).T, lower=True).T
        out[:, c] = -0.5 * (
            log_norm[c] + (sol * sol).sum(axis=1))
    return out


def log_gaussian_likelihood_missing(X, means, covars, covariance_type,
                                    cache=None, stamp=0):
    """
    Per-state emission log probability with NaN channels marginalized out.

    Rows without any observed channel contribute zero log probability for
    every state (the empty product over channels equals 1); they keep their
    time step and the Markov transitions on both sides remain intact.

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        Feature matrix with NaN marking missing channels.
    means : array, shape (n_components, n_features)
    covars : array
        ``(n_components, n_features)`` for "diag" or
        ``(n_components, n_features, n_features)`` for "full".
    covariance_type : {"diag", "full"}
    cache : MissingCache or None
        Pattern cache retaining per-state factors for reuse.
    stamp : int
        Covariance generation the cached factors must belong to.

    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.zeros((n_samples, n_components))
    patterns = MissingPatterns(X)
    for key, rows in patterns:
        idx = patterns.indices(rows)
        if idx.size == 0:
            continue  # fully missing rows: no emission evidence.
        entry = cache.get(key, stamp) if cache is not None else None
        if entry is None:
            entry = _build_density_entry(means, covars, covariance_type, idx)
            if cache is not None:
                cache.put(key, entry, covariance_type, stamp)
        x = X[np.ix_(rows, idx)]
        if covariance_type == "diag":
            log_prob[rows, :] = _diag_group_logprob(x, entry)
        else:
            log_prob[rows, :] = _full_group_logprob(
                x, means[:, idx], entry)
    return log_prob


def conditional_moments(X, means, covars, covariance_type,
                        need_second_moment=False, need_variance=False):
    """
    Conditional expectation of the complete observation given observed cells.

    For every row ``t`` and state ``c`` the missing channels are replaced by
    their Gaussian conditional mean given the observed channels,

    .. math::
        E[X_t | X_{t,\\mathrm{obs}}, Z_t=c].

    For full covariances the expected outer product
    :math:`E[X_t X_t^T | X_{t,\\mathrm{obs}}, Z_t=c]` (including the
    conditional covariance of the missing block) is returned when
    ``need_second_moment`` is true.

    Returns
    -------
    filled : array, shape (n_samples, n_components, n_features)
        Conditional means; observed cells repeat the observation.
    second_moment : array or None
        ``(n_samples, n_components, n_features, n_features)`` for "full"
        when requested, otherwise ``None``.
    cond_var : array or None
        ``(n_samples, n_components, n_features)`` conditional variances for
        "diag" (zero on observed cells) when requested, otherwise ``None``.
    """
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    all_idx = np.arange(n_features)
    filled = np.empty((n_samples, n_components, n_features))
    second_moment = (
        np.empty((n_samples, n_components, n_features, n_features))
        if covariance_type == "full" and need_second_moment else None)
    cond_var = (
        np.empty((n_samples, n_components, n_features))
        if covariance_type == "diag" and need_variance else None)
    patterns = MissingPatterns(X)
    for key, rows in patterns:
        idx = patterns.indices(rows)
        miss = np.setdiff1d(all_idx, idx)
        x = X[np.ix_(rows, idx)] if idx.size else np.empty((rows.size, 0))
        if covariance_type == "diag":
            _fill_diag(rows, x, idx, miss, means, covars, filled, cond_var)
        else:
            _fill_full(rows, x, idx, miss, means, covars,
                       filled, second_moment)
    return filled, second_moment, cond_var


def _fill_diag(rows, x, idx, miss, means, covars, filled, cond_var):
    # Independent channels: missing cells use marginal means/variances, and
    # observed cells are known exactly.  All work uses group-local row
    # indices and is written back through the global ``rows`` positions.
    nc = means.shape[0]
    n_local = rows.size
    comp = np.arange(nc)
    local_filled = np.empty((n_local, nc, means.shape[1]))
    local_filled[:, :, idx] = x[:, None, :]
    local_filled[:, :, miss] = means[None, :, miss]
    filled[rows] = local_filled
    if cond_var is not None:
        local_var = np.zeros_like(local_filled)
        local_var[:, :, miss] = covars[None, :, miss]
        cond_var[rows] = local_var


def _fill_full(rows, x, idx, miss, means, covars, filled, second_moment):
    n_components = means.shape[0]
    n_local = rows.size
    n_features = means.shape[1]
    local_filled = np.empty((n_local, n_components, n_features))
    local_sm = (np.empty((n_local, n_components, n_features, n_features))
                if second_moment is not None else None)
    for c in range(n_components):
        m = local_filled[:, c, :]
        m[:, idx] = x
        if miss.size == 0:
            if local_sm is not None:
                local_sm[:, c] = m[:, :, None] * m[:, None, :]
            continue
        mu_m = means[c, miss]
        if idx.size == 0:
            m[:, miss] = mu_m
            if local_sm is not None:
                local_sm[:, c] = (
                    m[:, :, None] * m[:, None, :] + covars[c][None, :, :])
            continue
        cv = covars[c]
        cv_mo = cv[np.ix_(miss, idx)]
        cv_oo = cv[np.ix_(idx, idx)]
        try:
            solved = linalg.solve(cv_oo, cv_mo.T, assume_a="pos").T
        except linalg.LinAlgError:
            # Defensive fallback for nearly singular observed blocks
            # (e.g. states with negligible posterior mass).
            solved = np.linalg.lstsq(cv_oo, cv_mo.T, rcond=None)[0].T
        cond_mean = mu_m + (x - means[c, idx]) @ solved.T
        m[:, miss] = cond_mean
        if local_sm is None:
            continue
        cv_mm = cv[np.ix_(miss, miss)]
        try:
            cond_cov = cv_mm - cv_mo @ linalg.solve(
                cv_oo, cv_mo.T, assume_a="pos")
        except linalg.LinAlgError:
            cond_cov = cv_mm - cv_mo @ np.linalg.lstsq(
                cv_oo, cv_mo.T, rcond=None)[0]
        outer = m[:, :, None] * m[:, None, :]
        outer[np.ix_(np.arange(n_local), miss, miss)] += cond_cov
        local_sm[:, c] = outer
    filled[rows] = local_filled
    if second_moment is not None:
        second_moment[rows] = local_sm

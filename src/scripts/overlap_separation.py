"""
overlap_separation.py
=====================
Separate overlapping signals in digitized 1-D or 2-D data and decide, pixel by
pixel, which signal each bit of intensity belongs to.

The frame is modelled as a sum of K compact sources plus a flat background.
Each source is a (Gaussian) kernel whose position, size, orientation and total
intensity are fitted with an intensity-weighted Expectation-Maximisation loop.
The E-step "responsibilities" are exactly the requested quantity: for every
pixel, the fraction of its signal belonging to each source -- so overlap
regions are split instead of being lost or double counted.

Pipeline
--------
1. Robust background + noise estimate (median / MAD)
2. Seed detection: smoothed local maxima above threshold  -> number of signals
3. EM refinement of the K-component mixture, weighted by pixel intensity
   (spurious seeds are pruned automatically)
4. Outputs:
     fractions     (K, *shape)  soft split of every pixel, sums to 1
     components    (K, *shape)  background-subtracted intensity per signal
     labels        (*shape,)    hard assignment, -1 = background
     overlap_mask  (*shape,)    pixels genuinely shared by >= 2 signals

Works unchanged on 1-D traces (samples are then the "pixels").
Only requires numpy + scipy.

Typical use
-----------
>>> from overlap_separation import separate_signals
>>> res = separate_signals(frame)            # frame: 2-D ndarray
>>> res.n_signals, res.labels, res.overlap_mask
>>> per_signal_images = res.components       # overlap already divided
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


# --------------------------------------------------------------------------- #
#  Result container
# --------------------------------------------------------------------------- #
@dataclass
class SeparationResult:
    n_signals: int
    labels: np.ndarray          # int map, -1 = background
    fractions: np.ndarray       # (K, *shape) pixel ownership in [0, 1]
    components: np.ndarray      # (K, *shape) intensity owned by each signal
    overlap_mask: np.ndarray    # bool, pixels shared by >= 2 signals
    signal_mask: np.ndarray     # bool, pixels above detection threshold
    means: np.ndarray           # (K, ndim) fitted centroids
    covs: np.ndarray            # (K, ndim, ndim) fitted shapes
    amplitudes: np.ndarray      # (K,) total background-subtracted intensity
    background: float
    noise_sigma: float
    converged: bool
    n_iter: int


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _robust_stats(x: np.ndarray) -> tuple[float, float]:
    """Median and MAD-based sigma; robust against the signals themselves."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) * 1.4826)
    if mad == 0.0:
        mad = float(x.std()) or 1.0
    return med, mad


def _find_seeds(sm: np.ndarray, threshold: float, min_distance: int) -> np.ndarray:
    """Local maxima of the smoothed frame above `threshold`."""
    size = 2 * int(min_distance) + 1
    footprint = np.ones((size,) * sm.ndim, dtype=bool)
    is_peak = (maximum_filter(sm, footprint=footprint) == sm) & (sm > threshold)
    coords = np.argwhere(is_peak)
    if len(coords) < 2:
        return coords
    # de-duplicate plateau ties: greedy, brightest first
    vals = sm[tuple(coords.T)]
    kept: list[int] = []
    for i in np.argsort(vals)[::-1]:
        if all(np.linalg.norm(coords[i] - coords[j]) > min_distance for j in kept):
            kept.append(i)
    return coords[np.sort(kept)]


def _log_gauss(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Log of a multivariate normal pdf, evaluated at rows of X."""
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (X - mu).T)
    maha = (sol ** 2).sum(axis=0)
    logdet = 2.0 * np.log(np.diag(L)).sum()
    return -0.5 * (X.shape[1] * np.log(2.0 * np.pi) + logdet + maha)


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #
def separate_signals(
    data: np.ndarray,
    smooth_sigma: float = 1.0,      # px, pre-smoothing for detection only
    seed_nsigma: float = 4.0,       # peak threshold (in noise sigmas)
    mask_nsigma: float = 2.0,       # signal-mask threshold
    min_distance: int = 3,          # px, minimum peak separation
    init_width: float = 2.0,        # px, initial component width
    cov_reg: float = 0.25,          # px^2, covariance regularisation
    overlap_threshold: float = 0.2, # a pixel is "shared" if >=2 signals own >=20%
    min_flux_frac: float = 2e-3,    # prune components owning < this flux fraction
    max_iter: int = 300,
    tol: float = 1e-5,
) -> SeparationResult:
    """Detect, fit and separate overlapping signals in a 1-D or 2-D array."""
    img = np.asarray(data, dtype=float)
    was_1d = img.ndim == 1
    if was_1d:
        img = img[:, None]
    if img.ndim != 2:
        raise ValueError("separate_signals expects a 1-D or 2-D array")

    shape, d = img.shape, 2

    # -- 1. background / noise ------------------------------------------------
    sm = gaussian_filter(img, smooth_sigma)
    bg, noise = _robust_stats(img)
    bg_s, noise_s = _robust_stats(sm)

    signal_mask = sm > bg_s + mask_nsigma * noise_s

    # -- 2. seeds -------------------------------------------------------------
    seeds = _find_seeds(sm, bg_s + seed_nsigma * noise_s, min_distance)
    K = len(seeds)

    X = np.argwhere(signal_mask).astype(float)      # (n, 2) pixel coordinates
    w = np.clip(img[signal_mask] - bg, 0.0, None)   # intensity weights
    Wtot = float(w.sum())
    if Wtot <= 0.0:
        K = 0

    # -- 3. intensity-weighted EM --------------------------------------------
    converged, it = True, 0
    if K == 0:
        r = np.zeros((len(X), 0))
        Nk = np.zeros(0)
        mu = np.zeros((0, d))
        cov = np.zeros((0, d, d))
    else:
        mu = seeds.astype(float)
        cov = np.repeat(np.eye(d)[None] * init_width ** 2, K, axis=0)
        pi = np.full(K, 1.0 / K)
        prev_ll, converged = -np.inf, False

        def e_step():
            logp = np.empty((len(X), K))
            for k in range(K):
                logp[:, k] = np.log(pi[k]) + _log_gauss(X, mu[k], cov[k])
            mx = logp.max(axis=1, keepdims=True)
            p = np.exp(logp - mx)
            tot = p.sum(axis=1, keepdims=True)
            ll = float((w * (np.log(tot[:, 0]) + mx[:, 0])).sum() / Wtot)
            return p / tot, ll

        for it in range(1, max_iter + 1):
            r, ll = e_step()
            Nk = (w[:, None] * r).sum(axis=0)

            # prune components that grabbed essentially no flux (false seeds)
            keep = Nk > min_flux_frac * Wtot
            if not keep.all():
                mu, cov, Nk, r = mu[keep], cov[keep], Nk[keep], r[:, keep]
                K = int(keep.sum())
                if K == 0:
                    break
                r /= r.sum(axis=1, keepdims=True)
                pi = Nk / Nk.sum()
                prev_ll = -np.inf
                continue

            pi = Nk / Wtot
            for k in range(K):
                wr = w * r[:, k]
                mu[k] = wr @ X / Nk[k]
                diff = X - mu[k]
                cov[k] = (np.einsum("n,ni,nj->ij", wr, diff, diff) / Nk[k]
                          + cov_reg * np.eye(d))

            if abs(ll - prev_ll) < tol * (abs(prev_ll) + 1.0):
                converged = True
                break
            prev_ll = ll

        if K:
            r, _ = e_step()                      # responsibilities @ final fit
            Nk = (w[:, None] * r).sum(axis=0)
            order = np.argsort(Nk)[::-1]         # brightest signal first
            mu, cov, Nk, r = mu[order], cov[order], Nk[order], r[:, order]

    # -- 4. package per-pixel outputs ----------------------------------------
    fractions = np.zeros((K,) + shape)
    for k in range(K):
        fractions[k][signal_mask] = r[:, k]

    net = np.where(signal_mask, np.clip(img - bg, 0.0, None), 0.0)
    components = fractions * net

    labels = np.full(shape, -1, dtype=int)
    overlap = np.zeros(shape, dtype=bool)
    if K:
        labels[signal_mask] = r.argmax(axis=1)
        overlap[signal_mask] = (r >= overlap_threshold).sum(axis=1) >= 2

    if was_1d:
        labels, overlap = labels[:, 0], overlap[:, 0]
        signal_mask = signal_mask[:, 0]
        fractions, components = fractions[:, :, 0], components[:, :, 0]
        mu, cov = mu[:, :1], cov[:, :1, :1]

    return SeparationResult(
        n_signals=K, labels=labels, fractions=fractions, components=components,
        overlap_mask=overlap, signal_mask=signal_mask, means=mu, covs=cov,
        amplitudes=Nk, background=bg, noise_sigma=noise,
        converged=converged, n_iter=it,
    )

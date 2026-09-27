"""Entity-level decisions under macro F0.5.

Given calibrated match probabilities for candidate pairs, choose for every S1 entity the subset of
candidates that maximises its *expected* F0.5, including the option of predicting nothing (worth
P(no true match)). Pool exclusivity (a pool record belongs to at most one S1) is enforced first by
giving each pool record to its highest-probability claimant.

Expected F0.5 of a prefix (top-k by probability) is estimated by Monte Carlo over the candidates'
Bernoulli outcomes plus a Poisson number of true matches that blocking never retrieved
(`lam_miss`, measured on validation). This reproduces the exact break-even conditions of the metric
(e.g. a candidate must exceed ~0.73 when one other match is certain, ~0.77 with three, 0.5 when it
is the entity's only possible match) instead of a single global threshold.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def exclusivity_mask(i1: np.ndarray, i2: np.ndarray, p: np.ndarray, min_p: float = 0.0) -> np.ndarray:
    """True where the pair's S1 is the best claimant of the pool record (ties -> smallest i1)."""
    order = np.lexsort((i1, -p, i2))
    i2_sorted = i2[order]
    first = np.ones(len(i2), dtype=bool)
    first[1:] = i2_sorted[1:] != i2_sorted[:-1]
    mask = np.zeros(len(i2), dtype=bool)
    mask[order[first]] = True
    if min_p > 0:
        mask &= p >= min_p
    return mask


def _group_matrix(i1: np.ndarray, p: np.ndarray, max_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per S1 group: probabilities sorted desc, padded with 0 to max_k; returns (ids, P[n_groups, max_k], pair index matrix)."""
    order = np.lexsort((-p, i1))
    g = i1[order]
    start = np.ones(len(g), dtype=bool); start[1:] = g[1:] != g[:-1]
    grp_start = np.flatnonzero(start)
    grp_id = np.cumsum(start) - 1
    pos = np.arange(len(g)) - grp_start[grp_id]
    n_groups = len(grp_start)
    P = np.zeros((n_groups, max_k), dtype=np.float32)
    IDX = np.full((n_groups, max_k), -1, dtype=np.int64)
    keep = pos < max_k
    P[grp_id[keep], pos[keep]] = p[order][keep]
    IDX[grp_id[keep], pos[keep]] = order[keep]
    return g[grp_start], P, IDX


def expected_f05_select(i1: np.ndarray, p: np.ndarray, lam_miss: float = 0.02, n_samples: int = 256,
                        max_k: int = 16, seed: int = 0, chunk: int = 50_000, p_empty_scale: float = 1.0
                        ) -> np.ndarray:
    """Boolean mask over pairs: the expected-F0.5-optimal prefix per S1 (possibly empty)."""
    ids, P, IDX = _group_matrix(i1, p, max_k)
    n = P.shape[0]
    mask = np.zeros(len(p), dtype=bool)
    rng = np.random.default_rng(seed)
    ks = np.arange(1, max_k + 1, dtype=np.float32)
    for s in range(0, n, chunk):
        Pc = P[s:s + chunk]                                   # (g, K)
        g = Pc.shape[0]
        U = rng.random((n_samples, g, max_k), dtype=np.float32)
        Y = (U < Pc[None, :, :]).astype(np.float32)          # sampled truth of candidates
        M = rng.poisson(lam_miss, size=(n_samples, g)).astype(np.float32)  # unretrieved true matches
        T = np.cumsum(Y, axis=2)                              # (S, g, K): true positives in top-k
        N = T[:, :, -1] + M                                   # total true matches
        F = 1.25 * T / (ks[None, None, :] + 0.25 * N[:, :, None])
        EF = F.mean(axis=0)                                   # (g, K)
        # k=0: utility 1 iff N == 0
        E0 = (N == 0).mean(axis=0) * p_empty_scale
        # candidates beyond the group's size have p=0 -> never chosen (EF non-increasing there)
        valid = Pc > 0
        EF = np.where(valid, EF, -1.0)
        best_k = EF.argmax(axis=1) + 1
        best_v = EF.max(axis=1)
        choose = best_v > E0
        for gi in np.flatnonzero(choose):
            k = best_k[gi]
            idx = IDX[s + gi, :k]
            mask[idx[idx >= 0]] = True
    return mask


def threshold_select(p: np.ndarray, thr: np.ndarray | float) -> np.ndarray:
    return p >= thr


def decide(i1: np.ndarray, i2: np.ndarray, p: np.ndarray, method: str = "ef", thr: float = 0.7,
           lam_miss: float = 0.02, min_p: float = 0.05, n_samples: int = 256, seed: int = 0,
           p_empty_scale: float = 1.0) -> np.ndarray:
    """Final boolean mask over candidate pairs."""
    excl = exclusivity_mask(i1, i2, p, min_p=min_p)
    p_eff = np.where(excl, p, 0.0).astype(np.float32)
    if method == "thr":
        return excl & (p_eff >= thr)
    sel = expected_f05_select(i1, p_eff, lam_miss=lam_miss, n_samples=n_samples, seed=seed, p_empty_scale=p_empty_scale)
    return sel & excl


def write_lists(s1_eids: np.ndarray, pool_eids: np.ndarray, i1: np.ndarray, i2: np.ndarray, path, header: str) -> None:
    """Write a two-column TSV with one row per S1 (in s1_eids order), comma-joined pool ids."""
    order = np.argsort(i1, kind="stable")
    i1s, i2s = i1[order], i2[order]
    lists: Dict[int, list] = {}
    if len(i1s):
        bounds = np.flatnonzero(np.diff(i1s)) + 1
        starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(i1s)]])
        for a, b in zip(starts, ends):
            lists[int(i1s[a])] = pool_eids[i2s[a:b]].tolist()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header + "\n")
        for k, e in enumerate(s1_eids.tolist()):
            f.write(f"{e}\t{','.join(lists.get(k, []))}\n")

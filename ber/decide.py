"""Entity-level decisions under macro F0.5.

Given calibrated match probabilities for candidate pairs, choose for every S1 entity the subset of
candidates that maximises its *expected* F0.5, including the option of predicting nothing (worth
P(no true match)).

Two probability roles are kept apart:
  * truth probability  p       - every retrieved candidate contributes to the distribution of the
                                 entity's true-match count N (even if it cannot be predicted);
  * eligibility        p_eff   - only candidates the entity is allowed to predict (pool exclusivity:
                                 a pool record is offered to its best claimant first; if that claimant
                                 rejects it, it is offered to the next claimant, and so on).

Expected F0.5 of a prefix (top-k eligible by probability) is estimated by Monte Carlo over the
candidates' Bernoulli outcomes plus a Poisson number of true matches that blocking never retrieved
(`lam_miss`). This reproduces the exact break-even conditions of the metric (a candidate must exceed
~0.73 when one other match is certain, ~0.77 with three, 0.5 when it is the entity's only possible
match) instead of a single global threshold.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _group_matrix(i1: np.ndarray, p: np.ndarray, p_eff: np.ndarray, max_k: int):
    """Per S1 group: candidates ordered by (eligible first, p_eff desc, p desc), padded to max_k.
    Returns (P_truth[g,K], P_elig[g,K], IDX[g,K] pair indices or -1, n_groups)."""
    order = np.lexsort((-p, -p_eff, i1))
    g = i1[order]
    start = np.ones(len(g), dtype=bool); start[1:] = g[1:] != g[:-1]
    grp_start = np.flatnonzero(start)
    grp_id = np.cumsum(start) - 1
    pos = np.arange(len(g)) - grp_start[grp_id]
    n_groups = len(grp_start)
    P = np.zeros((n_groups, max_k), dtype=np.float32)
    E = np.zeros((n_groups, max_k), dtype=np.float32)
    IDX = np.full((n_groups, max_k), -1, dtype=np.int64)
    keep = pos < max_k
    P[grp_id[keep], pos[keep]] = p[order][keep]
    E[grp_id[keep], pos[keep]] = p_eff[order][keep]
    IDX[grp_id[keep], pos[keep]] = order[keep]
    # truth mass of candidates beyond max_k still counts towards N (as an expected count)
    tail = np.zeros(n_groups, dtype=np.float32)
    if (~keep).any():
        np.add.at(tail, grp_id[~keep], p[order][~keep])
    return P, E, IDX, tail


def expected_f05_select(i1: np.ndarray, p: np.ndarray, p_eff: np.ndarray, lam_miss: float = 0.02,
                        n_samples: int = 128, max_k: int = 16, seed: int = 0, chunk: int = 8_000) -> np.ndarray:
    """Boolean mask over pairs: the expected-F0.5-optimal eligible prefix per S1 (possibly empty)."""
    P, E, IDX, tail = _group_matrix(i1, p, p_eff, max_k)
    n = P.shape[0]
    mask = np.zeros(len(p), dtype=bool)
    rng = np.random.default_rng(seed)
    ks = np.arange(1, max_k + 1, dtype=np.float32)
    for s in range(0, n, chunk):
        Pc = P[s:s + chunk]; Ec = E[s:s + chunk]; g = Pc.shape[0]
        U = rng.random((n_samples, g, max_k), dtype=np.float32)
        Y = (U < Pc[None, :, :]).astype(np.float32)                    # sampled truth of all candidates
        M = rng.poisson(lam_miss + tail[s:s + chunk][None, :], size=(n_samples, g)).astype(np.float32)
        N = Y.sum(axis=2) + M                                         # total true matches
        elig = (Ec > 0)[None, :, :]
        T = np.cumsum(Y * elig, axis=2)                               # true positives among the first k eligible
        F = 1.25 * T / (ks[None, None, :] + 0.25 * N[:, :, None])
        EF = F.mean(axis=0)
        E0 = (N == 0).mean(axis=0)
        n_elig = (Ec > 0).sum(axis=1)
        valid = np.arange(max_k)[None, :] < n_elig[:, None]
        EF = np.where(valid, EF, -1.0)
        best_k = EF.argmax(axis=1) + 1
        best_v = EF.max(axis=1)
        choose = (best_v > E0) & (n_elig > 0)
        for gi in np.flatnonzero(choose):
            idx = IDX[s + gi, :best_k[gi]]
            mask[idx[idx >= 0]] = True
    return mask


def _best_claimant_mask(i1: np.ndarray, i2: np.ndarray, p: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Among available pairs, True where the pair's S1 has the highest p for that pool record."""
    p_av = np.where(available, p, -1.0)
    order = np.lexsort((i1, -p_av, i2))
    i2_sorted = i2[order]
    first = np.ones(len(i2), dtype=bool); first[1:] = i2_sorted[1:] != i2_sorted[:-1]
    m = np.zeros(len(i2), dtype=bool)
    m[order[first]] = True
    return m & available


def decide(i1: np.ndarray, i2: np.ndarray, p: np.ndarray, method: str = "ef", thr: float = 0.7, lam_miss: float = 0.02,
           min_p: float = 0.05, n_samples: int = 128, seed: int = 0, rounds: int = 1) -> np.ndarray:
    """Final boolean mask over candidate pairs."""
    p = p.astype(np.float32)
    if method == "thr":
        best = _best_claimant_mask(i1, i2, p, p >= thr)
        return best & (p >= thr)
    available = p >= min_p               # pairs still on offer
    taken = np.zeros(len(p), dtype=bool)  # pool record already assigned
    final = np.zeros(len(p), dtype=bool)
    pool_taken = np.zeros(i2.max() + 1 if len(i2) else 0, dtype=bool)
    for r in range(rounds):
        offered = _best_claimant_mask(i1, i2, p, available & ~pool_taken[i2])
        # entities already finalised keep their picks; others decide with the currently offered records
        p_eff = np.where(offered | final, p, 0.0).astype(np.float32)
        # truth mass: records this entity may still predict or has already taken (records lost to a stronger
        # claimant most likely belong to that claimant and are excluded, as are records below min_p)
        sel = expected_f05_select(i1, p_eff, p_eff, lam_miss=lam_miss, n_samples=n_samples, seed=seed + r)
        newly = sel & offered & ~pool_taken[i2]
        final |= newly
        pool_taken[i2[newly]] = True
        # records offered but rejected are withdrawn from that claimant and re-offered to the next one
        rejected = offered & ~sel
        available &= ~rejected
        if not rejected.any():
            break
    return final


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

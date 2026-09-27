"""Meta-blocking pruner: a cheap LightGBM model on blocking-stage signals that trims the raw candidate
union (25-50 per S1) to the set the matcher scores (~10 per S1) while keeping >99.5% of true pairs.
Policy: keep p >= tau, always keep the top-`keep_top` candidates of every S1, cap at `cap` per S1.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from .blocking import CH_ADDR, CH_COMBO, CH_GLOBAL, CH_KEY, CH_NAME, CH_REV
from .features import group_rank_gap

PRUNE_FEATURES = [
    "s_name", "s_combo", "s_addr", "ch_name", "ch_combo", "ch_addr", "ch_key", "ch_rev", "ch_global",
    "n_first_eq", "n_core_eq", "n_sorted_eq", "n_compact_eq", "n_skel_eq", "hn_both", "hn_eq", "hn_prefix", "hn_sub",
    "hn_conflict", "a_empty", "reg_same", "reg_conflict", "src3", "loc_inter", "street_inter", "nt_jacc", "nt_inter",
    "s_combo_rank1", "s_combo_gap1", "s_combo_rank2", "s_combo_gap2", "s_name_rank1", "s_name_gap1", "s_name_rank2",
    "s_addr_rank1", "cand_per_s1", "claimants_per_pool", "amb_s1_name", "amb_pool_name_in_s1", "n_ntok1", "n_ntok2",
]


def prune_features(S: pd.DataFrame, Q: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """Cheap features for the pruner (a subset of stage-1 features, computed directly)."""
    i1 = C["i1"].to_numpy(); i2 = C["i2"].to_numpy()
    F: Dict[str, np.ndarray] = {}
    for c in ("s_name", "s_combo", "s_addr"):
        F[c] = C[c].to_numpy(np.float32)
    ch = C["ch"].to_numpy()
    for nm, bit in (("ch_name", CH_NAME), ("ch_combo", CH_COMBO), ("ch_addr", CH_ADDR), ("ch_key", CH_KEY), ("ch_rev", CH_REV), ("ch_global", CH_GLOBAL)):
        F[nm] = ((ch & bit) > 0).astype(np.float32)
    def eq(col_a, col_b, nonempty=True):
        a = S[col_a].to_numpy()[i1]; b = Q[col_b].to_numpy()[i2]
        e = a == b
        if nonempty:
            e &= a != ""
        return e.astype(np.float32)
    F["n_first_eq"] = eq("n_first", "n_first"); F["n_core_eq"] = eq("n_core2", "n_core2")
    F["n_sorted_eq"] = eq("n_sorted2", "n_sorted2"); F["n_compact_eq"] = eq("n_compact", "n_compact"); F["n_skel_eq"] = eq("n_skel", "n_skel")
    hn1 = S["a_hn"].to_numpy()[i1]; hn2 = Q["a_hn"].to_numpy()[i2]
    r1 = S["a_hn_runs"].to_numpy()[i1]; r2 = Q["a_hn_runs"].to_numpy()[i2]
    both = (hn1 != "") & (hn2 != "")
    F["hn_both"] = both.astype(np.float32)
    F["hn_eq"] = (both & (hn1 == hn2)).astype(np.float32)
    pref = np.zeros(len(i1), np.float32); sub = np.zeros(len(i1), np.float32); conf = np.zeros(len(i1), np.float32)
    for k in np.flatnonzero(both & (hn1 != hn2)):
        a, b = hn1[k], hn2[k]
        if (a.startswith(b) and len(a) - len(b) == 1) or (b.startswith(a) and len(b) - len(a) == 1):
            pref[k] = 1
        ra = r1[k].split(); rb = r2[k].split()
        if a in rb or b in ra:
            sub[k] = 1
        elif not set(ra) & set(rb) and not pref[k]:
            conf[k] = 1
    F["hn_prefix"] = pref; F["hn_sub"] = sub; F["hn_conflict"] = conf
    F["a_empty"] = Q["addr_empty"].to_numpy()[i2].astype(np.float32)
    pa = S["part"].to_numpy()[i1]; pb = Q["part"].to_numpy()[i2]
    F["reg_same"] = ((pa != "") & (pa == pb)).astype(np.float32)
    F["reg_conflict"] = ((pa != "") & (pb != "") & (pa != pb)).astype(np.float32)
    F["src3"] = (Q["src"].to_numpy()[i2] == 3).astype(np.float32)
    def inter(col):
        a = S[col].to_numpy()[i1]; b = Q[col].to_numpy()[i2]
        out = np.zeros(len(i1), np.float32)
        for k in range(len(i1)):
            if a[k] and b[k]:
                out[k] = len(set(a[k].split()) & set(b[k].split()))
        return out
    F["loc_inter"] = inter("a_loc2"); F["street_inter"] = inter("a_street")
    a = S["n_core2"].to_numpy()[i1]; b = Q["n_core2"].to_numpy()[i2]
    it = np.zeros(len(i1), np.float32); ja = np.zeros(len(i1), np.float32)
    for k in range(len(i1)):
        sa = set(a[k].split()); sb = set(b[k].split())
        if sa and sb:
            x = len(sa & sb); it[k] = x; ja[k] = x / len(sa | sb)
    F["nt_inter"] = it; F["nt_jacc"] = ja
    for sc in ("s_combo", "s_name", "s_addr"):
        r, g, bst, sz = group_rank_gap(i1, F[sc]); F[f"{sc}_rank1"] = r; F[f"{sc}_gap1"] = g
        r2_, g2, b2, sz2 = group_rank_gap(i2, F[sc]); F[f"{sc}_rank2"] = r2_; F[f"{sc}_gap2"] = g2
    F["cand_per_s1"] = sz; F["claimants_per_pool"] = sz2
    core_s = pd.Series(S["n_core2"].to_numpy()); vc = core_s.value_counts()
    F["amb_s1_name"] = np.log1p(core_s.map(vc).to_numpy(np.float32)[i1])
    F["amb_pool_name_in_s1"] = np.log1p(pd.Series(Q["n_core2"].to_numpy()).map(vc).fillna(0).to_numpy(np.float32)[i2])
    F["n_ntok1"] = S["n_ntok"].to_numpy()[i1].astype(np.float32); F["n_ntok2"] = Q["n_ntok"].to_numpy()[i2].astype(np.float32)
    return pd.DataFrame({k: F[k] for k in PRUNE_FEATURES})


def fit_pruner(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int = 42, n_threads: int = 2) -> lgb.Booster:
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    hold = np.isin(groups, rng.choice(ug, max(1, len(ug) // 10), replace=False))
    params = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=n_threads, seed=seed)
    dtr = lgb.Dataset(X[~hold], y[~hold]); dva = lgb.Dataset(X[hold], y[hold])
    return lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva], callbacks=[lgb.early_stopping(40, verbose=False)])


def prune_mask(i1: np.ndarray, p: np.ndarray, tau: float = 0.003, keep_top: int = 2, cap: int = 40) -> np.ndarray:
    rank, gap, best, size = group_rank_gap(i1, p)
    return ((p >= tau) | (rank < keep_top)) & (rank < cap)

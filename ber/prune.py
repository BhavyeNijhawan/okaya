"""Meta-blocking pruner: a cheap GBDT on blocking-stage signals that trims the raw candidate union
(25-50 per S1) to the set the matcher scores (~5-10 per S1) while keeping >99.5% of true pairs.
Policy: keep p >= tau, always keep the top-`keep_top` candidates of every S1, cap at `cap` per S1.

All features are vectorised (numpy string ops + sparse token-set arithmetic); competition features
(ranks, claimant counts) are computed over the FULL raw candidate frame of the country so that the
pruner sees the same distribution at fit time and at inference time.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .blocking import CH_ADDR, CH_COMBO, CH_GLOBAL, CH_KEY, CH_NAME, CH_REV
from .features import TokenSpace, _obj, group_rank_gap

PRUNE_FEATURES = [
    "s_name", "s_combo", "s_addr", "ch_name", "ch_combo", "ch_addr", "ch_key", "ch_rev", "ch_global",
    "n_first_eq", "n_core_eq", "n_sorted_eq", "n_compact_eq", "n_skel_eq", "hn_both", "hn_eq", "hn_prefix", "hn_sub",
    "hn_conflict", "a_empty", "reg_same", "reg_conflict", "src3", "loc_inter", "street_inter", "nt_jacc", "nt_inter",
    "s_combo_rank1", "s_combo_gap1", "s_combo_rank2", "s_combo_gap2", "s_name_rank1", "s_name_gap1", "s_name_rank2",
    "s_addr_rank1", "cand_per_s1", "claimants_per_pool", "amb_s1_name", "amb_pool_name_in_s1", "n_ntok1", "n_ntok2",
]


def _hn_flags(hn1: np.ndarray, hn2: np.ndarray, runs1: np.ndarray, runs2: np.ndarray) -> Dict[str, np.ndarray]:
    both = (hn1 != "") & (hn2 != "")
    eq = both & (hn1 == hn2)
    a = np.char.asarray(hn1.astype(str)); b = np.char.asarray(hn2.astype(str))
    la = np.char.str_len(a); lb = np.char.str_len(b)
    pref = both & ~eq & ((np.char.startswith(a, b) & (la == lb + 1)) | (np.char.startswith(b, a) & (lb == la + 1)))
    r1 = np.char.asarray(np.char.add(np.char.add(" ", runs1.astype(str)), " "))
    r2 = np.char.asarray(np.char.add(np.char.add(" ", runs2.astype(str)), " "))
    sub = both & ~eq & ((np.char.find(r1, np.char.add(np.char.add(" ", b), " ")) >= 0) |
                        (np.char.find(r2, np.char.add(np.char.add(" ", a), " ")) >= 0))
    conflict = both & ~eq & ~pref & ~sub
    return {"hn_both": both.astype(np.float32), "hn_eq": eq.astype(np.float32), "hn_prefix": pref.astype(np.float32),
            "hn_sub": sub.astype(np.float32), "hn_conflict": conflict.astype(np.float32)}


class PruneContext:
    """Per-country state for pruner features: record columns, token spaces, ambiguity counts, and the
    competition features (ranks / group sizes) computed once over the FULL candidate frame."""

    def __init__(self, S: pd.DataFrame, Q: pd.DataFrame, C: pd.DataFrame):
        self.s = {c: _obj(S, c) for c in ("n_first", "n_core2", "n_sorted2", "n_compact", "n_skel", "a_hn", "a_hn_runs", "part")}
        self.q = {c: _obj(Q, c) for c in ("n_first", "n_core2", "n_sorted2", "n_compact", "n_skel", "a_hn", "a_hn_runs", "part")}
        self.q_empty = Q["addr_empty"].to_numpy().astype(np.float32); self.q_src = Q["src"].to_numpy()
        self.s_ntok = S["n_ntok"].to_numpy().astype(np.float32); self.q_ntok = Q["n_ntok"].to_numpy().astype(np.float32)
        self.ts_core = TokenSpace(self.s["n_core2"].tolist(), self.q["n_core2"].tolist())
        self.ts_loc = TokenSpace(_obj(S, "a_loc2").tolist(), _obj(Q, "a_loc2").tolist())
        self.ts_street = TokenSpace(_obj(S, "a_street").tolist(), _obj(Q, "a_street").tolist())
        core_s = pd.Series(self.s["n_core2"]); vc = core_s.value_counts()
        self.amb_s1 = np.log1p(core_s.map(vc).to_numpy(np.float32))
        self.amb_pool = np.log1p(pd.Series(self.q["n_core2"]).map(vc).fillna(0).to_numpy(np.float32))
        i1_all = C["i1"].to_numpy(); i2_all = C["i2"].to_numpy()
        self.ctx: Dict[str, np.ndarray] = {}
        for sc in ("s_combo", "s_name", "s_addr"):
            v = C[sc].to_numpy(np.float32)
            r, g, bst, sz = group_rank_gap(i1_all, v); self.ctx[f"{sc}_rank1"] = r; self.ctx[f"{sc}_gap1"] = g
            r2_, g2, b2, sz2 = group_rank_gap(i2_all, v); self.ctx[f"{sc}_rank2"] = r2_; self.ctx[f"{sc}_gap2"] = g2
        self.ctx["cand_per_s1"] = sz; self.ctx["claimants_per_pool"] = sz2

    def features(self, C: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
        """Pruner features for the candidate rows `rows` (positions in the full frame C)."""
        i1 = C["i1"].to_numpy()[rows]; i2 = C["i2"].to_numpy()[rows]
        s, q = self.s, self.q
        F: Dict[str, np.ndarray] = {}
        for c in ("s_name", "s_combo", "s_addr"):
            F[c] = C[c].to_numpy(np.float32)[rows]
        ch = C["ch"].to_numpy()[rows]
        for nm, bit in (("ch_name", CH_NAME), ("ch_combo", CH_COMBO), ("ch_addr", CH_ADDR), ("ch_key", CH_KEY), ("ch_rev", CH_REV), ("ch_global", CH_GLOBAL)):
            F[nm] = ((ch & bit) > 0).astype(np.float32)
        for col, name in (("n_first", "n_first_eq"), ("n_core2", "n_core_eq"), ("n_sorted2", "n_sorted_eq"), ("n_compact", "n_compact_eq"), ("n_skel", "n_skel_eq")):
            a = s[col][i1]; b = q[col][i2]
            F[name] = ((a == b) & (a != "")).astype(np.float32)
        F.update(_hn_flags(s["a_hn"][i1], q["a_hn"][i2], s["a_hn_runs"][i1], q["a_hn_runs"][i2]))
        F["a_empty"] = self.q_empty[i2]
        pa_ = s["part"][i1]; pb_ = q["part"][i2]
        F["reg_same"] = ((pa_ != "") & (pa_ == pb_)).astype(np.float32)
        F["reg_conflict"] = ((pa_ != "") & (pb_ != "") & (pa_ != pb_)).astype(np.float32)
        F["src3"] = (self.q_src[i2] == 3).astype(np.float32)
        st = self.ts_core.pair_stats(i1, i2, weighted=False); F["nt_inter"] = st["inter"]; F["nt_jacc"] = st["jacc"]
        F["loc_inter"] = self.ts_loc.pair_stats(i1, i2, weighted=False)["inter"]
        F["street_inter"] = self.ts_street.pair_stats(i1, i2, weighted=False)["inter"]
        F["amb_s1_name"] = self.amb_s1[i1]; F["amb_pool_name_in_s1"] = self.amb_pool[i2]
        F["n_ntok1"] = self.s_ntok[i1]; F["n_ntok2"] = self.q_ntok[i2]
        for k, v in self.ctx.items():
            F[k] = v[rows]
        return pd.DataFrame({k: F[k] for k in PRUNE_FEATURES})


def prune_features(S: pd.DataFrame, Q: pd.DataFrame, C: pd.DataFrame, rows=None) -> pd.DataFrame:
    """Pruner features for rows of C (default: all). Competition features use the full frame."""
    ctx = PruneContext(S, Q, C)
    rows = np.arange(len(C)) if rows is None else rows
    return ctx.features(C, rows)


def prune_scores(S: pd.DataFrame, Q: pd.DataFrame, C: pd.DataFrame, model, chunk: int = 1_000_000) -> np.ndarray:
    """Pruner probability for every row of C, computed chunk by chunk (memory-safe at tens of millions of pairs)."""
    ctx = PruneContext(S, Q, C)
    out = np.empty(len(C), dtype=np.float32)
    for s0 in range(0, len(C), chunk):
        rows = np.arange(s0, min(s0 + chunk, len(C)))
        out[s0:s0 + len(rows)] = model.predict(ctx.features(C, rows))
    return out


def prune_mask(i1: np.ndarray, p: np.ndarray, tau: float = 0.002, keep_top: int = 3, cap: int = 40) -> np.ndarray:
    rank, gap, best, size = group_rank_gap(i1, p)
    return ((p >= tau) | (rank < keep_top)) & (rank < cap)

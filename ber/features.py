"""Pairwise, context and stage-2 (sibling / competition) features for candidate pairs.

All heavy work is vectorised: rapidfuzz `cpdist` for string similarities, sparse incidence matrices
for token-set arithmetic, numpy group operations for ranks. Learned tables (legal-form transition
rates, extra/missing-token rates) are fitted on labelled candidate pairs of S1 entities that are
disjoint from the matcher's training entities (see train.py).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import Indel, JaroWinkler
from rapidfuzz.process import cpdist

from .blocking import CH_ADDR, CH_COMBO, CH_GLOBAL, CH_KEY, CH_NAME, CH_REV
from .translit import has_indic


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _sim(a: List[str], b: List[str], scorer, workers: int = -1) -> np.ndarray:
    if len(a) == 0:
        return np.zeros(0, dtype=np.float32)
    return cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _take(col: np.ndarray, idx: np.ndarray) -> List[str]:
    return col[idx].tolist()


class TokenSpace:
    """Binary token incidence matrices for two frames over a shared vocabulary."""

    def __init__(self, docs_a: List[str], docs_b: List[str], min_len: int = 1):
        vocab: Dict[str, int] = {}
        self.A = self._build(docs_a, vocab, min_len, grow=True)
        self.B = self._build(docs_b, vocab, min_len, grow=True)
        self.A = self.A.tocsr(); self.B = self.B.tocsr()
        V = len(vocab)
        self.A.resize((self.A.shape[0], V)); self.B.resize((self.B.shape[0], V))
        self.vocab = vocab
        self.na = np.asarray(self.A.sum(axis=1)).ravel().astype(np.float32)
        self.nb = np.asarray(self.B.sum(axis=1)).ravel().astype(np.float32)
        # idf from A (the reference side)
        df = np.asarray(self.A.sum(axis=0)).ravel() + np.asarray(self.B.sum(axis=0)).ravel()
        N = self.A.shape[0] + self.B.shape[0]
        self.idf = np.log((1.0 + N) / (1.0 + df)).astype(np.float32)

    @staticmethod
    def _build(docs: List[str], vocab: Dict[str, int], min_len: int, grow: bool) -> sp.coo_matrix:
        rows, cols = [], []
        for i, d in enumerate(docs):
            if not d:
                continue
            seen = set()
            for t in d.split():
                if len(t) < min_len or t in seen:
                    continue
                seen.add(t)
                j = vocab.get(t)
                if j is None:
                    if not grow:
                        continue
                    j = len(vocab); vocab[t] = j
                rows.append(i); cols.append(j)
        V = max(len(vocab), 1)
        return sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(docs), V))

    def pair_stats(self, i1: np.ndarray, i2: np.ndarray, weighted: bool = True) -> Dict[str, np.ndarray]:
        X = self.A[i1]; Y = self.B[i2]
        inter_m = X.multiply(Y)
        inter = np.asarray(inter_m.sum(axis=1)).ravel().astype(np.float32)
        na, nb = self.na[i1], self.nb[i2]
        union = na + nb - inter
        out = {
            "inter": inter,
            "jacc": np.where(union > 0, inter / np.maximum(union, 1e-6), 0).astype(np.float32),
            "overlap": np.where(np.minimum(na, nb) > 0, inter / np.maximum(np.minimum(na, nb), 1e-6), 0).astype(np.float32),
            "extra": (nb - inter).astype(np.float32),     # tokens only on the pool side
            "missing": (na - inter).astype(np.float32),   # tokens only on the S1 side
            "na": na, "nb": nb,
        }
        if weighted:
            w = sp.diags(self.idf, format="csr")
            wi = np.asarray((inter_m @ w).sum(axis=1)).ravel().astype(np.float32)
            wa = np.asarray((X @ w).sum(axis=1)).ravel().astype(np.float32)
            wb = np.asarray((Y @ w).sum(axis=1)).ravel().astype(np.float32)
            wu = wa + wb - wi
            out["wjacc"] = np.where(wu > 0, wi / np.maximum(wu, 1e-6), 0).astype(np.float32)
            out["wextra"] = (wb - wi).astype(np.float32)
            out["wmissing"] = (wa - wi).astype(np.float32)
            out["_extra_m"] = Y - inter_m       # sparse indicator of extra tokens (pool only)
            out["_missing_m"] = X - inter_m     # sparse indicator of missing tokens (S1 only)
        return out


# ---------------------------------------------------------------------------------------------
# learned tables
# ---------------------------------------------------------------------------------------------
class RateTables:
    """P(match | token is extra), P(match | token is missing), P(match | legal transition), smoothed."""

    def __init__(self, alpha: float = 20.0):
        self.alpha = alpha
        self.extra: Dict[str, Tuple[float, int]] = {}
        self.missing: Dict[str, Tuple[float, int]] = {}
        self.legal: Dict[str, Tuple[float, int]] = {}
        self.prior_extra = 0.05; self.prior_missing = 0.05; self.prior_legal = 0.1

    def fit(self, S: pd.DataFrame, Q: pd.DataFrame, i1: np.ndarray, i2: np.ndarray, y: np.ndarray) -> "RateTables":
        core_a = S["n_core2"].to_numpy(); core_b = Q["n_core2"].to_numpy()
        leg_a = S["n_legal"].to_numpy(); leg_b = Q["n_legal"].to_numpy()
        ce: Dict[str, List[int]] = defaultdict(lambda: [0, 0]); cm: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        cl: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for a, b, lab in zip(i1.tolist(), i2.tolist(), y.tolist()):
            ta = set(core_a[a].split()); tb = set(core_b[b].split())
            for t in tb - ta:
                ce[t][0] += lab; ce[t][1] += 1
            for t in ta - tb:
                cm[t][0] += lab; cm[t][1] += 1
            k = f"{leg_a[a]}->{leg_b[b]}"
            cl[k][0] += lab; cl[k][1] += 1
        self.prior_extra = float(y.mean()) if len(y) else 0.05
        self.prior_missing = self.prior_extra; self.prior_legal = self.prior_extra
        al = self.alpha
        self.extra = {t: ((m + al * self.prior_extra) / (n + al), n) for t, (m, n) in ce.items() if n >= 5}
        self.missing = {t: ((m + al * self.prior_missing) / (n + al), n) for t, (m, n) in cm.items() if n >= 5}
        self.legal = {t: ((m + al * self.prior_legal) / (n + al), n) for t, (m, n) in cl.items() if n >= 5}
        return self

    def rate_vector(self, vocab: Dict[str, int], which: str) -> np.ndarray:
        table = self.extra if which == "extra" else self.missing
        prior = self.prior_extra if which == "extra" else self.prior_missing
        v = np.full(len(vocab), prior, dtype=np.float32)
        for t, j in vocab.items():
            r = table.get(t)
            if r is not None:
                v[j] = r[0]
        return v

    def legal_rate(self, la: List[str], lb: List[str]) -> np.ndarray:
        out = np.empty(len(la), dtype=np.float32)
        for k, (a, b) in enumerate(zip(la, lb)):
            r = self.legal.get(f"{a}->{b}")
            out[k] = r[0] if r is not None else self.prior_legal
        return out

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"alpha": self.alpha, "extra": self.extra, "missing": self.missing, "legal": self.legal,
                                    "prior_extra": self.prior_extra, "prior_missing": self.prior_missing,
                                    "prior_legal": self.prior_legal}, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "RateTables":
        d = json.loads(path.read_text(encoding="utf-8"))
        t = cls(d["alpha"])
        t.extra = {k: tuple(v) for k, v in d["extra"].items()}
        t.missing = {k: tuple(v) for k, v in d["missing"].items()}
        t.legal = {k: tuple(v) for k, v in d["legal"].items()}
        t.prior_extra, t.prior_missing, t.prior_legal = d["prior_extra"], d["prior_missing"], d["prior_legal"]
        return t


def _rowmin_from_indicator(M: sp.csr_matrix, rate: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(min rate, max rate) over the indicated tokens of each row; rows without tokens -> (1, 0)."""
    M = M.tocsr(); M.eliminate_zeros()
    n = M.shape[0]
    mn = np.ones(n, dtype=np.float32); mx = np.zeros(n, dtype=np.float32)
    if M.nnz == 0:
        return mn, mx
    vals = rate[M.indices]
    rows = np.repeat(np.arange(n), np.diff(M.indptr))
    np.minimum.at(mn, rows, vals)
    np.maximum.at(mx, rows, vals)
    return mn, mx


# ---------------------------------------------------------------------------------------------
# group statistics
# ---------------------------------------------------------------------------------------------
def group_rank_gap(key: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Within groups of `key`: rank of score (0 = best), gap to the best, best score, group size."""
    order = np.lexsort((-score, key))
    k_sorted = key[order]
    start = np.ones(len(key), dtype=bool); start[1:] = k_sorted[1:] != k_sorted[:-1]
    grp_start_idx = np.flatnonzero(start)
    grp_id = np.cumsum(start) - 1
    pos = np.arange(len(key)) - grp_start_idx[grp_id]
    best = score[order][grp_start_idx][grp_id]
    sizes = np.diff(np.append(grp_start_idx, len(key)))[grp_id]
    rank = np.empty(len(key), dtype=np.float32); gap = np.empty(len(key), dtype=np.float32)
    bst = np.empty(len(key), dtype=np.float32); sz = np.empty(len(key), dtype=np.float32)
    rank[order] = pos; gap[order] = best - score[order]; bst[order] = best; sz[order] = sizes
    return rank, gap, bst, sz


def group_second(key: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Second-best score in the group (0 if the group has one member)."""
    order = np.lexsort((-score, key))
    k_sorted = key[order]; s_sorted = score[order]
    start = np.ones(len(key), dtype=bool); start[1:] = k_sorted[1:] != k_sorted[:-1]
    grp_start_idx = np.flatnonzero(start)
    grp_id = np.cumsum(start) - 1
    sizes = np.diff(np.append(grp_start_idx, len(key)))
    second_pos = grp_start_idx + 1
    second = np.where(sizes >= 2, s_sorted[np.minimum(second_pos, len(key) - 1)], 0.0)
    out = np.empty(len(key), dtype=np.float32)
    out[order] = second[grp_id]
    return out


def group_sum_count(key: np.ndarray, score: np.ndarray, thr: float) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"k": key, "s": score, "c": (score >= thr).astype(np.float32)})
    g = df.groupby("k", sort=False)
    return g["s"].transform("sum").to_numpy(np.float32), g["c"].transform("sum").to_numpy(np.float32)


# ---------------------------------------------------------------------------------------------
# stage-1 features
# ---------------------------------------------------------------------------------------------
def _hn_features(hn1: np.ndarray, hn2: np.ndarray, runs1: np.ndarray, runs2: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(hn1)
    both = np.zeros(n, np.float32); eq = np.zeros(n, np.float32); pref = np.zeros(n, np.float32)
    sub = np.zeros(n, np.float32); rinter = np.zeros(n, np.float32); rjacc = np.zeros(n, np.float32)
    diff = np.full(n, -1.0, np.float32); one_missing = np.zeros(n, np.float32); conflict = np.zeros(n, np.float32)
    for k in range(n):
        a, b = hn1[k], hn2[k]
        if a and b:
            both[k] = 1
            if a == b:
                eq[k] = 1
            else:
                if (a.startswith(b) and len(a) - len(b) == 1) or (b.startswith(a) and len(b) - len(a) == 1):
                    pref[k] = 1
                if a.isdigit() and b.isdigit() and len(a) == len(b):
                    d = abs(int(a) - int(b)); diff[k] = min(d, 10_000)
            ra = set(runs1[k].split()); rb = set(runs2[k].split())
            it = len(ra & rb)
            rinter[k] = it
            rjacc[k] = it / max(1, len(ra | rb))
            if (a in rb) or (b in ra):
                sub[k] = 1
            if it == 0 and not pref[k]:
                conflict[k] = 1
        elif a or b:
            one_missing[k] = 1
    return {"hn_both": both, "hn_eq": eq, "hn_prefix": pref, "hn_sub": sub, "hn_runs_inter": rinter, "hn_runs_jacc": rjacc,
            "hn_absdiff": diff, "hn_one_missing": one_missing, "hn_conflict": conflict}


def _dom_cover(dom: np.ndarray, compact: np.ndarray) -> np.ndarray:
    """Fraction of the domain-core letters covered by the other name's letters (multiset), 0 if no domain."""
    out = np.zeros(len(dom), np.float32)
    for k in range(len(dom)):
        d = dom[k]
        if not d:
            continue
        c = Counter(compact[k]); cov = 0
        for ch in d:
            if c[ch] > 0:
                c[ch] -= 1; cov += 1
        out[k] = cov / max(1, len(d))
    return out


def _obj(df: pd.DataFrame, col: str) -> np.ndarray:
    """Column as a numpy object array of Python str (works for object, category and Arrow-backed columns)."""
    a = np.asarray(df[col].astype(object).to_numpy(), dtype=object)
    if len(a) and not isinstance(a[0], str):
        a = np.array(["" if (x is None or x is pd.NA) else str(x) for x in a], dtype=object)
    return a


class ArrowCol:
    """A string column kept as a pyarrow array; rows are materialised per chunk with `take`."""

    def __init__(self, series: pd.Series):
        vals = series.tolist()
        self.arr = pa.array(["" if (x is None or x is pd.NA) else str(x) for x in vals], type=pa.string())
        self.n = len(self.arr)

    def take(self, idx: np.ndarray) -> np.ndarray:
        out = pc.take(self.arr, pa.array(idx, type=pa.int64())).to_pylist()
        return np.array(["" if x is None else x for x in out], dtype=object)

    def tolist(self) -> List[str]:
        return ["" if x is None else x for x in self.arr.to_pylist()]


class FeatureContext:
    """Per-country state shared by all feature chunks: record columns, token spaces, ambiguity counts.

    Long text columns are held as pyarrow arrays (compact), short ones as numpy object arrays.
    """

    TEXT_COLS = ("n_core2", "n_name", "n_compact", "n_sorted2", "n_skel", "a_alpha", "a_street", "a_loc2", "a_nums", "a_hn_runs")
    SHORT_COLS = ("n_first", "n_legal", "n_dom", "a_stype", "a_hn", "a_unit", "region", "part")

    def __init__(self, S: pd.DataFrame, Q: pd.DataFrame, tables: RateTables):
        self.tables = tables
        self.s_text = {c: ArrowCol(S[c]) for c in self.TEXT_COLS}
        self.q_text = {c: ArrowCol(Q[c]) for c in self.TEXT_COLS + ("n_name_alt",)}
        self.s = {c: _obj(S, c) for c in self.SHORT_COLS}
        self.q = {c: _obj(Q, c) for c in self.SHORT_COLS}
        self.s_raw = ArrowCol(pd.Series([x.lower() for x in _obj(S, "raw_name")]))
        q_raw = _obj(Q, "raw_name")
        self.q_script = np.array([1.0 if has_indic(x) else 0.0 for x in q_raw], dtype=np.float32)
        self.q_raw = ArrowCol(pd.Series([x.lower() for x in q_raw]))
        del q_raw
        self.s_ntok = S["n_ntok"].to_numpy().astype(np.float32); self.q_ntok = Q["n_ntok"].to_numpy().astype(np.float32)
        self.s_ncomp = S["a_ncomp"].to_numpy().astype(np.float32); self.q_ncomp = Q["a_ncomp"].to_numpy().astype(np.float32)
        self.q_empty = Q["addr_empty"].to_numpy().astype(np.float32); self.q_src = Q["src"].to_numpy()
        self.q_hnkw = Q["a_has_hn_kw"].to_numpy().astype(np.float32)
        core_s = self.s_text["n_core2"].tolist(); core_q = self.q_text["n_core2"].tolist()
        self.ts_core = TokenSpace(core_s, core_q)
        self.rate_extra = tables.rate_vector(self.ts_core.vocab, "extra")
        self.rate_missing = tables.rate_vector(self.ts_core.vocab, "missing")
        self.ts_addr = {}
        for name, col in (("street", "a_street"), ("loc", "a_loc2"), ("nums", "a_nums"), ("alpha", "a_alpha")):
            self.ts_addr[name] = TokenSpace(self.s_text[col].tolist(), self.q_text[col].tolist())
        s_core = pd.Series(core_s); vc = s_core.value_counts()
        self.s1_core_freq = np.log1p(s_core.map(vc).to_numpy(np.float32))
        s_cp = pd.Series([f"{a}|{b}" for a, b in zip(core_s, self.s["part"])])
        self.s1_core_part_freq = np.log1p(s_cp.map(s_cp.value_counts()).to_numpy(np.float32))
        q_core = pd.Series(core_q)
        self.pool_core_freq = np.log1p(q_core.map(q_core.value_counts()).to_numpy(np.float32))
        self.pool_in_s1 = np.log1p(q_core.map(vc).fillna(0).to_numpy(np.float32))
        del core_s, core_q, s_core, q_core, s_cp

    def pair_features(self, i1: np.ndarray, i2: np.ndarray, C_chunk: pd.DataFrame) -> Dict[str, np.ndarray]:
        s, q = self.s, self.q
        A = {c: self.s_text[c].take(i1) for c in self.TEXT_COLS}
        Bq = {c: self.q_text[c].take(i2) for c in self.TEXT_COLS + ("n_name_alt",)}
        F: Dict[str, np.ndarray] = {}
        F["s_name"] = C_chunk["s_name"].to_numpy(np.float32); F["s_combo"] = C_chunk["s_combo"].to_numpy(np.float32)
        F["s_addr"] = C_chunk["s_addr"].to_numpy(np.float32)
        ch = C_chunk["ch"].to_numpy()
        for nm, bit in (("ch_name", CH_NAME), ("ch_combo", CH_COMBO), ("ch_addr", CH_ADDR), ("ch_key", CH_KEY), ("ch_rev", CH_REV), ("ch_global", CH_GLOBAL)):
            F[nm] = ((ch & bit) > 0).astype(np.float32)
        a_core = A["n_core2"].tolist(); b_core = Bq["n_core2"].tolist()
        a_name = A["n_name"].tolist(); b_name = Bq["n_name"].tolist()
        a_comp = A["n_compact"].tolist(); b_comp = Bq["n_compact"].tolist()
        a_raw = self.s_raw.take(i1); b_raw = self.q_raw.take(i2)
        F["n_ratio"] = _sim(a_core, b_core, fuzz.ratio) / 100
        F["n_tset"] = _sim(a_core, b_core, fuzz.token_set_ratio) / 100
        F["n_tsort"] = _sim(a_core, b_core, fuzz.token_sort_ratio) / 100
        F["n_partial"] = _sim(a_core, b_core, fuzz.partial_ratio) / 100
        F["n_full_ratio"] = _sim(a_name, b_name, fuzz.ratio) / 100
        F["n_compact_jw"] = _sim(a_comp, b_comp, JaroWinkler.normalized_similarity)
        F["n_compact_indel"] = _sim(a_comp, b_comp, Indel.normalized_similarity)
        F["n_raw_ratio"] = _sim(a_raw.tolist(), b_raw.tolist(), fuzz.ratio) / 100
        F["n_raw_eq"] = (a_raw == b_raw).astype(np.float32)
        F["n_eq"] = (A["n_name"] == Bq["n_name"]).astype(np.float32)
        F["n_core_eq"] = ((A["n_core2"] == Bq["n_core2"]) & (A["n_core2"] != "")).astype(np.float32)
        F["n_sorted_eq"] = (A["n_sorted2"] == Bq["n_sorted2"]).astype(np.float32)
        F["n_compact_eq"] = (A["n_compact"] == Bq["n_compact"]).astype(np.float32)
        F["n_skel_eq"] = (A["n_skel"] == Bq["n_skel"]).astype(np.float32)
        F["n_skel_ratio"] = _sim(A["n_skel"].tolist(), Bq["n_skel"].tolist(), fuzz.ratio) / 100
        fa = s["n_first"][i1]; fb = q["n_first"][i2]
        F["n_first_eq"] = ((fa == fb) & (fa != "")).astype(np.float32)
        b_alt = Bq["n_name_alt"]
        F["n_has_alt"] = (b_alt != "").astype(np.float32)
        F["n_alt_ratio"] = np.where(F["n_has_alt"] > 0, _sim(a_core, b_alt.tolist(), fuzz.token_set_ratio) / 100, 0).astype(np.float32)
        b_dom = q["n_dom"][i2]
        F["n_has_dom"] = (b_dom != "").astype(np.float32)
        F["n_dom_cover"] = _dom_cover(b_dom, A["n_compact"])
        F["n_dom_partial"] = np.where(F["n_has_dom"] > 0, _sim(b_dom.tolist(), a_comp, fuzz.partial_ratio) / 100, 0).astype(np.float32)
        la = s["n_legal"][i1]; lb = q["n_legal"][i2]
        F["lg_both"] = ((la != "") & (lb != "")).astype(np.float32)
        F["lg_eq"] = ((la != "") & (la == lb)).astype(np.float32)
        F["lg_s1_only"] = ((la != "") & (lb == "")).astype(np.float32)
        F["lg_pool_only"] = ((lb != "") & (la == "")).astype(np.float32)
        F["lg_conflict"] = ((la != "") & (lb != "") & (la != lb)).astype(np.float32)
        F["lg_rate"] = self.tables.legal_rate(la.tolist(), lb.tolist())
        st = self.ts_core.pair_stats(i1, i2)
        for k in ("inter", "jacc", "overlap", "extra", "missing", "wjacc", "wextra", "wmissing"):
            F["nt_" + k] = st[k]
        F["nt_na"] = st["na"]; F["nt_nb"] = st["nb"]
        mn, mx = _rowmin_from_indicator(st["_extra_m"], self.rate_extra)
        F["nt_extra_rate_min"] = mn; F["nt_extra_rate_max"] = mx
        F["nt_extra_identity"] = np.asarray((st["_extra_m"] @ (self.rate_extra < 0.02).astype(np.float32))).ravel().astype(np.float32)
        F["nt_extra_noise"] = np.asarray((st["_extra_m"] @ (self.rate_extra > 0.15).astype(np.float32))).ravel().astype(np.float32)
        mn, mx = _rowmin_from_indicator(st["_missing_m"], self.rate_missing)
        F["nt_missing_rate_min"] = mn; F["nt_missing_rate_max"] = mx
        F["nt_missing_identity"] = np.asarray((st["_missing_m"] @ (self.rate_missing < 0.02).astype(np.float32))).ravel().astype(np.float32)
        del st
        F["a_empty"] = self.q_empty[i2]
        F["a_ncomp1"] = self.s_ncomp[i1]; F["a_ncomp2"] = self.q_ncomp[i2]
        a_alpha = A["a_alpha"].tolist(); b_alpha = Bq["a_alpha"].tolist()
        F["a_alpha_tset"] = _sim(a_alpha, b_alpha, fuzz.token_set_ratio) / 100
        F["a_alpha_ratio"] = _sim(a_alpha, b_alpha, fuzz.ratio) / 100
        F["a_street_ratio"] = _sim(A["a_street"].tolist(), Bq["a_street"].tolist(), fuzz.ratio) / 100
        F["a_loc_tset"] = _sim(A["a_loc2"].tolist(), Bq["a_loc2"].tolist(), fuzz.token_set_ratio) / 100
        F["a_loc_partial"] = _sim(A["a_loc2"].tolist(), Bq["a_loc2"].tolist(), fuzz.partial_ratio) / 100
        sa = s["a_stype"][i1]; sb = q["a_stype"][i2]
        F["a_stype_eq"] = ((sa != "") & (sa == sb)).astype(np.float32)
        F["a_stype_conflict"] = ((sa != "") & (sb != "") & (sa != sb)).astype(np.float32)
        ua = s["a_unit"][i1]; ub = q["a_unit"][i2]
        F["a_unit_eq"] = ((ua != "") & (ua == ub)).astype(np.float32)
        F["a_unit_conflict"] = ((ua != "") & (ub != "") & (ua != ub)).astype(np.float32)
        for k, v in _hn_features(s["a_hn"][i1], q["a_hn"][i2], A["a_hn_runs"], Bq["a_hn_runs"]).items():
            F[k] = v
        F["hn_kw_pool"] = self.q_hnkw[i2]
        for name, tsp in self.ts_addr.items():
            stp = tsp.pair_stats(i1, i2, weighted=(name != "nums"))
            F[f"{name}_inter"] = stp["inter"]; F[f"{name}_jacc"] = stp["jacc"]; F[f"{name}_overlap"] = stp["overlap"]
            F[f"{name}_na"] = stp["na"]; F[f"{name}_nb"] = stp["nb"]
            if name != "nums":
                F[f"{name}_wjacc"] = stp["wjacc"]; F[f"{name}_wextra"] = stp["wextra"]; F[f"{name}_wmissing"] = stp["wmissing"]
            del stp
        pa_ = s["part"][i1]; pb_ = q["part"][i2]; rb = q["region"][i2]
        F["reg_same"] = ((pa_ != "") & (pa_ == pb_)).astype(np.float32)
        F["reg_conflict"] = ((pa_ != "") & (pb_ != "") & (pa_ != pb_)).astype(np.float32)
        F["reg_pool_unknown"] = (rb == "").astype(np.float32)
        F["x_name_in_addr"] = np.maximum(_sim(a_core, b_alpha, fuzz.partial_token_set_ratio),
                                         _sim(b_core, a_alpha, fuzz.partial_token_set_ratio)) / 100
        F["src3"] = (self.q_src[i2] == 3).astype(np.float32)
        F["pool_script"] = self.q_script[i2]
        F["n_ntok1"] = self.s_ntok[i1]; F["n_ntok2"] = self.q_ntok[i2]
        F["n_len1"] = np.array([len(x) for x in a_core], np.float32); F["n_len2"] = np.array([len(x) for x in b_core], np.float32)
        F["amb_s1_name"] = self.s1_core_freq[i1]; F["amb_s1_name_part"] = self.s1_core_part_freq[i1]
        F["amb_pool_name"] = self.pool_core_freq[i2]; F["amb_pool_name_in_s1"] = self.pool_in_s1[i2]
        return F


def context_features(C: pd.DataFrame) -> pd.DataFrame:
    """Candidate-set context over the FULL candidate frame: ranks/gaps by blocking scores, group sizes."""
    i1 = C["i1"].to_numpy(); i2 = C["i2"].to_numpy()
    F: Dict[str, np.ndarray] = {}
    sz = sz2 = None
    for sc in ("s_combo", "s_name", "s_addr"):
        v = C[sc].to_numpy(np.float32)
        r, g, b, sz = group_rank_gap(i1, v)
        F[f"{sc}_rank1"] = r; F[f"{sc}_gap1"] = g; F[f"{sc}_best1"] = b
        r, g, b, sz2 = group_rank_gap(i2, v)
        F[f"{sc}_rank2"] = r; F[f"{sc}_gap2"] = g; F[f"{sc}_best2"] = b
    F["cand_per_s1"] = sz; F["claimants_per_pool"] = sz2
    if "pp" in C:
        pp = C["pp"].to_numpy(np.float32)
        F["pp"] = pp
        r, g, b, _ = group_rank_gap(i1, pp); F["pp_rank1"] = r; F["pp_gap1"] = g
        r, g, b, _ = group_rank_gap(i2, pp); F["pp_rank2"] = r; F["pp_gap2"] = g; F["pp_best2"] = b
    return pd.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})


def stage1_features(S: pd.DataFrame, Q: pd.DataFrame, C: pd.DataFrame, tables: RateTables, chunk: int = 1_000_000,
                    writer=None, log=print) -> Optional[pd.DataFrame]:
    """Stage-1 feature frame aligned with C. If `writer(df_chunk)` is given, chunks are streamed to it and
    None is returned; otherwise the full frame is returned."""
    ctx = FeatureContext(S, Q, tables)
    CF = context_features(C)
    i1_all = C["i1"].to_numpy(); i2_all = C["i2"].to_numpy()
    out_parts = []
    for s0 in range(0, len(C), chunk):
        sl = slice(s0, min(s0 + chunk, len(C)))
        F = ctx.pair_features(i1_all[sl], i2_all[sl], C.iloc[sl])
        df = pd.DataFrame({k: np.asarray(v, np.float32) for k, v in F.items()})
        df = pd.concat([df, CF.iloc[sl].reset_index(drop=True)], axis=1)
        if writer is not None:
            writer(df)
        else:
            out_parts.append(df)
        log(f"      features {sl.stop:,}/{len(C):,}")
    if writer is not None:
        return None
    return pd.concat(out_parts, ignore_index=True) if len(out_parts) > 1 else out_parts[0]


# ---------------------------------------------------------------------------------------------
# stage-2 features: score context + sibling evidence
# ---------------------------------------------------------------------------------------------
def stage2_features(S: pd.DataFrame, Q: pd.DataFrame, C: pd.DataFrame, p1: np.ndarray, anchor_thr: float = 0.9,
                    max_anchors: int = 3, sib_p1_cap: float = 0.95) -> pd.DataFrame:
    i1 = C["i1"].to_numpy(); i2 = C["i2"].to_numpy(); n = len(C)
    F: Dict[str, np.ndarray] = {"p1": p1.astype(np.float32)}
    logit = np.log(np.clip(p1, 1e-6, 1 - 1e-6) / np.clip(1 - p1, 1e-6, 1))
    F["p1_logit"] = logit.astype(np.float32)
    r, g, b, sz = group_rank_gap(i1, p1)
    F["p1_rank1"] = r; F["p1_gap1"] = g; F["p1_best1"] = b
    F["p1_second1"] = group_second(i1, p1)
    s_sum, c05 = group_sum_count(i1, p1, 0.5); _, c09 = group_sum_count(i1, p1, 0.9)
    F["p1_sum1"] = s_sum; F["p1_n05_1"] = c05; F["p1_n09_1"] = c09
    r2, g2, b2, sz2 = group_rank_gap(i2, p1)
    F["p1_rank2"] = r2; F["p1_gap2"] = g2; F["p1_best2"] = b2
    F["p1_second2"] = group_second(i2, p1)
    F["p1_rel2"] = np.where(b2 > 0, p1 / np.maximum(b2, 1e-6), 0).astype(np.float32)
    # competitor mass on the pool side: sum of other claimants' p1
    s2_sum, _ = group_sum_count(i2, p1, 0.5)
    F["p1_others2"] = (s2_sum - p1).astype(np.float32)
    # per-source confident counts for the S1 (excluding this pair)
    src = Q["src"].to_numpy()[i2]
    conf = (p1 >= anchor_thr).astype(np.float32)
    for s in (2, 3):
        key = i1
        m = (src == s).astype(np.float32) * conf
        tot, _ = group_sum_count(key, m, 2.0)
        F[f"anch_s{s}"] = (tot - m).astype(np.float32)
    # sibling evidence: similarity of the candidate to the S1's confident anchors in the same source
    raw_q = _obj(Q, "raw_name")
    raw_b = np.array([x.lower() for x in raw_q], dtype=object)
    script_b = np.array([has_indic(x) for x in raw_q]); del raw_q
    name_b = _obj(Q, "n_name"); hn_b = _obj(Q, "a_hn"); street_b = _obj(Q, "a_street")
    loc_b = _obj(Q, "a_loc2"); legal_b = _obj(Q, "n_legal")
    order = np.lexsort((-p1, i1))
    # anchors per (i1, src): top-`max_anchors` pairs with p1 >= thr
    anchors: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for k in order:
        if p1[k] < anchor_thr:
            continue
        key = (int(i1[k]), int(src[k]))
        if len(anchors[key]) < max_anchors:
            anchors[key].append(int(i2[k]))
    # sibling comparisons only where they can change a decision (p1 below `sib_p1_cap`); confident pairs keep 0
    pair_idx: List[int] = []; anc_idx: List[int] = []
    for k in np.flatnonzero(p1 < sib_p1_cap):
        lst = anchors.get((int(i1[k]), int(src[k])))
        if not lst:
            continue
        for a in lst:
            if a != i2[k]:
                pair_idx.append(int(k)); anc_idx.append(a)
    sib_raw = np.zeros(n, np.float32); sib_name = np.zeros(n, np.float32); sib_hn = np.zeros(n, np.float32)
    sib_street = np.zeros(n, np.float32); sib_loc = np.zeros(n, np.float32); sib_legal = np.zeros(n, np.float32)
    sib_script = np.zeros(n, np.float32); sib_n = np.zeros(n, np.float32)
    if pair_idx:
        pi = np.asarray(pair_idx); ai = np.asarray(anc_idx); cj = i2[pi]
        s_raw = _sim(raw_b[cj].tolist(), raw_b[ai].tolist(), fuzz.ratio) / 100
        s_name = _sim(name_b[cj].tolist(), name_b[ai].tolist(), fuzz.token_set_ratio) / 100
        s_hn = np.array([1.0 if (hn_b[c] and hn_b[c] == hn_b[a]) else 0.0 for c, a in zip(cj, ai)], np.float32)
        s_street = _sim(street_b[cj].tolist(), street_b[ai].tolist(), fuzz.ratio) / 100
        s_loc = _sim(loc_b[cj].tolist(), loc_b[ai].tolist(), fuzz.token_set_ratio) / 100
        s_legal = np.array([1.0 if legal_b[c] == legal_b[a] else 0.0 for c, a in zip(cj, ai)], np.float32)
        s_script = np.array([1.0 if script_b[c] == script_b[a] else 0.0 for c, a in zip(cj, ai)], np.float32)
        np.maximum.at(sib_raw, pi, s_raw); np.maximum.at(sib_name, pi, s_name); np.maximum.at(sib_hn, pi, s_hn)
        np.maximum.at(sib_street, pi, s_street); np.maximum.at(sib_loc, pi, s_loc); np.maximum.at(sib_legal, pi, s_legal)
        np.maximum.at(sib_script, pi, s_script); np.add.at(sib_n, pi, 1.0)
    F["sib_n"] = sib_n; F["sib_raw"] = sib_raw; F["sib_name"] = sib_name; F["sib_hn"] = sib_hn
    F["sib_street"] = sib_street; F["sib_loc"] = sib_loc; F["sib_legal"] = sib_legal; F["sib_script"] = sib_script
    # competitor sibling evidence: best sibling similarity achieved by any other claimant of the same pool record
    comp = pd.DataFrame({"i2": i2, "v": np.maximum(sib_raw, sib_name)})
    best_any = comp.groupby("i2")["v"].transform("max").to_numpy(np.float32)
    F["sib_comp_best"] = best_any  # includes self; model reads it together with sib_raw/sib_name
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in F.items()})

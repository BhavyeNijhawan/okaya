"""Candidate generation (blocking) for one country.

Channels (all within the country; TF-IDF channels also within the region partition when both
records carry a region code, with global passes for records without one):

  NAME   char 3-gram TF-IDF cosine on the core name                     top-K per S1, top-R per pool record
  COMBO  char 3-gram TF-IDF cosine on core name + house number + street + locality
  ADDR   char 3-gram TF-IDF cosine on house number + street + locality + other numbers
  KEYS   exact joins on high-precision keys: region|house-number|street token, region|house-number|locality
         token, sorted core tokens, concatenated core (domain forms), consonant skeleton, sorted letters
         (shuffled+concatenated domain forms), region|first token|house number

Every candidate pair gets all three cosines (row-wise sparse dot products) plus a bit mask of the
channels that proposed it. Output: DataFrame(i1, i2, s_name, s_combo, s_addr, ch) with i1/i2 row
ids into the country's S1 / pool tables.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from .chargram import CharGramTfidf
from .gpu_topk import Projector, gpu_available, topk_dense

try:  # exact multithreaded top-K sparse matmul (Apache-2.0); pure scipy fallback below
    from sparse_dot_topn import sp_matmul_topn as _sp_matmul_topn
except Exception:  # pragma: no cover
    _sp_matmul_topn = None

CH_NAME, CH_COMBO, CH_ADDR, CH_KEY, CH_REV, CH_GLOBAL = 1, 2, 4, 8, 16, 32


# ---------------------------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------------------------
def name_doc(df: pd.DataFrame) -> List[str]:
    core = df["n_core"].to_numpy()
    name = df["n_name"].to_numpy()
    return [c if c else n for c, n in zip(core, name)]


def combo_doc(df: pd.DataFrame) -> List[str]:
    parts = [name_doc(df), df["a_hn"].to_numpy(), df["a_street"].to_numpy(), df["a_loc2"].to_numpy()]
    return [" ".join(x for x in row if x) for row in zip(*parts)]


def addr_doc(df: pd.DataFrame) -> List[str]:
    parts = [df["a_hn"].to_numpy(), df["a_street"].to_numpy(), df["a_loc2"].to_numpy(), df["a_nums"].to_numpy()]
    return [" ".join(x for x in row if x) for row in zip(*parts)]


class CharTfidf:
    """char 3-gram TF-IDF (vectorised implementation in chargram.py); very common grams dropped."""

    def __init__(self, n=3, max_df=0.03, min_df=2, sample=800_000, seed=0):
        self.vec = CharGramTfidf(n=n, max_df=max_df, min_df=min_df)
        self.sample = sample
        self.seed = seed

    def fit(self, docs: List[str]) -> "CharTfidf":
        self.vec.fit(docs, sample=self.sample, seed=self.seed)
        return self

    def transform(self, docs: List[str]) -> sp.csr_matrix:
        return self.vec.transform(docs)

    @property
    def n_features(self) -> int:
        return int((self.vec.idf > 0).sum()) if self.vec.idf is not None else 0


# ---------------------------------------------------------------------------------------------
# Top-K sparse cosine
# ---------------------------------------------------------------------------------------------
def topk_pairs(A: sp.csr_matrix, B: sp.csr_matrix, k: int, threshold: float, n_threads: int = 2,
               chunk: int = 20_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every row of A return up to k columns of B with cosine >= threshold. Returns (rows, cols, scores)."""
    rows_out, cols_out, vals_out = [], [], []
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    BT = B.T.tocsr() if _sp_matmul_topn is not None else B.T.tocsc()
    for start in range(0, A.shape[0], chunk):
        a = A[start:start + chunk]
        if _sp_matmul_topn is not None:
            c = _sp_matmul_topn(a, BT, top_n=k, threshold=threshold, sort=False, n_threads=n_threads)
        else:
            c = (a @ BT).tocsr()
            c.data[c.data < threshold] = 0
            c.eliminate_zeros()
            # keep top-k per row
            keep_r, keep_c, keep_v = [], [], []
            for r in range(c.shape[0]):
                lo, hi = c.indptr[r], c.indptr[r + 1]
                if hi - lo > k:
                    sel = np.argpartition(c.data[lo:hi], -k)[-k:]
                    keep_r.append(np.full(k, r)); keep_c.append(c.indices[lo:hi][sel]); keep_v.append(c.data[lo:hi][sel])
                elif hi > lo:
                    keep_r.append(np.full(hi - lo, r)); keep_c.append(c.indices[lo:hi]); keep_v.append(c.data[lo:hi])
            if keep_r:
                c = sp.csr_matrix((np.concatenate(keep_v), (np.concatenate(keep_r), np.concatenate(keep_c))), shape=c.shape)
            else:
                c = sp.csr_matrix(c.shape, dtype=np.float32)
        c = c.tocoo()
        rows_out.append(c.row.astype(np.int64) + start)
        cols_out.append(c.col.astype(np.int64))
        vals_out.append(c.data.astype(np.float32))
    return np.concatenate(rows_out), np.concatenate(cols_out), np.concatenate(vals_out)


def rowwise_cosine(X1: sp.csr_matrix, X2: sp.csr_matrix, i1: np.ndarray, i2: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """cos(X1[i1[k]], X2[i2[k]]) for every k (rows are L2-normalised)."""
    out = np.zeros(len(i1), dtype=np.float32)
    for s in range(0, len(i1), chunk):
        a = X1[i1[s:s + chunk]]
        b = X2[i2[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


# ---------------------------------------------------------------------------------------------
# Exact keys
# ---------------------------------------------------------------------------------------------
def _first_tok(s: str) -> str:
    return s.split(" ", 1)[0] if s else ""


def _sorted_letters(s: str) -> str:
    return "".join(sorted(s)) if len(s) >= 8 else ""


def key_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Blocking keys per record: dict name -> list (len n) of '|'-separated key strings ('' = none).

    Multi-valued keys (one per locality / street / core token) are encoded as a single string with
    ';' between alternatives; `exact_key_pairs` explodes them.
    """
    hn = df["a_hn"].to_numpy()
    hn_runs = df["a_hn_runs"].to_numpy()
    reg = df["part"].to_numpy()
    street = df["a_street"].to_numpy()
    loc = df["a_loc2"].to_numpy()
    core2 = df["n_core2"].to_numpy()
    sorted2 = df["n_sorted2"].to_numpy()
    compact = df["n_compact"].to_numpy()
    skel = df["n_skel"].to_numpy()
    dom = df["n_dom"].to_numpy()
    n = len(df)
    k_hn_street, k_hn_loc, k_tok_hn, k_sorted, k_compact, k_skel, k_letters, k_dom = ([""] * n for _ in range(8))
    for i in range(n):
        h = hn[i]
        r = reg[i] or "_"
        if h and (len(h) >= 2 or " " in hn_runs[i]):
            if " " in hn_runs[i]:
                h = hn_runs[i].replace(" ", "-")  # compound numbers: 4-8-139 -> "4-8-139"
            st = street[i].split()[:4]
            if st:
                k_hn_street[i] = ";".join(f"{r}|{h}|{t}" for t in st if len(t) >= 3)
            lc = loc[i].split()[:4]
            if lc:
                k_hn_loc[i] = ";".join(f"{r}|{h}|{t}" for t in lc if len(t) >= 3)
            ct = core2[i].split()[:4]
            if ct:
                k_tok_hn[i] = ";".join(f"{r}|{t}|{h}" for t in ct if len(t) >= 3)
        if len(sorted2[i]) >= 6:
            k_sorted[i] = sorted2[i]
        c = compact[i]
        if len(c) >= 6:
            k_compact[i] = c
            if len(c) >= 8:
                k_letters[i] = "".join(sorted(c))
        if len(skel[i]) >= 6:
            k_skel[i] = skel[i]
        d = dom[i]
        if len(d) >= 5:
            k_dom[i] = d
    # a domain-form name should hit the concatenated core of the other side (and vice versa)
    k_compact_dom = [d if d else c for d, c in zip(k_dom, k_compact)]
    return {"k_hn_street": k_hn_street, "k_hn_loc": k_hn_loc, "k_tok_hn": k_tok_hn, "k_sorted": k_sorted,
            "k_compact": k_compact_dom, "k_skel": k_skel, "k_letters": k_letters}


def _explode(keys: List[str]) -> pd.DataFrame:
    idx, vals = [], []
    for i, k in enumerate(keys):
        if not k:
            continue
        if ";" in k:
            for v in set(k.split(";")):
                idx.append(i); vals.append(v)
        else:
            idx.append(i); vals.append(k)
    return pd.DataFrame({"k": vals, "i": np.asarray(idx, dtype=np.int64)})


def exact_key_pairs(k1: Dict[str, List[str]], k2: Dict[str, List[str]], max_s1: int = 40, max_pool: int = 150,
                    log=print) -> Tuple[np.ndarray, np.ndarray]:
    """Join S1 and pool records on every key; skip over-populated keys (common names / numbers)."""
    pairs: List[np.ndarray] = []
    for name in k1:
        a = _explode(k1[name]).rename(columns={"i": "i1"})
        b = _explode(k2[name]).rename(columns={"i": "i2"})
        if a.empty or b.empty:
            log(f"      key {name}: 0 pairs")
            continue
        ca = a.groupby("k").size(); cb = b.groupby("k").size()
        ok = set(ca[ca <= max_s1].index) & set(cb[cb <= max_pool].index)
        a = a[a.k.isin(ok)]; b = b[b.k.isin(ok)]
        m = a.merge(b, on="k")
        if len(m):
            pairs.append(np.unique(np.stack([m.i1.to_numpy(), m.i2.to_numpy()], axis=1), axis=0))
        log(f"      key {name}: {len(m):,} pairs")
    if not pairs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    P = np.unique(np.concatenate(pairs), axis=0)
    return P[:, 0], P[:, 1]


# ---------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------
class BlockConfig:
    k_name = 8
    k_combo = 8
    k_addr = 4
    k_rev = 2          # per pool record, S1s with known matching partition
    k_rev_unknown = 10  # per pool record without a partition (mostly empty addresses), against all S1
    thr_name = 0.30
    thr_combo = 0.30
    thr_addr = 0.45
    thr_rev = 0.30
    n_threads = 2
    min_part = 2000    # partitions smaller than this are folded into the global pass
    use_gpu = True       # dense random-projection top-K on CUDA when available (see gpu_topk.py)
    gpu_dim = 1024
    gpu_k_extra = 4      # approximate scores: retrieve k + extra per row
    max_df_name = 0.01   # grams in more than this share of documents are dropped (speed; low IDF anyway)
    max_df_addr = 0.01
    ngram = 3


def _accumulate(acc: Dict[str, list], i1, i2, ch):
    acc["i1"].append(np.asarray(i1, np.int64)); acc["i2"].append(np.asarray(i2, np.int64))
    acc["ch"].append(np.full(len(i1), ch, np.int16))


def block_country(s1: pd.DataFrame, pool: pd.DataFrame, cfg: BlockConfig = BlockConfig(), log=print,
                  vectorizers: Optional[Dict[str, CharTfidf]] = None) -> Tuple[pd.DataFrame, Dict[str, CharTfidf]]:
    """Generate candidates for one country. s1/pool must carry the normalized columns and 'part'."""
    t0 = time.time()
    n1, n2 = len(s1), len(pool)
    docs = {"name": (name_doc(s1), name_doc(pool)), "combo": (combo_doc(s1), combo_doc(pool)), "addr": (addr_doc(s1), addr_doc(pool))}
    if vectorizers is None:
        vectorizers = {}
        for key, (d1, d2) in docs.items():
            vectorizers[key] = CharTfidf(n=cfg.ngram, max_df=cfg.max_df_addr if key == "addr" else cfg.max_df_name).fit(d1 + d2)
    X = {}
    for key, (d1, d2) in docs.items():
        X[key] = (vectorizers[key].transform(d1), vectorizers[key].transform(d2))
        log(f"    [{key}] grams {vectorizers[key].n_features:,}, nnz S1 {X[key][0].nnz:,} pool {X[key][1].nnz:,} ({time.time() - t0:.0f}s)")
    del docs

    use_gpu = cfg.use_gpu and gpu_available()
    projectors = {key: Projector(X[key][0].shape[1], D=cfg.gpu_dim, seed=7) for key in X} if use_gpu else {}
    log(f"    retrieval backend: {'GPU dense projection' if use_gpu else 'CPU sparse exact'}")

    def topk(key: str, A, B, k: int, thr: float):
        if use_gpu:
            return topk_dense(A, B, k + cfg.gpu_k_extra, thr - 0.05, projectors[key])
        return topk_pairs(A, B, k, thr, cfg.n_threads)

    acc: Dict[str, list] = {"i1": [], "i2": [], "ch": []}
    part1 = s1["part"].to_numpy(); part2 = pool["part"].to_numpy()
    parts = pd.Series(part1).value_counts()
    big = [p for p, c in parts.items() if p and c >= cfg.min_part]
    small_mask1 = ~np.isin(part1, big)            # unknown or small partitions -> global handling
    small_mask2 = ~np.isin(part2, big)
    idx1_small = np.flatnonzero(small_mask1); idx2_small = np.flatnonzero(small_mask2)
    log(f"    partitions: {len(big)} large; S1 global {len(idx1_small):,}/{n1:,}; pool global {len(idx2_small):,}/{n2:,}")

    def run_forward(i1_idx: np.ndarray, i2_idx: np.ndarray, tag: int):
        for key, k, thr in (("name", cfg.k_name, cfg.thr_name), ("combo", cfg.k_combo, cfg.thr_combo), ("addr", cfg.k_addr, cfg.thr_addr)):
            A = X[key][0][i1_idx]; B = X[key][1][i2_idx]
            r, c, v = topk(key, A, B, k, thr)
            _accumulate(acc, i1_idx[r], i2_idx[c], {"name": CH_NAME, "combo": CH_COMBO, "addr": CH_ADDR}[key] | tag)

    addr_empty = pool["addr_empty"].to_numpy().astype(bool)

    def run_reverse(i2_idx: np.ndarray, i1_idx: np.ndarray, k: int, tag: int):
        for key in ("name", "combo"):
            idx2 = i2_idx if key == "name" else i2_idx[~addr_empty[i2_idx]]  # combo = name for empty addresses
            if len(idx2) == 0:
                continue
            A = X[key][1][idx2]; B = X[key][0][i1_idx]
            kk = k if key == "name" else max(2, k // 2)
            r, c, v = topk(key, A, B, kk, cfg.thr_rev)
            _accumulate(acc, i1_idx[c], idx2[r], CH_REV | tag)

    # 1) large partitions: forward (S1 -> pool) and reverse (pool -> S1) within the partition
    for p in big:
        i1_idx = np.flatnonzero(part1 == p); i2_idx = np.flatnonzero(part2 == p)
        if len(i2_idx) == 0:
            continue
        run_forward(i1_idx, i2_idx, 0)
        run_reverse(i2_idx, i1_idx, cfg.k_rev, 0)
    log(f"    partition passes done ({time.time() - t0:.0f}s)")
    # 2) S1 without (large) partition -> all pool; pool without partition -> all S1 (reverse)
    if len(idx1_small):
        run_forward(idx1_small, np.arange(n2), CH_GLOBAL)
    if len(idx2_small):
        run_reverse(idx2_small, np.arange(n1), cfg.k_rev_unknown, CH_GLOBAL)
    log(f"    global passes done ({time.time() - t0:.0f}s)")
    # 3) exact keys
    k1 = key_columns(s1); k2 = key_columns(pool)
    i1k, i2k = exact_key_pairs(k1, k2, log=log)
    _accumulate(acc, i1k, i2k, CH_KEY)
    log(f"    exact keys done ({time.time() - t0:.0f}s)")

    i1 = np.concatenate(acc["i1"]); i2 = np.concatenate(acc["i2"]); ch = np.concatenate(acc["ch"])
    # unique (i1, i2) with OR-ed channel bits (vectorised)
    key = i1 * np.int64(n2) + i2
    order = np.argsort(key, kind="stable")
    key, ch = key[order], ch[order]
    first = np.ones(len(key), dtype=bool)
    first[1:] = key[1:] != key[:-1]
    starts = np.flatnonzero(first)
    ch_u = np.bitwise_or.reduceat(ch, starts) if len(starts) else ch[:0]
    key_u = key[starts]
    df = pd.DataFrame({"i1": key_u // n2, "i2": key_u % n2, "ch": ch_u.astype(np.int16)})
    # scores for every pair
    for key, col in (("name", "s_name"), ("combo", "s_combo"), ("addr", "s_addr")):
        df[col] = rowwise_cosine(X[key][0], X[key][1], df.i1.to_numpy(), df.i2.to_numpy())
    df["i1"] = df["i1"].astype(np.int32); df["i2"] = df["i2"].astype(np.int32)
    log(f"    candidates: {len(df):,} ({len(df) / max(1, n1):.2f}/S1) ({time.time() - t0:.0f}s)")
    return df, vectorizers

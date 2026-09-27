"""Sibling expansion: a second retrieval pass through already-anchored pool records.

Copies of one business inside one source share base-copy noise, so an unresolved pool record whose
corrupted name barely resembles its S1 may still closely resemble a *sibling* that was retrieved
and confidently matched. We index the names of anchored pool records (pruner score >= anchor_thr
and a clear best claimant), query the unresolved records (best pruner score < unresolved_thr, or
no candidate at all) against that index, and propose the anchors' owners as additional S1
candidates. The matcher then decides; this is candidate generation, not transitive matching.
"""
from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pandas as pd

from . import blocking as B
from .features import group_rank_gap

CH_SIB = 64


def sibling_expansion(s1: pd.DataFrame, pool: pd.DataFrame, C: pd.DataFrame, anchor_thr: float = 0.97,
                      unresolved_thr: float = 0.5, k: int = 3, min_cos: float = 0.5, n_threads: int = 2,
                      log=print, use_gpu: bool = False) -> pd.DataFrame:
    """Return new candidate rows (i1, i2, s_name, s_combo, s_addr, ch) not already in C."""
    t0 = time.time()
    i1 = C["i1"].to_numpy(); i2 = C["i2"].to_numpy(); pp = C["pp"].to_numpy(np.float32)
    n2 = len(pool)
    # best claimant per pool record
    rank2, gap2, best2, size2 = group_rank_gap(i2, pp)
    is_best = rank2 == 0
    # anchors: best claimant with pp >= thr and (single claimant or big gap to the second best)
    df = pd.DataFrame({"i2": i2, "pp": pp, "rank": rank2})
    sec = df[df["rank"] == 1].set_index("i2")["pp"]
    second_of = np.zeros(n2, np.float32)
    second_of[sec.index.to_numpy()] = sec.to_numpy(np.float32)
    anchor_mask = is_best & (pp >= anchor_thr) & (pp - second_of[i2] >= 0.3)
    anchor_pool = i2[anchor_mask]; anchor_owner = i1[anchor_mask]
    # unresolved pool records: no candidate or best pp below threshold
    best_of = np.zeros(n2, np.float32); best_of[i2[is_best]] = pp[is_best]
    has_cand = np.zeros(n2, bool); has_cand[i2] = True
    unresolved = np.flatnonzero((~has_cand) | (best_of < unresolved_thr))
    log(f"    [expand] anchors {len(anchor_pool):,}, unresolved pool records {len(unresolved):,}")
    if len(anchor_pool) == 0 or len(unresolved) == 0:
        return pd.DataFrame(columns=["i1", "i2", "s_name", "s_combo", "s_addr", "ch"])
    # name TF-IDF over pool names (anchors as the index, unresolved as queries)
    docs = B.name_doc(pool)
    vec = B.CharTfidf(max_df=0.03).fit(docs)
    Xp = vec.transform(docs)
    A = Xp[unresolved]; Bm = Xp[anchor_pool]
    if use_gpu and B.gpu_available():
        from .gpu_topk import Projector, topk_dense
        r, c, v = topk_dense(A, Bm, k + 2, min_cos - 0.05, Projector(Xp.shape[1], D=1024, seed=11))
    else:
        r, c, v = B.topk_pairs(A, Bm, k, min_cos, n_threads)
    # same source preferred? keep all; the matcher sees the source. Map to (owner, unresolved record)
    new_i1 = anchor_owner[c]; new_i2 = unresolved[r]
    # exact cosine between the unresolved record and its anchor sibling (kept as s_name of the pair proxy)
    sib_cos = v.astype(np.float32)
    key_new = new_i1.astype(np.int64) * n2 + new_i2
    key_old = i1.astype(np.int64) * n2 + i2
    fresh = ~np.isin(key_new, key_old)
    new_i1, new_i2, sib_cos = new_i1[fresh], new_i2[fresh], sib_cos[fresh]
    # dedupe (owner, record), keep max sibling cosine
    order = np.lexsort((-sib_cos, key_new[fresh]))
    kk = key_new[fresh][order]
    first = np.ones(len(kk), bool); first[1:] = kk[1:] != kk[:-1]
    sel = order[first]
    new_i1, new_i2, sib_cos = new_i1[sel], new_i2[sel], sib_cos[sel]
    # blocking scores for the new pairs (S1 vs pool)
    d1 = {"name": B.name_doc(s1), "combo": B.combo_doc(s1), "addr": B.addr_doc(s1)}
    d2 = {"name": docs, "combo": B.combo_doc(pool), "addr": B.addr_doc(pool)}
    out = pd.DataFrame({"i1": new_i1.astype(np.int32), "i2": new_i2.astype(np.int32)})
    for key, col in (("name", "s_name"), ("combo", "s_combo"), ("addr", "s_addr")):
        vk = B.CharTfidf(max_df=0.03).fit(d1[key] + d2[key])
        X1 = vk.transform([d1[key][a] for a in new_i1.tolist()]); X2 = vk.transform([d2[key][b] for b in new_i2.tolist()])
        out[col] = np.asarray(X1.multiply(X2).sum(axis=1)).ravel().astype(np.float32)
    out["ch"] = np.int16(CH_SIB)
    out["sib_cos"] = sib_cos
    log(f"    [expand] {len(out):,} new candidate pairs ({time.time() - t0:.0f}s)")
    return out

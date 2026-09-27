"""Fast character n-gram TF-IDF (vectorised, no Python-level tokenisation).

Alphabet: a-z, 0-9 and space (37 symbols); every other character is mapped to space. A document is
padded with one space on each side, so 3-grams see word boundaries. Vocabulary = all 37^3 codes
(no hashing collisions). Weights: (1 + log tf) * idf, rows L2-normalised, with very common grams
(df > max_df) dropped. Throughput is tens of millions of characters per second.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import scipy.sparse as sp

_ALPHA = "abcdefghijklmnopqrstuvwxyz0123456789 "
_SPACE = 36
_LUT = np.full(256, _SPACE, dtype=np.int32)  # unknown bytes -> space
for _i, _c in enumerate(_ALPHA):
    _LUT[ord(_c)] = _i


def _encode(docs: List[str], max_len: int):
    """Docs -> ((n, max_len+2) int32 codes, lengths). Space padded (code 36), truncated at max_len.
    Vectorised: one bytes buffer for all docs, one frombuffer, one scatter."""
    n = len(docs)
    arr = np.full((n, max_len + 2), _SPACE, dtype=np.int32)
    if n == 0:
        return arr, np.zeros(0, dtype=np.int32)
    b = [d.encode("ascii", "replace")[:max_len] for d in docs]
    lengths = np.fromiter((len(x) for x in b), dtype=np.int32, count=n)
    buf = np.frombuffer(b"".join(b), dtype=np.uint8)
    if len(buf):
        rows = np.repeat(np.arange(n, dtype=np.int64), lengths)
        starts = np.cumsum(lengths) - lengths
        cols = (np.arange(len(buf), dtype=np.int64) - np.repeat(starts, lengths)) + 1
        arr[rows, cols] = _LUT[buf]
    return arr, lengths


class CharGramTfidf:
    def __init__(self, n: int = 3, max_df: float = 0.03, min_df: int = 2, max_len: int = 96):
        assert n in (2, 3, 4)
        self.n = n
        self.max_df = max_df
        self.min_df = min_df
        self.max_len = max_len
        self.V = 37 ** n
        self.idf: Optional[np.ndarray] = None

    def _counts(self, docs: List[str], chunk: int = 200_000) -> sp.csr_matrix:
        mats = []
        for s in range(0, len(docs), chunk):
            arr, lengths = _encode(docs[s:s + chunk], self.max_len)
            m, L = arr.shape
            # positions of n-grams: 0 .. L-n
            codes = np.zeros((m, L - self.n + 1), dtype=np.int64)
            for k in range(self.n):
                codes = codes * 37 + arr[:, k:L - self.n + 1 + k]
            # valid n-grams: those that overlap the document (start < len+1) and are not all-space
            pos = np.arange(L - self.n + 1)[None, :]
            valid = pos < (lengths[:, None] + 3 - self.n)  # grams inside " " + doc + " " (one boundary space each side)
            allspace = codes == (37 ** self.n - 1)  # code of "   "
            valid &= ~allspace
            rows = np.repeat(np.arange(m), L - self.n + 1).reshape(m, -1)[valid]
            cols = codes[valid]
            mat = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(m, self.V)).tocsr()
            mat.sum_duplicates()
            mats.append(mat)
        return sp.vstack(mats).tocsr() if len(mats) > 1 else mats[0]

    def fit(self, docs: List[str], sample: int = 800_000, seed: int = 0) -> "CharGramTfidf":
        if len(docs) > sample:
            rng = np.random.default_rng(seed)
            docs = [docs[i] for i in rng.choice(len(docs), sample, replace=False)]
        m = self._counts(docs)
        df = np.asarray((m > 0).sum(axis=0)).ravel().astype(np.float64)
        N = m.shape[0]
        idf = np.log((1.0 + N) / (1.0 + df)) + 1.0
        idf[df < self.min_df] = 0.0
        idf[df > self.max_df * N] = 0.0
        self.idf = idf.astype(np.float32)
        return self

    def transform(self, docs: List[str]) -> sp.csr_matrix:
        assert self.idf is not None
        m = self._counts(docs)
        m.data = (1.0 + np.log(m.data)).astype(np.float32)
        m = m @ sp.diags(self.idf, format="csr", dtype=np.float32)
        m = m.tocsr()
        m.eliminate_zeros()
        norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        m = sp.diags((1.0 / norms).astype(np.float32), format="csr") @ m
        m = m.tocsr()
        m.sort_indices()
        return m.astype(np.float32)

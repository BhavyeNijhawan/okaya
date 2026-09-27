"""Learned token-level transliteration for Indic-script names (from the training ground truth only).

anyascii gives a rough phonetic Latin form (सिल्वर टेक -> "silvr tek"). For frequent words the
challenge data itself tells us the intended Latin spelling: in true pairs where the pool name is in an
Indic script and has the same number of tokens as the S1 core name, tokens align position-wise
(word order is rarely shuffled in script copies). We count (anyascii token -> S1 token) and keep
consistent mappings. Unknown tokens keep their anyascii form (and are still compared through the
consonant skeleton and character n-grams).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .normalize import LEGAL_TOKENS

_INDIC_RE = re.compile(r"[ऀ-෿]")


def has_indic(s: str) -> bool:
    return bool(_INDIC_RE.search(s))


class TranslitDict:
    def __init__(self, mapping: Dict[str, str] | None = None):
        self.map: Dict[str, str] = dict(mapping or {})

    def learn(self, pool_raw_names: Iterable[str], pool_norm_names: Iterable[str], s1_norm_names: Iterable[str],
              min_count: int = 3, min_purity: float = 0.6) -> int:
        """pool_norm_names / s1_norm_names are the normalized (`n_name`) forms of true pairs."""
        cnt: Dict[str, Counter] = defaultdict(Counter)
        for raw, pn, sn in zip(pool_raw_names, pool_norm_names, s1_norm_names):
            if not has_indic(raw):
                continue
            a, b = pn.split(), sn.split()
            if len(a) != len(b) or not a:
                continue
            for x, y in zip(a, b):
                if x == y or not x or not y:
                    continue
                if x.isascii() and any(ch.isdigit() for ch in x):
                    continue
                cnt[x][y] += 1
        added = 0
        for x, ctr in cnt.items():
            tot = sum(ctr.values())
            if tot < min_count:
                continue
            y, n = ctr.most_common(1)[0]
            if n / tot >= min_purity and y != x:
                self.map[x] = y
                added += 1
        return added

    def apply(self, norm_name: str) -> str:
        if not self.map:
            return norm_name
        toks = norm_name.split()
        out = [self.map.get(t, t) for t in toks]
        return " ".join(out)

    def apply_many(self, raw_names: Iterable[str], norm_names: Iterable[str]) -> List[str]:
        """Apply only to names that contain Indic script (others returned unchanged)."""
        return [self.apply(n) if has_indic(r) else n for r, n in zip(raw_names, norm_names)]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.map, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "TranslitDict":
        return cls(json.loads(path.read_text(encoding="utf-8")))


def retranslit_frame(df, td: TranslitDict) -> None:
    """Rewrite n_name/n_core/n_core2/n_sorted2/n_compact/n_skel/n_first for Indic-script records in place.

    Works column by column through Python lists (one transient copy at a time) so it also fits
    Arrow-backed frames of several million rows.
    """
    from .normalize import GENERIC_TOKENS, consonant_skeleton
    raw = df["raw_name"].tolist()
    idx = [i for i, r in enumerate(raw) if isinstance(r, str) and has_indic(r)]
    del raw
    if not idx:
        return
    names = df["n_name"].tolist()
    new = {"n_name": {}, "n_core": {}, "n_core2": {}, "n_sorted2": {}, "n_compact": {}, "n_skel": {}, "n_first": {}}
    for i in idx:
        name = td.apply(names[i] or "")
        toks = name.split()
        core = [t for t in toks if t not in LEGAL_TOKENS] or toks
        core2 = [t for t in core if t not in GENERIC_TOKENS] or core
        new["n_name"][i] = name
        new["n_core"][i] = " ".join(core)
        new["n_core2"][i] = " ".join(core2)
        new["n_sorted2"][i] = " ".join(sorted(core2))
        new["n_compact"][i] = "".join(core2)
        new["n_skel"][i] = " ".join(s for s in (consonant_skeleton(t) for t in core2) if s)
        new["n_first"][i] = core2[0] if core2 else ""
    del names
    dtype = df["n_name"].dtype
    for col, repl in new.items():
        vals = df[col].tolist()
        for i, v in repl.items():
            vals[i] = v
        df[col] = pd.array(vals, dtype=dtype) if str(dtype).startswith("string") or "pyarrow" in str(dtype) else np.array(vals, dtype=object)
        del vals

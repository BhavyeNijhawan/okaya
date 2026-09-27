"""Region (state / région) codes for records, learned alias tables and blocking partitions.

Why: for US records the state, when present on both sides, agrees on 100% of true pairs, and the
Indian/French data behave the same way up to renamings (Telangana/Andhra Pradesh) and
department-vs-region substitutions (Nord vs Hauts-de-France). A region code therefore gives a
lossless blocking partition and a strong pairwise feature -- provided the many spellings
(abbreviations, Indic-script transliterations, departments, cities without state) are mapped.

Seeds are ordinary normalization tables (US states, Indian states/UTs, French regions and
departments). Everything else is *learned from the challenge data*:
  * supervised: components of pool records in ground-truth pairs vs. the S1 region (train);
  * unsupervised: components co-occurring with a known region inside the same record (any split,
    including the France test records).
Codes that the labels show to be interchangeable (e.g. Telangana <-> Andhra Pradesh) are merged
into one partition with union-find.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .normalize import to_ascii

_KEY_RE = re.compile(r"[^a-z0-9]+")


def comp_key(c: str) -> str:
    return _KEY_RE.sub(" ", to_ascii(c)).strip()


# ---------------------------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------------------------
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
INDIA_STATES = {
    "ap": ["andhra pradesh", "andhrapradesh"], "ar": ["arunachal pradesh"], "as": ["assam"], "br": ["bihar"],
    "cg": ["chhattisgarh", "chattisgarh", "ct"], "ga": ["goa"], "gj": ["gujarat", "gujrat"], "hr": ["haryana"],
    "hp": ["himachal pradesh"], "jh": ["jharkhand"], "ka": ["karnataka"], "kl": ["kerala", "keralam"],
    "mp": ["madhya pradesh", "madhyapradesh"], "mh": ["maharashtra", "maharastra"], "mn": ["manipur"],
    "ml": ["meghalaya"], "mz": ["mizoram"], "nl": ["nagaland"], "od": ["odisha", "orissa", "or"], "pb": ["punjab"],
    "rj": ["rajasthan"], "sk": ["sikkim"], "tn": ["tamil nadu", "tamilnadu"], "ts": ["telangana", "tg", "telengana"],
    "tr": ["tripura"], "up": ["uttar pradesh", "uttarpradesh"], "uk": ["uttarakhand", "uttaranchal", "ua"],
    "wb": ["west bengal", "westbengal"], "dl": ["delhi", "new delhi", "nct of delhi", "national capital territory of delhi"],
    "jk": ["jammu and kashmir", "jammu kashmir", "j k"], "la": ["ladakh"], "ch": ["chandigarh"],
    "py": ["puducherry", "pondicherry"], "an": ["andaman and nicobar islands", "andaman nicobar"],
    "dn": ["dadra and nagar haveli", "daman and diu", "dadra and nagar haveli and daman and diu"], "ld": ["lakshadweep"],
}
FRANCE_REGIONS = {
    "ara": ["auvergne rhone alpes", "ain", "allier", "ardeche", "cantal", "drome", "isere", "loire", "haute loire",
            "puy de dome", "rhone", "savoie", "haute savoie"],
    "bfc": ["bourgogne franche comte", "cote d or", "doubs", "jura", "nievre", "haute saone", "saone et loire", "yonne",
            "territoire de belfort"],
    "bre": ["bretagne", "cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "cvl": ["centre val de loire", "cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"],
    "cor": ["corse", "corse du sud", "haute corse"],
    "ges": ["grand est", "ardennes", "aube", "marne", "haute marne", "meurthe et moselle", "meuse", "moselle", "bas rhin",
            "haut rhin", "vosges", "alsace", "lorraine", "champagne ardenne"],
    "hdf": ["hauts de france", "aisne", "nord", "oise", "pas de calais", "somme", "nord pas de calais", "picardie"],
    "idf": ["ile de france", "paris", "seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis",
            "val de marne", "val d oise"],
    "nor": ["normandie", "calvados", "eure", "manche", "orne", "seine maritime", "basse normandie", "haute normandie"],
    "naq": ["nouvelle aquitaine", "charente", "charente maritime", "correze", "creuse", "dordogne", "gironde", "landes",
            "lot et garonne", "pyrenees atlantiques", "deux sevres", "vienne", "haute vienne", "aquitaine", "limousin",
            "poitou charentes"],
    "occ": ["occitanie", "ariege", "aude", "aveyron", "gard", "haute garonne", "gers", "herault", "lot", "lozere",
            "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne", "languedoc roussillon", "midi pyrenees"],
    "pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "pac": ["provence alpes cote d azur", "paca", "alpes de haute provence", "hautes alpes", "alpes maritimes",
            "bouches du rhone", "var", "vaucluse"],
}


def seed_aliases(country: str) -> Dict[str, str]:
    """alias key (comp_key form) -> region code, for one country."""
    al: Dict[str, str] = {}
    if country == "us":
        for code, name in US_STATES.items():
            al[code] = code
            al[name] = code
    elif country == "india":
        for code, names in INDIA_STATES.items():
            al[code] = code
            for n in names:
                al[comp_key(n)] = code
    elif country == "france":
        for code, names in FRANCE_REGIONS.items():
            for n in names:
                al[comp_key(n)] = code
    return al


# ---------------------------------------------------------------------------------------------
# Union-find for interchangeable codes
# ---------------------------------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------------------------------------
# Region tables
# ---------------------------------------------------------------------------------------------
class RegionTable:
    """Per-country alias dictionary + code equivalences. Serialisable to JSON."""

    def __init__(self, country: str, aliases: Optional[Dict[str, str]] = None, merges: Optional[List[Tuple[str, str]]] = None):
        self.country = country
        self.aliases: Dict[str, str] = dict(aliases) if aliases is not None else seed_aliases(country)
        self.seed_keys = set(seed_aliases(country))
        self.uf = UnionFind()
        for a, b in (merges or []):
            self.uf.union(a, b)
        self.merges: List[Tuple[str, str]] = list(merges or [])
        self._token_aliases = {k: v for k, v in self.aliases.items() if " " not in k and len(k) >= 4}

    # -- assignment -------------------------------------------------------------------------
    def code_of_comps(self, comps: List[str]) -> Tuple[str, int]:
        """Return (raw code, index of the component that carried it) or ('', -1).

        Priority: seed 2-letter codes > seed full names > learned aliases > single-token aliases.
        A record whose components map to two different seed *partitions* (e.g. "Nevada, TX",
        "Punjab, Chandigarh") is ambiguous and gets '' (it is then matched without a partition).
        """
        keys = [comp_key(c) for c in comps]
        best: Tuple[int, int, str] | None = None  # (priority, -index, code)
        seed_parts = set()
        for i in range(len(keys) - 1, -1, -1):
            k = keys[i]
            code = self.aliases.get(k)
            if code is None:
                continue
            if k in self.seed_keys:
                pr = 0 if len(k) == 2 else 1
                seed_parts.add(self.uf.find(code))
            else:
                pr = 2
            cand = (pr, -i, code)
            if best is None or cand < best:
                best = cand
        if len(seed_parts) >= 2:
            return "", -1
        if best is not None:
            return best[2], -best[1]
        for i in range(len(keys) - 1, -1, -1):
            for t in keys[i].split():
                if t in self._token_aliases:
                    return self._token_aliases[t], i
        return "", -1

    def partition(self, code: str) -> str:
        return self.uf.find(code) if code else ""

    def assign(self, comps_joined: Iterable[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised over '|'-joined component strings -> (code, partition, component index)."""
        codes, parts, idx = [], [], []
        cache: Dict[str, Tuple[str, str, int]] = {}
        for s in comps_joined:
            r = cache.get(s)
            if r is None:
                comps = s.split("|") if s else []
                c, i = self.code_of_comps(comps)
                r = (c, self.partition(c), i)
                if len(cache) < 3_000_000:
                    cache[s] = r
            codes.append(r[0]); parts.append(r[1]); idx.append(r[2])
        return np.asarray(codes, dtype=object), np.asarray(parts, dtype=object), np.asarray(idx, dtype=np.int16)

    # -- learning -----------------------------------------------------------------------------
    def learn_unsupervised(self, comps_joined: Iterable[str], min_count: int = 30, purity: float = 0.95) -> int:
        """Components co-occurring (inside one record) with a known region become aliases of it."""
        cnt: Dict[str, Counter] = defaultdict(Counter)
        for s in comps_joined:
            if not s:
                continue
            comps = s.split("|")
            code, i = self.code_of_comps(comps)
            if not code:
                continue
            for j, c in enumerate(comps):
                if j == i:
                    continue
                k = comp_key(c)
                if not k or k in self.aliases or any(ch.isdigit() for ch in k) or len(k) < 3:
                    continue
                cnt[k][self.partition(code)] += 1
        return self._absorb(cnt, min_count, purity)

    def learn_supervised(self, s1_comps: Iterable[str], pool_comps: Iterable[str], min_count: int = 20,
                         purity: float = 0.9, merge_frac: float = 0.2) -> int:
        """Pool components of true pairs vs. the S1 region. Also merges codes the labels confuse."""
        cnt: Dict[str, Counter] = defaultdict(Counter)
        seed_conf: Dict[str, Counter] = defaultdict(Counter)
        for s_a, s_b in zip(s1_comps, pool_comps):
            if not s_a or not s_b:
                continue
            code_a, _ = self.code_of_comps(s_a.split("|"))
            if not code_a:
                continue
            pa = self.partition(code_a)
            comps_b = s_b.split("|")
            code_b, ib = self.code_of_comps(comps_b)
            if code_b:
                seed_conf[self.partition(code_b)][pa] += 1
            for j, c in enumerate(comps_b):
                k = comp_key(c)
                if not k or k in self.aliases or any(ch.isdigit() for ch in k) or len(k) < 2:
                    continue
                cnt[k][pa] += 1
        # merge codes whose pool-side occurrences are frequently labelled with another S1 code
        for pb, ctr in seed_conf.items():
            tot = sum(ctr.values())
            for pa, n in ctr.items():
                if pa != pb and tot >= 200 and n / tot >= merge_frac:
                    self.uf.union(pa, pb)
                    self.merges.append((pa, pb))
        return self._absorb(cnt, min_count, purity)

    def _absorb(self, cnt: Dict[str, Counter], min_count: int, purity: float) -> int:
        added = 0
        for k, ctr in cnt.items():
            tot = sum(ctr.values())
            if tot < min_count:
                continue
            code, n = ctr.most_common(1)[0]
            if n / tot >= purity:
                self.aliases[k] = code
                added += 1
        self._token_aliases = {k: v for k, v in self.aliases.items() if " " not in k and len(k) >= 4}
        return added

    # -- persistence ---------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {"country": self.country, "aliases": self.aliases, "merges": self.merges}

    @classmethod
    def from_json(cls, d: dict) -> "RegionTable":
        return cls(d["country"], d["aliases"], [tuple(m) for m in d["merges"]])


def save_tables(tables: Dict[str, RegionTable], path: Path) -> None:
    path.write_text(json.dumps({c: t.to_json() for c, t in tables.items()}, ensure_ascii=False))


def load_tables(path: Path) -> Dict[str, RegionTable]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {c: RegionTable.from_json(v) for c, v in d.items()}


def region_token_set(table: RegionTable) -> set:
    """Tokens of *seed* aliases (state names / codes) -- removed from locality tokens for comparisons."""
    toks = set()
    for k in table.seed_keys:
        toks.update(k.split())
    return toks


def strip_region_tokens(loc: str, comp_idx: int, comps_joined: str, table: RegionTable) -> str:
    """Locality tokens without the tokens of the component that carried the region code."""
    if comp_idx < 0 or not loc:
        return loc
    comps = comps_joined.split("|")
    if comp_idx >= len(comps):
        return loc
    drop = set(comp_key(comps[comp_idx]).split())
    return " ".join(t for t in loc.split() if t not in drop)


def add_region_columns(df: pd.DataFrame, table: RegionTable) -> pd.DataFrame:
    codes, parts, idx = table.assign(df["a_comps"].tolist())
    df["region"] = codes
    df["part"] = parts
    df["a_loc2"] = [strip_region_tokens(l, i, c, table) for l, i, c in zip(df["a_loc"].tolist(), idx.tolist(), df["a_comps"].tolist())]
    return df

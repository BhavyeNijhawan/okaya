"""Record-level normalization and light parsing of business names and addresses.

Everything here is deterministic and uses only the record text itself (no external data).
The normalization is deliberately *lossless where it matters*: several parallel views of a
name/address are produced (full, core, sorted core, compact, consonant skeleton, digit runs, ...)
so that the pairwise feature layer can measure agreement under each of the noise transforms seen
in the data (case, legal-form variants, token shuffles, duplicated/dropped words, l33t digits,
random accents, Indic-script transliteration, domain-name forms, alias/DBA forms, decorations,
address abbreviations, component reordering/dropping, house-number noise).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Tuple

from anyascii import anyascii

# ----------------------------------------------------------------------------------------------
# Character level
# ----------------------------------------------------------------------------------------------
# Indic anusvara / nasal marks -> "n" before transliteration (मार्केटिंग -> marketing, not marketimg)
_NASAL = str.maketrans({c: "n" for c in "ंંংਂஂంಂംଂ"})
# Danda / typographic quotes / dashes
_TYPO = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "‐": "-", "।": ".", "|": " | "})


def to_ascii(text: str) -> str:
    """NFKC + transliteration of any non-ASCII text (accents, Indic scripts) to lowercase ASCII."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text).translate(_TYPO)
    if not text.isascii():
        text = anyascii(text.translate(_NASAL))
    return text.lower()


# l33t digits inside otherwise alphabetic tokens: 5olutions, 6alaxy, M0unt, Lifec0, c0m, Nati0nal
_LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g", "$": "s", "@": "a"}
_LEET_TOKEN_RE = re.compile(r"^(?=.*[a-z]{2})(?=.*[0-9$@])[a-z0-9$@]{3,}$")


def unleet_token(tok: str) -> str:
    """Decode l33t digits when a token is mostly letters with at most two digit-like chars."""
    if not _LEET_TOKEN_RE.match(tok):
        return tok
    n_digits = sum(ch in _LEET for ch in tok)
    if n_digits > 2 or n_digits >= len(tok) - 1:
        return tok
    # do not touch pure alphanumeric codes like "b3" or ordinals like "12th"
    if re.match(r"^\d+(st|nd|rd|th)$", tok) or re.match(r"^[a-z]\d+$", tok) or re.match(r"^\d+[a-z]$", tok):
        return tok
    return "".join(_LEET.get(ch, ch) for ch in tok)


# ----------------------------------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------------------------------
# Alias markers: "<fake> formerly known as <real>", "<fake> dba <real>", "<fake> t/a <real>", "<fake> nee <real>"
_ALIAS_RE = re.compile(
    r"\s(?:d\.?\s?b\.?\s?a\.?|t/a|t\.a\.|trading as|doing business as|formerly known as|formerly:|formerly|"
    r"f/k/a|fka|a\.?k\.?a\.?|also known as|now known as|n[eé]e|operating as|o/a)\s+",
    flags=re.IGNORECASE,
)
# "(ID: 8283)", "- 4317145536", trailing phone-like numbers
_ID_RE = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|\bid\s*:\s*\d+|(?:^|\s)-\s*\d{6,}\b")
# a URL / domain token, possibly with l33t 'c0m'
_DOMAIN_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9\-]*)\.(?:c[o0]m|net|[o0]rg|in|co\.in|fr|io|biz|inf[o0]|us|co|eu|de|uk)$"
)
_HANDLE_RE = re.compile(r"^[#@]([a-z0-9][a-z0-9_\-]*)$")

# canonical legal forms: one alternation regex, longest phrases first; the matched text is
# normalised (dots/spaces removed) and looked up in _LEGAL_CANON.
_LEGAL_VARIANTS = {
    "pvtltd": ["private limited", "pvt limited", "pvt. limited", "private ltd", "private ltd.", "pvt ltd", "pvt. ltd.",
               "pvt.ltd.", "pvt.ltd", "pvt ltd.", "pvt. ltd", "p ltd", "(p) ltd", "p. ltd", "pra li", "pra. li.", "pra.li.",
               "pvt. limited", "private limited."],
    "plc": ["public limited company", "public limited", "plc", "p.l.c."],
    "llp": ["limited liability partnership", "llp", "l.l.p.", "l.l.p"],
    "llc": ["limited liability company", "llc", "l.l.c.", "l.l.c", "l l c"],
    "pllc": ["pllc", "p.l.l.c."],
    "lp": ["lp", "l.p.", "l.p"],
    "pc": ["pc", "p.c.", "p.c"],
    "eurl": ["eurl", "e.u.r.l.", "e.u.r.l"],
    "sarl": ["sarl", "s.a.r.l.", "s.a.r.l"],
    "sasu": ["sasu", "s.a.s.u.", "s.a.s.u"],
    "sas": ["sas", "s.a.s.", "s.a.s"],
    "sci": ["sci", "s.c.i.", "s.c.i"],
    "snc": ["snc", "s.n.c.", "s.n.c"],
    "sa": ["sa", "s.a.", "s.a"],
    "inc": ["incorporated", "inc", "inc."],
    "corp": ["corporation", "corp", "corp.", "corpn", "corpn."],
    "co": ["company", "co", "co."],
    "ltd": ["limited", "ltd", "ltd.", "ltda", "ltda."],
    "pvt": ["private", "pvt", "pvt.", "pvte", "pvte."],
    "gmbh": ["gmbh", "g.m.b.h."],
    "ets": ["etablissements", "etablissement", "ets", "ets."],
}
_LEGAL_CANON: Dict[str, str] = {}
for _canon, _vs in _LEGAL_VARIANTS.items():
    for _v in _vs:
        _LEGAL_CANON[re.sub(r"[ .()]", "", _v)] = _canon
_LEGAL_ALL = sorted({v for vs in _LEGAL_VARIANTS.values() for v in vs}, key=len, reverse=True)
_LEGAL_RE = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(v) for v in _LEGAL_ALL) + r")(?![a-z0-9])")


def _legal_sub(m: "re.Match") -> str:
    return _LEGAL_CANON.get(re.sub(r"[ .()]", "", m.group(0)), m.group(0))


LEGAL_TOKENS = {
    "pvtltd", "ltd", "pvt", "llc", "inc", "corp", "co", "lp", "llp", "plc", "pc", "pllc",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "snc", "ei", "gmbh", "ag", "bv", "nv", "pty", "ets",
}
# words the generator appends / prepends as noise, plus stopwords and honorifics (dropped in core2)
GENERIC_TOKENS = {
    "the", "and", "of", "&", "center", "centre", "services", "service", "partners", "group", "groupe",
    "sri", "shri", "shree", "smt", "mr", "mrs", "ms", "m/s", "ms.", "dr", "messrs", "ets",
    "le", "la", "les", "de", "du", "des", "et", "en", "au", "aux",
    "international", "india", "france", "usa", "us",
}
STOP_TOKENS = {"the", "and", "of", "le", "la", "les", "de", "du", "des", "et", "&"}
_HONORIFIC_RE = re.compile(r"^(?:m/s\.?|messrs\.?|mr\.?|mrs\.?|ms\.?|smt\.?|sri|shri|shree|dr\.?)\s+")
_PUNCT_KEEP_RE = re.compile(r"[^a-z0-9&'/+.\- ]+")
_WS_RE = re.compile(r"\s+")

_SKEL_RULES = [("tion", "sn"), ("sion", "sn"), ("ph", "f"), ("x", "ks"), ("q", "k"), ("ck", "k"),
               ("sh", "s"), ("ch", "k"), ("th", "t"), ("kh", "k"), ("bh", "b"), ("dh", "d"), ("gh", "j"),
               ("g", "j"), ("z", "s"), ("w", "v")]
_SOFT_C = re.compile(r"c(?=[eiy])")
_VOWELS_RE = re.compile(r"[aeiouyh]")


def consonant_skeleton(tok: str) -> str:
    """Vowel-free skeleton aligning transliteration re-spellings (silvr/silver -> slvr, tek/tech -> tk)."""
    if tok.isdigit() or len(tok) < 3:
        return tok
    tok = _SOFT_C.sub("s", tok)
    for a, b in _SKEL_RULES:
        tok = tok.replace(a, b)
    tok = _VOWELS_RE.sub("", tok.replace("c", "k"))
    out = []
    for ch in tok:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


def _dedupe_consecutive(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        if not out or out[-1] != t:
            out.append(t)
    return out


def _clean_name_text(text: str) -> str:
    """ascii, lower, decorations removed, legal phrases canonicalised; returns space-joined tokens."""
    text = to_ascii(text)
    text = _ID_RE.sub(" ", text)
    text = text.replace("&", " and ").replace("+", " and ")
    text = text.replace("'", "").replace('"', " ")
    text = re.sub(r"[\[\]\(\)\{\}<>*!?:;,_~^=\"]+", " ", text)  # brackets/decorations -> space (content kept)
    text = re.sub(r"(?<=[a-z])\.(?=[a-z])", ".", text)  # keep dotted abbreviations for legal regexes
    text = text.replace("/", " ").replace("-", " ")
    text = _WS_RE.sub(" ", text).strip()
    text = _HONORIFIC_RE.sub("", text)
    text = _LEGAL_RE.sub(_legal_sub, text)
    text = text.replace(".", " ")
    toks = [unleet_token(t) for t in text.split()]
    toks = [t for t in toks if t and t != "|"]
    return " ".join(_dedupe_consecutive(toks))


def parse_name(raw: str) -> Dict[str, str]:
    """Return the parallel views of a business name.

    Keys: name (primary, alias-resolved), name_alt (the other alias side or ''), core (legal forms
    removed), core2 (legal+generic removed), sorted2 (core2 tokens sorted), compact (core2 without
    spaces), skel (consonant skeleton of core2), legal (space-joined canonical legal tokens),
    dom (domain/handle core letters or ''), first (first core2 token), ntok (token count of core2).
    """
    if raw is None:
        raw = ""
    raw = str(raw)
    low = to_ascii(raw)
    # URL appended after a pipe ("Peridance Alchemy L.L.C. | www.peridance.com"), domain-only names
    # ("hrsinvestments.com", "hdiner.c0m") and handles ("#alhigh", "@lynx") -> dom = concatenated core
    dom = ""
    kept_tokens: List[str] = []
    for tok in low.replace("|", " ").split():
        t = tok.strip(",;:!?()[]{}<>*\"'")
        m = _DOMAIN_RE.match(t) or _HANDLE_RE.match(t)
        if m:
            d = m.group(1).replace("-", "").replace("_", "")
            if any(ch in "0135" for ch in d) and re.search(r"[a-z]", d):
                d = "".join(_LEET.get(ch, ch) if ch in "0135" else ch for ch in d)
            if not dom:
                dom = d
            if _HANDLE_RE.match(t) or len(kept_tokens) == 0 and len(low.split()) <= 2:
                kept_tokens.append(d)  # handle / bare domain: the core letters stand in for the name
            continue
        kept_tokens.append(tok)
    text = " ".join(kept_tokens)
    # alias forms: "<fake> formerly known as <real>" -> primary = real (after marker)
    alias_alt = ""
    m = _ALIAS_RE.search(" " + text + " ")
    if m:
        before = text[: max(0, m.start() - 1)].strip()
        after = text[m.end() - 1:].strip()
        if after:
            text, alias_alt = after, before
        else:
            text = before
    name = _clean_name_text(text)
    alt = _clean_name_text(alias_alt) if alias_alt else ""
    if not name and dom:
        name = dom
    toks = name.split()
    legal = [t for t in toks if t in LEGAL_TOKENS]
    core = [t for t in toks if t not in LEGAL_TOKENS]
    core2 = [t for t in core if t not in GENERIC_TOKENS]
    if not core2:
        core2 = core if core else toks
    if not core:
        core = toks
    compact = "".join(core2)
    return {
        "name": name,
        "name_alt": alt,
        "core": " ".join(core),
        "core2": " ".join(core2),
        "sorted2": " ".join(sorted(core2)),
        "compact": compact,
        "skel": " ".join(s for s in (consonant_skeleton(t) for t in core2) if s),
        "legal": " ".join(sorted(set(legal))),
        "dom": dom,
        "first": core2[0] if core2 else "",
        "ntok": len(core2),
    }


# ----------------------------------------------------------------------------------------------
# Addresses
# ----------------------------------------------------------------------------------------------
STREET_TYPES = {
    # US
    "street": "st", "str": "st", "st": "st", "saint": "st", "ste.": "ste",
    "road": "rd", "rd": "rd", "roda": "rd", "rod": "rd",
    "avenue": "ave", "av": "ave", "ave": "ave", "aven": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd",
    "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct", "cour": "ct",
    "place": "pl", "pl": "pl", "square": "sq", "sq": "sq", "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "circle": "cir", "cir": "cir", "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl", "point": "pt", "pt": "pt", "way": "way", "loop": "loop", "run": "run",
    "cove": "cv", "cv": "cv", "bend": "bnd", "ridge": "rdg", "rdg": "rdg", "hill": "hl", "hl": "hl",
    "hills": "hls", "path": "path", "pike": "pike", "turnpike": "tpke", "tpke": "tpke", "plaza": "plz", "plz": "plz",
    "route": "rte", "rte": "rte", "rt": "rte", "expressway": "expy", "expy": "expy", "freeway": "fwy", "fwy": "fwy",
    "crossing": "xing", "xing": "xing", "creek": "crk", "crk": "crk", "heights": "hts", "hts": "hts",
    "alley": "aly", "aly": "aly", "walk": "walk", "row": "row", "bridge": "brg", "brg": "brg", "bluff": "blf",
    "blf": "blf", "glen": "gln", "gln": "gln", "grove": "grv", "grv": "grv", "harbor": "hbr", "hbr": "hbr",
    "island": "is", "junction": "jct", "jct": "jct", "landing": "lndg", "lndg": "lndg", "meadows": "mdws",
    "mdws": "mdws", "park": "park", "pass": "pass", "shore": "shr", "shr": "shr", "spring": "spg", "spg": "spg",
    "springs": "spgs", "spgs": "spgs", "station": "sta", "sta": "sta", "valley": "vly", "vly": "vly",
    "view": "vw", "vw": "vw", "village": "vlg", "vlg": "vlg", "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
    # India
    "marg": "marg", "mg": "marg", "nagar": "nagar", "ngr": "nagar", "colony": "colony", "col": "colony",
    "layout": "layout", "cross": "cross", "crs": "cross", "main": "main", "mn": "main", "sector": "sector",
    "sec": "sector", "phase": "phase", "ph": "phase", "stage": "stage", "stg": "stage", "block": "block",
    "blk": "block", "gali": "gali", "chowk": "chowk", "chk": "chowk", "bazar": "bazar", "bazaar": "bazar",
    "mandi": "mandi", "peth": "peth", "society": "society", "soc": "society", "estate": "estate", "est": "estate",
    "industrial": "indl", "indl": "indl", "ind": "indl", "area": "area", "village": "village", "vill": "village",
    "taluka": "taluka", "taluk": "taluka", "tal": "taluka", "tehsil": "tehsil", "teh": "tehsil", "mandal": "mandal",
    "district": "dist", "dist": "dist", "post": "po", "p.o": "po", "po": "po", "near": "near", "nr": "near",
    "opposite": "opp", "opp": "opp", "behind": "behind", "b/h": "behind", "bh": "behind", "beside": "beside",
    "floor": "fl", "flr": "fl", "fl": "fl", "ground": "gr", "grnd": "gr", "gr": "gr",
    # France
    "rue": "rue", "r": "rue", "r.": "rue", "avenue": "ave", "chemin": "ch", "ch": "ch", "allee": "all",
    "all": "all", "all.": "all", "impasse": "imp", "imp": "imp", "cours": "crs", "quai": "quai", "passage": "pass",
    "faubourg": "fbg", "fbg": "fbg", "residence": "res", "res": "res", "lieu": "lieu", "dit": "dit", "cite": "cite",
    "esplanade": "esp", "esp": "esp", "promenade": "prom", "hameau": "ham", "zone": "zone", "za": "za", "zi": "zi",
    "zac": "zac", "batiment": "bat", "bat": "bat", "bis": "bis", "ter": "ter",
    # unit / building words (US/India)
    "suite": "ste", "ste": "ste", "unit": "unit", "apartment": "apt", "apt": "apt", "building": "bldg",
    "bldg": "bldg", "bld": "bldg", "room": "rm", "rm": "rm", "shop": "shop", "office": "office", "flat": "flat",
    "tower": "tower", "plot": "plot", "plt": "plot", "house": "house", "door": "door", "gate": "gate",
    "north": "n", "n": "n", "south": "s", "s": "s", "east": "e", "e": "e", "west": "w", "w": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
}
STREET_TYPE_SET = set(STREET_TYPES.values()) - {"n", "s", "e", "w", "ne", "nw", "se", "sw", "bis", "ter"}
STREET_STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "au", "aux", "sur", "sous", "en", "the", "of", "and", "at"}
UNIT_WORDS = {"ste", "unit", "apt", "fl", "rm", "shop", "office", "flat", "floor", "suite", "condo", "bldg", "tower", "gr"}
HN_WORDS = {"h", "hno", "hn", "no", "nos", "number", "num", "door", "flat", "plot", "shop", "office", "sf", "survey",
            "khasra", "kh", "building", "bldg", "house", "sy", "sno", "gala", "unit", "room", "rm", "site", "ward",
            "old", "new", "d", "wo", "ct", "mig", "lig", "hig", "cts", "ts", "block", "blk"}
# locality tokens that carry no information / are added as noise
LOC_JUNK = {
    "null", "n/a", "na", "none", "nil", "city", "town", "village", "of", "the", "cdp", "twp", "township", "county",
    "region", "hq", "urban", "rural", "dist", "district", "tehsil", "taluka", "taluk", "mandal", "po", "ps", "block",
    "unit", "sector", "phase", "part", "parts", "area", "zone", "greater", "metro", "metropolitan", "municipal",
    "corporation", "nagar", "and", "&", "cedex", "france", "india", "usa", "us", "borough", "parish",
    "suburban", "central", "north", "south", "east", "west", "new", "old", "upper", "lower",
}
_HN_KEYWORD_RE = re.compile(
    r"\b(?:h\.?\s*no\.?|hno|hn|house\s*no\.?|door\s*no\.?|flat\s*no\.?|plot\s*no\.?|shop\s*no\.?|office\s*no\.?|"
    r"survey\s*no\.?|building\s*no\.?|bldg\s*no\.?|khasra\s*no\.?|kh\.?\s*no\.?|s\.?\s*no\.?|sy\.?\s*no\.?|"
    r"gala\s*no\.?|unit\s*no\.?|room\s*no\.?|site\s*no\.?|no\.?|nos\.?|number|num\.?)\s*[:\-]?\s*#*\s*"
)
_PO_BOX_RE = re.compile(r"\b(?:p\.?\s*o\.?\s*box|po\s*box|post\s*box|bp|cs)\s*#?\s*\d+\b")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th|er|e|eme|ème)$")
_DIGIT_RUN_RE = re.compile(r"\d+")
_ALPHA_TOKEN_RE = re.compile(r"[a-z]+")
_FRACTION_RE = re.compile(r"\b\d+\s*/\s*\d+\b")
_RANGE_RE = re.compile(r"(?<=\d)\s*-\s*(?=\d)")


def _norm_digit(d: str) -> str:
    d = d.lstrip("0")
    return d if d else "0"


def _clean_component(comp: str) -> str:
    comp = comp.replace("&", " and ").replace("'", "")
    comp = re.sub(r"[\[\]\(\)\{\}<>*!?:;_~^=\"]+", " ", comp)
    comp = comp.replace("#", " # ")
    comp = _WS_RE.sub(" ", comp).strip(" .-")
    return comp


def _digit_runs(text: str) -> List[str]:
    return [_norm_digit(d) for d in _DIGIT_RUN_RE.findall(text)]


def parse_address(raw: str) -> Dict[str, object]:
    """Parse an address into comparable pieces.

    Returns dict with:
      comps      list of cleaned comma-components (ascii lower)
      alpha      space-joined alphabetic tokens (street types canonicalised, junk removed)
      hn         primary house number digit run ('' if none) taken from the house-number component
      hn_runs    all digit runs of the house-number component (leading zeros stripped)
      nums       all digit runs in the address except PO-box / ordinal / postal-looking runs
      street     alphabetic street tokens (component holding the house number, minus type/keywords)
      stype      canonical street type in the street component ('' if none)
      loc        locality tokens (all alpha tokens outside the street component, junk removed)
      unit       unit designator digits/letters ('' if none)
      postal     5/6-digit postal-looking run appearing outside the house-number component ('')
      has_hn_kw  1 if a house-number keyword (H.No / Plot No / Door No / # ...) was present
    """
    if raw is None:
        raw = ""
    text = to_ascii(str(raw))
    text = _WS_RE.sub(" ", text).strip()
    if not text:
        return {"comps": [], "alpha": "", "hn": "", "hn_runs": [], "nums": [], "street": "", "stype": "",
                "loc": "", "unit": "", "postal": "", "has_hn_kw": 0, "ncomp": 0}
    pobox = _PO_BOX_RE.findall(text)
    text = _PO_BOX_RE.sub(" ", text)
    comps = [_clean_component(c) for c in text.split(",")]
    comps = [c for c in comps if c]
    has_hn_kw = 1 if _HN_KEYWORD_RE.search(text) or "#" in text else 0
    # choose the house-number component: first component containing a digit that is not an ordinal-only,
    # preferring components that do not start with a unit designator ("Suite 7", "Unit 105", "Fl 2")
    hn_idx = -1
    fallback = -1
    for i, c in enumerate(comps):
        toks = c.split()
        if any(any(ch.isdigit() for ch in t) and not _ORDINAL_RE.match(t) for t in toks):
            first = toks[0].rstrip(".")
            if STREET_TYPES.get(first, first) in UNIT_WORDS or first in ("po", "p.o", "box"):
                if fallback < 0:
                    fallback = i
                continue
            hn_idx = i
            break
    if hn_idx < 0:
        hn_idx = fallback
    hn, hn_runs, street_tokens, stype, unit = "", [], [], "", ""
    all_alpha: List[str] = []
    loc_tokens: List[str] = []
    nums: List[str] = []
    postal = ""
    for i, c in enumerate(comps):
        c2 = _HN_KEYWORD_RE.sub(" ", c) if i == hn_idx else c
        c2 = c2.replace("#", " ")
        toks = _WS_RE.sub(" ", c2).strip().split()
        canon = []
        j = 0
        while j < len(toks):
            t = toks[j]
            t_l = t.rstrip(".")
            if t_l in STREET_TYPES:
                canon.append(STREET_TYPES[t_l])
            else:
                canon.append(t_l)
            j += 1
        # unit designators
        for j, t in enumerate(canon):
            if t in UNIT_WORDS and j + 1 < len(canon) and not unit:
                unit = canon[j + 1]
        # alphabetic view: ordinals kept whole (116th), unit values dropped, letters glued to digits dropped (7a, 314/b)
        alpha: List[str] = []
        skip_next = False
        for t in canon:
            if skip_next:
                skip_next = False
                continue
            if t in UNIT_WORDS:
                skip_next = True
                alpha.append(t)
                continue
            if _ORDINAL_RE.match(t):
                alpha.append(t)
                continue
            a = re.sub(r"[^a-z]", "", t)
            if not a:
                continue
            if any(ch.isdigit() for ch in t) and len(a) <= 2:
                continue  # 7a, 314/b, p3, 132c -> house-number suffix, not a word
            alpha.append(a)
        if i == hn_idx:
            # digit runs in the house-number component; primary = first run of the first token with a digit
            for t in canon:
                if any(ch.isdigit() for ch in t) and not _ORDINAL_RE.match(t):
                    r = _DIGIT_RUN_RE.findall(t)
                    if r:
                        hn = _norm_digit(r[0])
                        break
            hn_runs = [_norm_digit(d) for t in canon if not _ORDINAL_RE.match(t) for d in _DIGIT_RUN_RE.findall(t)]
            types_here = [t for t in alpha if t in STREET_TYPE_SET]
            stype = types_here[-1] if types_here else ""
            street_tokens = [t for t in alpha if t not in HN_WORDS and t not in UNIT_WORDS and t not in LOC_JUNK
                             and t not in STREET_STOP and t != stype]
            nums.extend(hn_runs)
        else:
            for t in canon:
                if _ORDINAL_RE.match(t):
                    continue
                for d in _DIGIT_RUN_RE.findall(t):
                    dn = _norm_digit(d)
                    if len(d) in (5, 6) and not postal:
                        postal = d
                    nums.append(dn)
            loc_tokens.extend(t for t in alpha if t not in LOC_JUNK and t not in STREET_TYPE_SET and t not in UNIT_WORDS
                              and t not in STREET_STOP and not _ORDINAL_RE.match(t))
        all_alpha.extend(t for t in alpha if t not in LOC_JUNK and t not in UNIT_WORDS)
    # dedupe while keeping order
    def _uniq(xs):
        seen = set(); out = []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return {
        "comps": comps,
        "alpha": " ".join(_uniq(all_alpha)),
        "hn": hn,
        "hn_runs": _uniq(hn_runs),
        "nums": _uniq(nums),
        "street": " ".join(_uniq(street_tokens)),
        "stype": stype,
        "loc": " ".join(_uniq(loc_tokens)),
        "unit": unit,
        "postal": postal,
        "has_hn_kw": has_hn_kw,
        "ncomp": len(comps),
    }


def normalize_country(raw: str) -> str:
    t = to_ascii(str(raw or "")).strip()
    t = re.sub(r"[^a-z ]", "", t).strip()
    aliases = {"usa": "us", "united states": "us", "united states of america": "us", "u s": "us", "u s a": "us",
               "in": "india", "ind": "india", "bharat": "india", "fr": "france", "fra": "france",
               "republique francaise": "france"}
    return aliases.get(t, t)


# ----------------------------------------------------------------------------------------------
# Frame-level driver
# ----------------------------------------------------------------------------------------------
NAME_COLS = ["name", "name_alt", "core", "core2", "sorted2", "compact", "skel", "legal", "dom", "first", "ntok"]
ADDR_COLS = ["alpha", "hn", "hn_runs", "nums", "street", "stype", "loc", "unit", "postal", "has_hn_kw", "ncomp", "comps"]


def normalize_records(names: List[str], addrs: List[str]) -> Tuple[Dict[str, list], Dict[str, list]]:
    """Normalize parallel lists of names/addresses. Returns (name_columns, address_columns) as dict of lists."""
    n_out = {k: [] for k in NAME_COLS}
    a_out = {k: [] for k in ADDR_COLS}
    cache_n: Dict[str, Dict[str, str]] = {}
    for nm in names:
        r = cache_n.get(nm)
        if r is None:
            r = parse_name(nm)
            if len(cache_n) < 2_000_000:
                cache_n[nm] = r
        for k in NAME_COLS:
            n_out[k].append(r[k])
    for ad in addrs:
        r = parse_address(ad)
        for k in ADDR_COLS:
            a_out[k].append(r[k])
    return n_out, a_out

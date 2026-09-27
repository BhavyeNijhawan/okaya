"""Merge a country-restricted prediction (new pipeline) into a full baseline submission.

For every Source-1 test entity whose country is in --countries, take the matched/candidate lists from
the new files; for every other entity keep the baseline's lists. Then run the official validator.

Usage:
  python utils/merge_hybrid.py --test-dir D:/code/dataset/test --countries france \
      --base-match D:/code/r11_candidate/matching_results.tsv --base-cand D:/code/r11_candidate/candidate_pairs.tsv \
      --new-match new/matching_results.tsv --new-cand new/candidate_pairs.tsv --out D:/code/hybrid
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def read_lists(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            out[row[0]] = row[1] if len(row) > 1 else ""
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--countries", required=True, help="comma-separated normalized country labels, e.g. france")
    ap.add_argument("--base-match", required=True); ap.add_argument("--base-cand", required=True)
    ap.add_argument("--new-match", required=True); ap.add_argument("--new-cand", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    countries = {c.strip().lower() for c in a.countries.split(",")}
    s1_country = {}
    order = []
    with open(Path(a.test_dir) / "test_source1.tsv", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        for row in r:
            s1_country[row[0]] = row[3].strip().lower()
            order.append(row[0])
    base_m = read_lists(Path(a.base_match)); base_c = read_lists(Path(a.base_cand))
    new_m = read_lists(Path(a.new_match)); new_c = read_lists(Path(a.new_cand)) if a.new_cand else {}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    n_new = 0
    with open(out / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as fm, \
         open(out / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n"); fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for e in order:
            if s1_country.get(e) in countries and e in new_m:
                m = new_m[e]; c = new_c.get(e, base_c.get(e, "")); n_new += 1
            else:
                m = base_m.get(e, ""); c = base_c.get(e, "")
            fm.write(f"{e}\t{m}\n"); fc.write(f"{e}\t{c}\n")
    print(f"hybrid written to {out}: {n_new:,} entities from the new prediction, {len(order) - n_new:,} from the baseline")
    validator = Path(__file__).resolve().parent / "validate_submission.py"
    subprocess.run([sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
                    "--candidate", str(out / "candidate_pairs.tsv"), "--test-dir", a.test_dir], check=False)


if __name__ == "__main__":
    main()

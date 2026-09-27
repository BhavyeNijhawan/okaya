# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** okaya
**Team Members:** Bhavye Nijhawan
**Submission Date:** 2026-09-27

---

## 1. Executive Summary
A two-stage entity-resolution pipeline: lossless region-partitioned blocking (char-3-gram TF-IDF in
both directions plus exact keys), a LightGBM meta-blocking pruner, and a two-stage LightGBM matcher
whose second stage sees score context, pool-side competition and *sibling evidence* (copies of one
business share source-specific noise). Decisions maximise the expected per-entity F0.5 with
calibrated probabilities and pool exclusivity. The key innovations are (i) a training/validation
universe that reproduces the test set's distractor density by hiding 18.8% of Source-1 entities,
(ii) region and transliteration tables learned from the labels (Indic scripts, French
départements), and (iii) generator-aware features (legal-form transitions and added content words
that mark *different* businesses, house-number corruption patterns, alias/domain decoding).

---

## 2. Methodology

### 2.1 Problem Analysis
* Copies are generated per source from a noisy base copy plus per-record noise: siblings share
  house-number and name corruptions (S1 `Office No.310` → every S3 copy `Office No.311`).
* Name noise: legal-suffix variants, dropped/moved suffixes, prefixes (Sri/Shri/Mr/M/s), appended
  generic words (Center/Services/Group), duplicated or dropped words, word shuffles, random letter
  scrambles, l33t digits, random accents, full Indic-script transliteration, domain forms
  (`hrsinvestments.com`, `#alhigh`), alias forms (`X formerly known as Y`, `dba`, `t/a`, `née`),
  decorations (`*** `, `(ID: 8283)`).
* Address noise: abbreviations (incl. Street→Saint), state full/abbr/Indic script, France
  region↔département, old/new names (Telangana↔Andhra Pradesh), nearby-locality substitutions,
  component reordering/dropping, house-number zero padding, dropped digits, fractions/ranges/letter
  suffixes, prefix junk; 3.3% of pool records have an empty address.
* Distractors look like siblings of real entities: a *changed* legal form (LLC→INC) or an *added*
  content word (Group/Holding/Solutions/India) almost never marks a copy, while added
  formerly/fka/mr/dr/com tokens are pure noise.
* 30% of S1 share their normalized name with another S1 of the same country (US 26%, India 37%);
  2.3% share name and city. Only 52% of colliding S1 have identical raw spelling, so raw-form
  agreement carries information.
* Test has 5.75 pool records per S1 vs 4.68 in train: about 1.9x more distractors per entity.
  Region (state) agrees on 100% (US) / 99.99% (India) of true pairs when present.

### 2.2 Solution Strategy
**Approach Type:** Blocking + pruner + two-stage classifier + expected-utility decision layer
**Core Innovation:** test-faithful training universe (hidden S1 → orphan density), learned
region/transliteration tables, sibling evidence in stage 2, expected-F0.5 decisions.

---

## 3. Candidate Generation (Blocking)
* **Partition:** country, then learned region code (US state, Indian state/UT with merged
  Telangana/Andhra Pradesh, French région with départements mapped to régions); records without a
  code are matched against the whole country.
* **Channels:** char-3-gram TF-IDF cosine top-K per S1 on core name (K=8), name+house number+
  street+locality (K=8) and address (K=4); reverse pool→S1 top-K (K=2, K=10 for records without a
  region/address); exact keys: region|house number|street token, region|house number|locality
  token, region|core token|house number, sorted core tokens, concatenated core (matches domain
  forms), consonant skeleton, sorted letters (shuffled concatenations). GPU random-projection
  matmuls (T4) or exact sparse products (CPU).
* **Pruner:** LightGBM on blocking signals (cosines, channel bits, ranks, house-number and
  region agreement, ambiguity counts); keep p ≥ 0.003, top-2 per S1, cap 40.
* **Blocking keys used:** see above.
* **Candidate pairs generated:** raw union ≈ 25–30 per S1; after the pruner ≈ 10 per S1
  (`candidate_pairs.tsv` = the pruned set the stage-2 model scores).
* **How we ensured true matches were not lost:** pair recall measured on a test-density
  development universe (US 99.8%, India 99.5% before pruning), per-channel unique-contribution
  analysis, learned transliteration dictionary for Indic-script names, multi-token house-number
  keys, larger reverse K for empty-address records.

---

## 4. Matching Model
**Features used:**
- Name: rapidfuzz ratio / token-set / token-sort / partial on core names, full names, concatenated
  names (Jaro-Winkler, Indel), raw spelling ratio and equality, sorted/compact/skeleton equality,
  alias-side and domain-form similarities, token-set intersection/Jaccard/IDF-weighted overlap,
  learned extra-token and missing-token match rates (min/max, identity-marker and noise-marker
  counts), legal-form agreement flags and learned transition rate.
- Address: empty flag, house-number relations (both/equal/digit-drop prefix/subset/run
  intersection/absolute difference/conflict), street and locality token overlap and fuzzy ratios,
  street-type and unit agreement, all-token set similarity, region same/conflict/unknown, name
  tokens inside the other side's address.
- Context: blocking cosines and channel bits, ranks and gaps within the S1 candidate set and among
  the pool record's claimants, pruner score, ambiguity counts (S1 name frequency, in-region name
  frequency, pool name frequency, claimants by name), source indicator, Indic-script flag.
- Stage 2: stage-1 probability and logit, ranks/gaps/sums within the S1 group and on the pool side
  (competitor mass), per-source confident-match counts, sibling evidence (best raw-name, name,
  house-number, street, locality, legal and script agreement with the S1's confident anchors in the
  same source; best competing claimant's sibling agreement).

**Model type:** LightGBM (stage 1: 255 leaves, early stopping; stage 2 trained on out-of-fold
stage-1 scores), isotonic calibration.
**Threshold selection method:** per-entity expected-F0.5 prefix selection (Monte Carlo over
calibrated probabilities plus a Poisson term for unretrieved matches) vs. global thresholds,
selected on the validation slice of the test-faithful universe; pool exclusivity.

---

## 5. Results & Error Analysis
- **F_0.5 Score (macro):** see `work/train/tune_report.json` produced by the run (validation
  slice of the test-density universe; per-country figures included).
- **Common false positives (wrong merges):** same-name entities in the same city with partial
  addresses; look-alike distractors differing only by legal form or an added content word.
- **Common false negatives (missed matches):** empty-address copies of names shared by several
  S1 entities (ambiguous by construction), heavy name rewrites with empty address, Indic-script
  copies with very partial addresses.

---

## 6. Conclusion
Matching the test distribution during training, learning the geography and transliteration from the
labels, and letting the decision layer follow the metric gave a compact pipeline that runs on a free
Colab T4 and generalises to the unseen France partition with country-agnostic features.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/ber/` (package), `src/run.py` (CLI: `train`, `predict`),
`requirements.txt`, `README.md` (exact commands), `colab/BER_Colab.ipynb` (one-click Colab run).
`python run.py train --data <dataset> --work work` then
`python run.py predict --data <dataset> --work work --out output` regenerate
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.

### B. Additional Results
Diagnostics of the test prediction (candidates per S1, predicted matches per S1 and per source,
share of S1 left empty, calibrated-probability profile) are written to `output/diagnostics.json`.

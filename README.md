# okaya — Business Entity Resolution (Amazon ML Challenge 2026)

A compact, reproducible pipeline that matches noisy Source-2 / Source-3 business records to the
deduplicated Source-1 reference (US, India and the unseen France), optimised for the official
metric (macro F0.5 per Source-1 entity) and built to run end to end on a free Google Colab T4 in a
few hours. It uses only the challenge data: no external lookups, no pretrained language models, all
libraries MIT/BSD/Apache/ISC licensed.

```
data → normalize → region/transliteration tables → blocking (GPU) → pruner → stage-1 LightGBM
     → stage-2 LightGBM (score context + sibling evidence) → isotonic calibration
     → expected-F0.5 decisions with pool exclusivity → matching_results.tsv + candidate_pairs.tsv
```

## Quick start (Colab)
Open `colab/BER_Colab.ipynb` in Google Colab (T4 GPU runtime) and run the cells top to bottom.
It clones this repo, downloads the official `student_resource.zip`, trains, predicts, validates the
two output files with the official checker and downloads one zip with outputs, code, models and
reports. All stages are checkpointed: after a disconnect, re-run the cells and finished stages are
skipped.

## Quick start (local)
```bash
pip install -r requirements.txt
python run.py train   --data <folder with train/ and test/> --work work --jobs 4
python run.py predict --data <...> --work work --out output --jobs 4
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir <...>/test
```
`--no-gpu` forces the exact CPU retrieval (sparse_dot_topn); expect several hours at full scale.

## What makes it work
| Component | What it does | Why |
|---|---|---|
| `ber/normalize.py` | parallel views of every record: full/core/sorted/concatenated/consonant-skeleton names, house-number digit runs, street, locality; legal forms canonicalised (SASU kept distinct from SAS); alias forms (`X formerly known as Y`, `dba`, `t/a`, `née`) split; domain/handle forms (`hrsinvestments.com`, `#alhigh`) decoded; l33t digits; Indic scripts transliterated | the generator applies these transforms; keeping several views lets the model measure agreement under each |
| `ber/geo.py` | region code per record: seed tables (US states, Indian states/UTs, French regions + départements) plus aliases *learned* from the data (Indic-script state names, cities, `Keralam`, `TG`…); codes the labels show to be interchangeable are merged (Telangana ↔ Andhra Pradesh) | on true pairs the region agrees >99.9% of the time, so region partitions make blocking lossless and cheap |
| `ber/translit.py` | token dictionary Indic → Latin learned from the labelled pairs (≈540 tokens, 76% token coverage) | `सिल्वर टेक` becomes `silver tech` instead of `silvr tek` |
| `ber/blocking.py`, `ber/chargram.py`, `ber/gpu_topk.py` | per country and region: char-3-gram TF-IDF top-K on name, name+address and address, both directions (pool→S1 too), exact keys (region\|house number\|street or locality token, region\|core token\|house number, sorted core, concatenated core, skeleton, sorted letters); GPU random-projection matmuls on Colab, exact sparse products on CPU | 99.8% (US) / 99.5% (India) pair recall on a test-density development universe |
| `ber/prune.py` | LightGBM meta-blocking pruner on blocking signals: keep p ≥ 0.003, top-2 per S1, cap 40 | the raw union (25–30 per S1) becomes ~10 per S1 with ~no recall loss; `candidate_pairs.tsv` is exactly this set |
| `ber/features.py` | ~110 stage-1 features: rapidfuzz similarities on several views (incl. raw spelling), sparse token-set arithmetic, learned legal-transition and extra/missing-token match rates (a changed legal form or an added content word marks a *different* business), house-number relations (equal / zero-padded / digit-dropped prefix / subset / conflict), locality, street, unit, region agreement, ambiguity counts, blocking ranks; stage 2 adds score context (ranks, gaps, competitor mass on the pool side, per-source confident counts) and **sibling evidence** (similarity of a candidate to the pool records already confidently matched to the same S1 in the same source — copies share base-copy noise) | precision-heavy metric: the model must separate copies from look-alike distractors |
| `ber/decide.py` | isotonic-calibrated probabilities → per S1 the expected-F0.5-optimal prefix (Monte Carlo incl. unretrieved matches), P(empty) for singletons, pool exclusivity | the break-even probability depends on how many confident matches an entity already has (0.5 … 0.77), not one threshold |
| `ber/pipeline.py` | training universe with 18.8% of S1 hidden (their copies become orphans → test-like distractor density), disjoint slices for tables/pruner, matcher, validation; out-of-fold stage-1 scores for stage-2 training; per-country checkpoints | validation that behaves like the leaderboard |

## Layout
```
ber/            pipeline package (see module docstrings)
run.py          CLI: train / predict / stage
colab/          Colab notebook
utils/          official validate_submission.py
Documentation_template.md   methodology write-up
```

## Reproducibility
Seeded (`--seed`, default 42). Every stage writes parquet/npy checkpoints under `--work`;
`candidate_pairs.tsv` is written from the exact pair list the stage-2 model scored and
`matching_results.tsv` is a mask over it, so matches are a subset of candidates by construction.

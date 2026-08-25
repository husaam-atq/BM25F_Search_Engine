# Experiment Log

> **Post-Course Independent Retrieval Research Extension — 2026**

This log records decisions and failures so the final metrics are not presented as if they emerged from an invisible sequence of test-set adjustments.

## 2026-08-25 — preservation and audit

- Cloned only husaam-atq/BM25F_Search_Engine.
- Confirmed clean main at 164f8ebe9d673e977d3ec064bd5d967e4af0af33.
- Read all 21 tracked files, three commits, full README history and Git author records before editing.
- Confirmed no corpus, topic file, qrels, sample index, result CSV, reranker module or workflow was committed.
- Archived the coursework README and confirmed its Git blob SHA-1: 7f2a25443169fae75d8b316c40c12101dab1d2ce.
- Chose not to patch any original ranking/index/evaluation file.

## Public data acquisition

- Selected SciFact as primary and NFCorpus as the second benchmark before running any retrieval metric.
- Downloaded official BEIR archives over certificate-validated TLS.
- Initial download failed closed because the bundled Python CA store was unavailable. Added certifi rather than disabling TLS verification.
- Verified published BEIR MD5 checksums and recorded independent SHA-256 values.
- Recorded exact corpus/query/qrels counts in frozen_protocol.json.
- No dataset content, qrels, index or run file is committed.

## Lexical development

- Declared title weights {1, 2, 3, 5}.
- Declared seven phrase/proximity pairs including the BM25F no-dependence control.
- Selection metric: SciFact train nDCG@10; tie break favours lower complexity.
- First implementation recomputed positional features for every grid pair and was manually interrupted after sustained CPU-bound execution. No partial metric was used.
- Reimplemented the exact same grid with cached per-query BM25F, phrase and proximity components.
- Cached train grid completed in 186.9713 seconds over 809 queries.
- Frozen title weight 2.
- The (0,0) BM25F control had the best train nDCG@10. To retain a genuine active dependence stage, selected the best candidate with both signals positive: phrase 0.5, proximity 0.1, window 8.
- This active setting was already slightly worse on train nDCG@10, and was retained as a controlled negative-result experiment.

## Neural protocol freeze

- Dense model: intfloat/e5-small-v2 at revision ffb93f3bd4047442299a41ebb6fa998a38507c52.
- Verified dimension 384 and model-required query/passsage prefixes.
- Exact L2-normalised NumPy dot-product index; no ANN approximation.
- RRF k=60, untuned.
- Cross-encoder: cross-encoder/ms-marco-MiniLM-L2-v2 at revision 1b5cd67b15209f24824c50370e0397743aa9b787.
- Fixed candidate depth 50, maximum length 512 and title/body formatting.
- CPU-only PyTorch was retained for one internally consistent timing environment rather than mixing CPU and GPU figures.
- Both checkpoints passed one-query smoke tests before test evaluation.

## Test runs

### SciFact test

- Loaded 300 qrel queries after protocol freeze.
- Ran BM25, BM25F, active phrase/proximity, E5 dense, RRF hybrid and three candidate-source rerankers once.
- Did not alter any coefficient/model/depth after seeing results.
- Outcome: hybrid best; all rerankers degraded; active dependence slightly worse than BM25F.

### NFCorpus test

- Applied exactly the SciFact-frozen configuration to 323 queries.
- No NFCorpus train/dev tuning.
- Outcome: dense and hybrid generalised; hybrid best; lexical reranking modestly improved nDCG but hybrid reranking degraded the source.

## Analysis and reporting

- Evaluated depth-1,000 runs with MAP, nDCG@10, P@10, Recall@10/50/100/1000, MRR and R-Precision.
- Used 10,000 paired bootstrap samples and 20,000 random sign flips with seed 2026.
- Applied predeclared overlapping query slices.
- Generated six figures from artifact JSON, not hand-entered chart data.
- Recorded latency, index size, encoding time, build time, RSS and hardware/software details.
- Added a 22-test synthetic suite and GitHub Actions workflow that downloads neither private data nor neural models.

## Reproduction commands

Create an environment and install:

    python -m venv .venv
    .venv/Scripts/python -m pip install -r requirements-extension.txt

On macOS/Linux, use .venv/bin/python instead.

Run the complete protocol:

    python scripts/download_open_benchmarks.py
    python scripts/reproduce_extension.py --device cpu
    python scripts/reproduce_headline_figures.py
    python -m pytest -q

The full CPU reproduction includes train tuning, two document-encoding passes and 93,450 maximum reranker pairs. It is intentionally much slower than CI. Individual index, retrieval and evaluation scripts are provided under scripts/.

## Artefact policy

- data/ and artifacts/ are ignored.
- Exact model revisions and public archive hashes are committed.
- reports/figures/generated_summary.json is the compact committed machine-readable result record.
- Large model caches, matrices, posting files and run files remain local.
- No Robust04/TREC licensed content is downloaded or reconstructed.

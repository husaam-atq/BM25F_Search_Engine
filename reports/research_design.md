# Research Design

> **Post-Course Independent Retrieval Research Extension — 2026**

## Central question

**When does modern semantic retrieval genuinely improve on a strong lexical engine, and what does it cost in latency, complexity and interpretability?**

The study preserves negative results. It does not assume dense retrieval or reranking must win, and it does not adapt configurations after inspecting test metrics.

## Public benchmarks

Both benchmarks use public BEIR-format archives documented by the [BEIR project](https://github.com/beir-cellar/beir). Archives are downloaded locally, checksum-verified and never committed.

| Benchmark | Role | Documents | Queries in archive | Development | Test queries | Test judgments | Archive SHA-256 |
|---|---|---:|---:|---:|---:|---:|---|
| SciFact | primary scientific claim retrieval | 5,183 | 1,109 | 809 train / 919 judgments | 300 | 339 | 536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165 |
| NFCorpus | nutrition/medical generalisation | 3,633 | 3,237 | not used for tuning | 323 | 12,334 | efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b |

Official BEIR MD5 values (5f7d…fca1 and a89d…d38d) are checked before extraction. SciFact licensing is described by the [official SciFact license](https://github.com/allenai/scifact/blob/master/LICENSE.md): CC BY 4.0 for claims/evidence annotations and ODC-By 1.0 for S2ORC abstracts. The public NFCorpus page provides the benchmark download and citation, but the BEIR archive contains no explicit license file; this repository therefore does not redistribute it and instructs users to verify permitted use. BEIR makes the same downstream-license caveat.

## Freeze and validation protocol

The machine-readable record is [frozen_protocol.json](../research_extension/frozen_protocol.json).

1. Inspect package integrity and counts only.
2. Tune a small declared lexical grid on **SciFact train** using nDCG@10.
3. Freeze one configuration for both test datasets.
4. Freeze model identifiers/revisions, RRF k=60, retrieval depth 1,000 and rerank depth 50.
5. Run SciFact test once.
6. Run NFCorpus test once without dataset-specific adaptation.
7. Preserve every result, including degraded additions.

Title weights {1, 2, 3, 5} and seven phrase/proximity pairs were tested. Title weight 2 won. The no-dependence control had the best train nDCG@10; because stage 3 explicitly tests both signals, the best pair with both coefficients active—phrase 0.5, proximity 0.1, window 8—was frozen. It was already slightly worse than BM25F on train nDCG@10, so carrying it forward is a deliberate negative-result test.

## Lexical systems

The extension implements a new positional engine; it does not call or patch the coursework engine.

- Unicode-aware case-folding; no stopword removal or stemming.
- Explicit title/body token streams and positions.
- Deterministic document-ID tie-breaking.
- Standard positive BM25 IDF and (k1+1) saturation factor.
- BM25 over flattened title+body.
- BM25F with title weight 2, body weight 1 and independent field-length normalisation.
- Ordered adjacent-query-pair phrase evidence.
- Minimum-gap proximity evidence across all query-term pairs.
- Repeated query terms retained and weighted.

Not stemming or removing stopwords makes token/position behaviour transparent across scientific terminology and avoids runtime corpus downloads.

## Dense retrieval

The encoder is [intfloat/e5-small-v2](https://huggingface.co/intfloat/e5-small-v2), revision ffb93f3bd4047442299a41ebb6fa998a38507c52:

- 384-dimensional embeddings;
- query input: “query: {query}”;
- document input: “passage: title: {title}\nbody: {text}”;
- maximum length 512 tokens with consistent truncation;
- model-defined average pooling;
- L2 normalisation and dot product (cosine);
- exact NumPy matrix search, not approximate ANN;
- no fine-tuning on either benchmark.

E5’s model card requires query/passage prefixes and documents the 384-dimensional representation. Exact search removes ANN approximation as a confound.

## Hybrid fusion

Lexical and dense ranks use Reciprocal Rank Fusion:

\[
\mathrm{RRF}(d)=\sum_r \frac{1}{60+\operatorname{rank}_r(d)}
\]

RRF was introduced by [Cormack, Clarke and Büttcher](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/). It avoids pretending BM25F and cosine scores share a calibrated scale. k=60 is fixed and untuned.

## Reranking

The cross-encoder is [cross-encoder/ms-marco-MiniLM-L2-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L2-v2), revision 1b5cd67b15209f24824c50370e0397743aa9b787 (Apache-2.0, 15.6M parameters).

- Candidate depth is fixed at 50.
- Inputs are query plus “title: …\nbody: …”.
- Maximum length is 512 tokens.
- The top 50 are reordered; the remainder is appended unchanged.
- Reranking cannot add, remove or duplicate a candidate.
- Lexical, dense and hybrid sources are tested separately.
- No reranker score is mixed with first-stage scores.

## Evaluation and uncertainty

Runs are depth 1,000. Every qrel query is included, including zero-retrieval cases. Duplicate run IDs are removed before scoring. Metrics are MAP, P@10, nDCG@10, Recall@10/50/100/1000, MRR and R-Precision.

Major paired differences use:

- 10,000 paired bootstrap samples for a 95% mean-difference interval;
- 20,000 two-sided random sign-flip samples;
- query wins/losses/ties; and
- paired standardised effect size dz.

Intervals crossing zero are described as statistically uncertain even when aggregate means differ.

## Query slices

Rules were fixed without test-performance inspection:

- short: at most 5 lexical tokens;
- medium: 6–9;
- long: at least 10;
- named-entity-heavy: at least 20% of tokens after the first are capitalised/acronyms;
- rare/OOV: at least one term has collection DF ratio at most 1%;
- strong lexical/dense disagreement: top-10 set Jaccard at most 0.20;
- phrase/proximity-sensitive: BM25F and BM25F+dependence top-10 orders differ.

These are overlapping descriptive slices, not learned predictors or multiple-comparison significance claims.

## Efficiency protocol

All timings use the same CPU-only environment:

- Windows 11;
- AMD Ryzen 9 9900X (12 cores / 24 logical processors);
- 128 GB RAM;
- Python 3.12.13;
- PyTorch 2.13.0+cpu.

The installed RTX 5090 was not used because the reproducible environment resolved a CPU-only PyTorch wheel. Index build/embedding time, serialised size, mean/p50/p95 query latency, reranking latency and process RSS are recorded. Hybrid end-to-end latency sums measured lexical, dense and fusion stages; no parallel-execution assumption is made.

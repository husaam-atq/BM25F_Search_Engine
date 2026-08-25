# Limitations

> **Post-Course Independent Retrieval Research Extension — 2026**

## Historical evidence

- The full Robust04 corpus, topics, qrels, sample package and original result CSVs were unavailable.
- Historical metrics are preserved from the coursework README, not reproduced.
- Audit findings cannot identify which exact uncommitted code/environment produced the historical table.
- The 2026 public-benchmark results must not be compared numerically as if they were Robust04 reruns.

## Dataset scope and licensing

- Only two modest English biomedical/scientific benchmarks are tested.
- SciFact has sparse qrels and few relevant documents per query; many per-query ties are expected.
- NFCorpus has many judgments but a narrow nutrition/medical domain.
- The NFCorpus public BEIR archive contains no explicit license file. It is downloaded, not redistributed; users must verify permitted use.
- No web-scale, multilingual, conversational or adversarial benchmark is included.
- Query examples are quoted only where benchmark licensing/publication permits; document text is not reproduced in reports.

## Protocol and tuning

- Lexical settings are tuned only on SciFact train and frozen for NFCorpus. This tests robustness but may understate an NFCorpus-specific optimum.
- Requiring both phrase and proximity coefficients to be positive forces an active diagnostic stage even though the no-dependence train control was slightly better.
- The extension tokenizer does not stem or remove stopwords. This is transparent and deterministic but is not an exhaustive lexical baseline search.
- No WordNet expansion is run on the public benchmarks; the original implementation is audited and preserved instead. Adding expansion would require another predeclared development protocol.
- RRF k=60 and rerank depth 50 are fixed, not sensitivity-tested on test data.

## Dense retrieval and reranking

- E5-small-v2 is one compact general retriever, not a survey of E5/BGE/SPLADE/ColBERT families.
- Exact matrix search is appropriate at 3.6–5.2K documents but does not model ANN recall/latency at millions of documents.
- Documents are encoded as one title+body passage and truncated at 512 tokens. Long-document chunking could change quality, storage and latency.
- The cross-encoder is a compact MS MARCO model. It may be too small or domain-mismatched for scientific claim semantics.
- Only one title/body formatting, maximum length and score policy is tested. They were not changed after test failure.
- No fine-tuning, distillation or final-test-query training is performed.
- Neural scores are less interpretable than lexical evidence; the extension does not claim to solve that limitation.

## Evaluation and inference

- Runs stop at depth 1,000; MAP and recall beyond that depth are unknown.
- Qrels are treated as complete for scoring even though IR judgments can be incomplete.
- Confidence intervals quantify query sampling variability, not dataset/domain uncertainty.
- Several paired comparisons are reported without family-wise multiple-testing correction. The primary comparison is hybrid versus lexical; other p-values are supporting diagnostics.
- Query categories overlap and are heuristic. The capitalization rule is especially noisy for title-cased NFCorpus queries.
- Representative explanations infer likely mechanisms from query/ranking behaviour; they are not document-level causal annotations.
- No learned query router is trained, because doing so safely would need more development data and a separate evaluation.

## Efficiency

- Timings are one Windows/CPU environment and are not universal deployment numbers.
- The available RTX 5090 was not used because installed PyTorch was CPU-only.
- No repeated-run confidence interval, energy use or monetary cost is measured.
- Hybrid latency is a serial component sum. Parallel lexical/dense execution is plausible but unmeasured.
- RSS is process-level and includes Python/model runtime overhead; it is not a clean per-index resident-memory measure.
- Pickle/NPZ compression choices affect reported index sizes.

## Software and CI

- CI validates a tiny committed synthetic fixture, not the full public datasets.
- Neural tests use deterministic fake scorers/embeddings and do not download checkpoints in CI.
- Pickle is used only for locally built trusted lexical indexes and must not be loaded from untrusted sources.
- Reproduction depends on public dataset/model hosts remaining available at the pinned content.

## Bottom line

The evidence supports a bounded conclusion: on these two benchmarks, untuned rank fusion adds useful complementary recall, while a fixed compact cross-encoder is not robust. It does not establish universal superiority of hybrid retrieval, universal reranker failure, or state-of-the-art performance.

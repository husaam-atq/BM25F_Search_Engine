# Historical Coursework Summary

> **Post-Course Independent Retrieval Research Extension — 2026**
>
> This preservation note distinguishes the submitted team coursework from all later independent work. Historical code and numbers are reported as historical evidence; they are not recomputed, corrected, or claimed as 2026 results.

## Part I — Original MSc Team Coursework

The repository identifies the submission as **ECS736P/U Information Retrieval — Coursework 2**. The historical system is a TREC Disk 4 & 5 / Robust04 search engine with:

- collection-specific SGML parsing for FT, FR94, Congressional Record, FBIS and LA Times;
- lowercasing, regex tokenisation, stopword removal and Porter stemming;
- chunked positional indexing described by the team as SPIMI;
- title/body BM25F, phrase and proximity bonuses;
- controlled WordNet synonym expansion;
- six ablation variants evaluated on 249 Robust04 topics;
- MAP, P@10, nDCG@10, Recall@100 and R-Precision;
- Streamlit and command-line interfaces; and
- a historical cross-encoder reranking attempt.

### Original team attribution

The historical README explicitly names:

- **Blazej Olszta**
- **Muhamad Husaam Ateeq**
- **Max Monaghan**
- **Sulaiman Bhatti**

Git history contains commits authored by Husaam (husaam.ateeq@gmail.com) and blazej_olszta (olsztablazej@gmail.com). Git commit authorship is not treated as a replacement for the four-person team attribution above.

### Preserved historical Robust04 results

These values are copied exactly from the coursework README at commit 164f8ebe9d673e977d3ec064bd5d967e4af0af33. They have not been modified in light of the later audit.

| Historical coursework system | MAP | P@10 | nDCG@10 | Recall@100 | R-Precision |
|---|---:|---:|---:|---:|---:|
| BM25 Flattened (baseline) | 0.1832 | 0.3843 | 0.3852 | 0.3777 | 0.2509 |
| BM25 Separate Fields (unweighted) | 0.1603 | 0.3631 | 0.3685 | 0.3418 | 0.2273 |
| BM25F (field-weighted) | 0.1865 | 0.4012 | 0.3997 | 0.3804 | 0.2528 |
| BM25F + Phrase & Proximity | **0.1961** | **0.4040** | **0.4033** | **0.3938** | 0.2655 |
| BM25F + Phrase/Prox + WordNet | 0.1958 | 0.4040 | 0.4014 | 0.3936 | **0.2657** |
| BM25F + Phrase/Prox + WordNet + Neural Rerank | 0.1795 | 0.3795 | 0.3794 | 0.3936 | 0.2449 |

The historical interpretation was that phrase/proximity supplied the strongest lexical improvement, WordNet was approximately neutral on MAP, and reranking reduced MAP. The later extension preserves that failed reranking result as part of the project’s academic history.

### Evidence and preservation boundary

- The original repository has 21 tracked files and three commits.
- The coursework README’s Git blob is 7f2a25443169fae75d8b316c40c12101dab1d2ce.
- A byte-equivalent Markdown copy is archived at [archive/README_coursework_164f8e.md](archive/README_coursework_164f8e.md).
- The full TREC corpus, topics, qrels, sample package, index, evaluation CSV and per-query CSV are not committed.
- Therefore the table above is preserved historical evidence, not an independently reproduced Robust04 run.
- The original indexing, ranking, expansion and evaluation Python files remain unmodified by the extension. Audit findings are documented in [original_system_audit.md](original_system_audit.md), not patched retrospectively.

## Part II — Independent Post-Course Research Extension

The work under research_extension/, tests/, scripts/, reports/figures/ and .github/workflows/ was performed independently after the team coursework and is labelled:

> **Post-Course Independent Retrieval Research Extension — 2026**

Independent extension attribution: **Muhamad Husaam Ateeq**, 2026. This attribution does not reassign or dilute authorship of the original team submission.

The extension evaluates a newly implemented public-benchmark framework on SciFact and NFCorpus. It does not use or reconstruct licensed Robust04 content, and its results must not be compared as if they came from the same corpus or submission period.

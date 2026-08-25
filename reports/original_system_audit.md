# Original System Audit

> **Post-Course Independent Retrieval Research Extension — 2026**
>
> Scope: read-only audit of the original MSc team coursework. The audit does **not** alter the historical implementation or reported Robust04 results.

## Audit basis

The audit covered every tracked file, all three Git commits, the complete README history, and the recorded Git authors/contributors. The worktree was clean before the extension. No TREC corpus, topics, qrels, sample package, index, evaluation CSV or reranker implementation was present, so a fresh historical run could not be attempted without licensed/private inputs.

Severity terms:

- **Material:** can prevent reproduction or change rankings/metrics.
- **Behavioural:** an intentional or plausible design choice requiring explicit interpretation.
- **Engineering:** affects robustness, efficiency or determinism more than mathematical intent.

## Executive findings

1. **Material — fresh indexing is not runnable as committed.** [build_index.py](../build_index.py#L173) and its resume path reference config.SNIPPET_LENGTH, but [config.py](../config.py) defines no such value. The first parsed document raises AttributeError.
2. **Material — the historical neural variant is not reproducible.** [search.py](../search.py#L129) imports reranker.rerank_results, but no reranker.py is present in any commit.
3. **Material — capped positions also become capped TF and field length.** Fresh parallel workers cap each term to 20 positions, then [build_index.py](../build_index.py#L225) uses the shortened token lists as field lengths. Frequent terms and repetitive fields are undercounted.
4. **Material — resume and fresh builds are not equivalent.** The fresh worker path caps positions; the sequential resume path at [build_index.py](../build_index.py#L245) does not. A resumed index can have different TFs, positions, field lengths and rankings.
5. **Material — reported “MAP” is calculated from depth 100.** [evaluate.py](../evaluate.py#L364) retrieves only 100 documents for every metric. average_precision is consequently AP@100 rather than a conventional depth-1000 Robust04 run; R-Precision is truncated whenever R exceeds 100.
6. **Material — repeated query terms are discarded.** [search.py](../search.py#L58) deduplicates normalised terms. Query term frequency and repeated-term phrase evidence are lost, and expansion’s surface/stem alignment can shift after a duplicate.
7. **Material for combined scoring — BM25 omits the conventional (k1+1) numerator factor.** This constant does not change pure BM25 ordering, but it changes lexical scale relative to fixed phrase/proximity bonuses.
8. **Engineering — the claimed encoding fallback is ineffective.** [parse_docs.py](../parse_docs.py#L163) opens UTF-8 with errors="replace"; invalid bytes are replaced rather than raising, so Latin-1/cp1252 fallback is normally never reached.

These findings do not prove that the preserved table is wrong: the exact environment and uncommitted artefacts used for the team’s run are unavailable. They do mean the table cannot be independently reproduced from this commit alone.

## Parsing and field extraction

### What the coursework does

- A non-greedy DOC regex scopes records, limiting accidental cross-document contamination.
- Collection-specific functions map FT/LA HEADLINE, FBIS H3/TI, Congressional Record TTL, and FR94 body-only documents into title/body fields.
- Inner tags and FR94 PJG comments are removed.
- DTD directories and known metadata filenames/extensions are skipped.

### Findings

- The record boundary is sensible for expected TREC SGML and keeps extraction isolated.
- The shared extractor uses only the first matching TEXT block. Documents with multiple TEXT elements lose later blocks.
- .sgml and .xml files are skipped unconditionally. That matches likely disk naming but is an undocumented assumption.
- The UTF-8 fallback issue can insert replacement characters and undermines the README’s “Latin-1 fallback” claim.
- No duplicate DOCNO check exists; a duplicate receives another integer ID and can appear twice under one external identifier.
- Bodies are silently truncated at 200,000 characters before indexing, while snippets use the full parsed string.

## Preprocessing, Porter stemming and stopwords

[preprocess.py](../preprocess.py) lowercases, strips markup/entities, extracts ASCII letter sequences, drops single-letter tokens and English NLTK stopwords, then applies NLTK’s Porter stemmer. Positions count dropped tokens, so phrase/proximity gaps refer to the original token stream.

- Document/query normalisation is shared and deterministic for a fixed Python/NLTK version.
- ASCII-only tokenisation drops digits and non-ASCII letters, affecting dates, model numbers, accented names and multilingual content.
- Removing stopwords while retaining their positional gaps means adjacent processed query terms do not necessarily form an exact phrase across a stopword. This is defensible but unstated.
- NLTK resources are downloaded at import time. The lookup uses a corpora path even for a tagger, so resource detection is unreliable; network state can change startup behaviour.
- Dependencies are lower-bounded but not upper-bounded/pinned, leaving stemmer, POS tagger and UI behaviour version-dependent.

## SPIMI construction, postings and statistics

The builder processes sorted files in parallel, flushes positional dictionaries every 20,000 documents, checkpoints metadata, concatenates disjoint posting runs, computes average field lengths, and pickles final structures.

- Document frequency is counted once per term/document within represented postings and is consistent with the final posting list.
- Positions are sorted and field-separated.
- The final merge retains the entire merged dictionary in memory. It is a practical blocked inverted-index build, but not a fully streaming disk merge.
- Fresh file sorting makes a clean build deterministic on a stable filesystem. Resume uses iter_all_collections, whose os.walk order is unsorted, then skips len(doc_map) records. A changed traversal order can skip different documents or duplicate indexed ones.
- Checkpoints validate run-file existence but not corpus identity, configuration, code version or source-file order.
- Field lengths represent retained/preprocessed tokens, which is internally coherent, but the fresh-path cap bug additionally term-frequency-caps them.
- The unused [index_store.py](../index_store.py) describes SQLite while the builder writes pickle files and creates no idx table. It is stale architecture, not the active access path.

## BM25, BM25F and IDF

- The code uses positive RSJ-style IDF: log(1 + (N-df+0.5)/(df+0.5)).
- Flat BM25 merges title/body lengths; separate-field BM25 sums two field scores; BM25F combines independently normalised fields with title weight 5.
- IDF is finite and positive for valid 0 < df ≤ N; df≤0 and N≤0 return zero. No corrupted df>N validation exists.
- The missing (k1+1) factor is rank-equivalent only when BM25 stands alone. Fixed dependence bonuses make its omission material to the combined model.
- Equal-weight separate scoring applies a saturated contribution per field, so matching both can receive two IDF-weighted contributions. That is a distinct ablation, not standard BM25F.
- The upstream field-length/TF cap affects every variant.
- Equal scores are sorted by score only; stable order inherits insertion order rather than an explicit document-ID tie-break.

## Phrase and proximity bonuses

- Phrase scoring checks each consecutive processed query pair and adds one fixed bonus if either title or body contains an ordered adjacent occurrence.
- Proximity considers every query-term pair, adds title and body contributions separately, and linearly decays to zero at the window edge.
- Multiple occurrences do not accumulate beyond the best check; evidence in both fields can accumulate for proximity but not phrase.
- Because repeated query terms are removed upstream, repeated-word phrases and query-frequency evidence cannot reach these functions.
- Position caps can miss evidence occurring after a term’s twentieth retained occurrence.

## WordNet expansion and leakage

The expansion path POS-tags surface terms, expands nouns only, picks one WordNet sense by gloss overlap, restricts candidates to single-word same-synset lemmas, filters by document frequency/co-occurrence, and downweights accepted stems.

- No qrels or relevance labels are consulted; there is no direct relevance leakage.
- Document frequency and co-occurrence use the complete evaluation corpus. This is normal unsupervised/transductive collection adaptation, but should be labelled.
- Deduplicated stems are zipped against non-deduplicated surface tokens. After a repeated term, POS and stem associations can be wrong.
- POS-tagging failure silently marks everything other, effectively disabling expansion.
- If two originals propose the same expansion, the later assignment overwrites rather than aggregates evidence.
- Rebuilding posting-ID sets for every candidate is expensive on the full collection.

## Topics, qrels and evaluation

- The topic parser extracts numeric ID and same-line TITLE text only.
- The qrels parser accepts four-column rows and lets a later duplicate overwrite an earlier judgment.
- Evaluation silently skips topics absent from qrels and topics with no positive judgments; it does not report set differences.
- P@10 and Recall@100 definitions are conventional.
- AP and R-Precision are truncated by the 100-result run depth.
- nDCG uses all qrel grades for the ideal list. Negative grades, if present, would create negative gains rather than being clamped to non-relevant.
- Metrics do not deduplicate external IDs. The active index returns unique integer documents, but duplicate DOCNO values could still double count.
- No assertion reconciles all 249 topic IDs, qrel IDs and run IDs.

## Neural reranking attempt

The variant fixes depth 50 and requests at least 100 lexical candidates. However:

- the reranker module, model identifier and dependencies are absent;
- document input appears intended to use snippets, but exact formatting, maximum length and score combination cannot be audited;
- the README’s truncation explanation is plausible but not verifiable; and
- candidate recall was not reported, so failure cannot be separated into first-stage ceiling and ordering.

The historical MAP 0.1795 remains preserved with this reproducibility limitation explicit.

## Determinism and reproducibility summary

| Area | Assessment |
|---|---|
| Clean file enumeration | Sorted and broadly deterministic |
| Resume enumeration | Potentially different order and semantics |
| Tokenisation/stemming | Deterministic for fixed NLTK/config |
| WordNet/POS resources | Network/version dependent |
| Score ties | No explicit doc-ID tie-break |
| Index format | Python-pickle/version dependent |
| Dependencies | Broad lower bounds only |
| Licensed data | Correctly absent, but no public CI fixture |
| Fresh build | Blocked by undefined SNIPPET_LENGTH |
| Neural variant | Blocked by missing module |

## Conclusion

The project contains a recognisable real search engine: collection parsing, a positional field-aware index, explicit scoring and a full metric path. Its strongest academic value is interpretability and ablation structure. The issues above are recorded transparently, but the 2026 work does not rewrite the team code or retroactively substitute new Robust04 numbers.

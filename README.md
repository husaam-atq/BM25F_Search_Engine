# Field-Aware BM25F Search Engine

**ECS736P/U Information Retrieval — Coursework 2**

This repository contains the full implementation of a search engine built over the TREC Disk 4 & 5 corpus, including indexing, retrieval, query expansion, and evaluation.

---

## Overview

The system implements:

- **BM25F (field-aware ranking)** over title and body fields
- **Positional inverted index** supporting phrase and proximity search
- **Phrase and proximity scoring bonuses**
- **Controlled WordNet query expansion** with drift filters
- **Ablation evaluation on 249 TREC Robust04 topics**
- **Streamlit GUI** and **command-line interface**

---

## Dataset & Reproducibility

Due to licensing restrictions, the full TREC Disk 4 & 5 dataset is **not included** in this repository. The following are excluded:

- TREC Disk 4 & 5 corpus files
- Prebuilt full index
- Qrels and topic files

A **lightweight sample dataset** (prebuilt index + evaluation files) is provided separately via a OneDrive link included in the submission materials.

**Instructions:**
1. Download the sample dataset from the provided OneDrive link
2. Place the files into the root directory of this project
3. Run:

```bash
run_sample.bat      # Windows
bash run_sample.sh  # Mac / Linux
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/husaam-atq/BM25F_Search_Engine.git
cd BM25F_Search_Engine
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK resources

```bash
python setup_nltk.py
```

### 4. Add the downloaded data

After downloading from OneDrive (link in submission materials), place the files inside the project folder.

---

## Folder Structure

### Sample Mode *(recommended for quick testing)*

```
project-folder/
├── app.py
├── ...
├── sample_index/
├── sample_topics.txt
└── sample_qrels.txt
```

### Full Mode *(requires full TREC dataset)*

```
project-folder/
├── app.py
├── ...
├── TREC-Disk-4/
│   └── TREC-Disk-4/
│       ├── FT/
│       ├── FR94/
│       └── CR_103RD/
└── TREC-Disk-5/
    └── TREC-Disk-5/
        ├── FBIS/
        └── LATIMES/
```

> **Note:** The outer folder name does not matter — the code locates the dataset automatically.

---

## Running the System

### Sample Mode *(FAST —  for markers)*

Runs instantly using the prebuilt sample data.

```bash
run_sample.bat      # Windows
bash run_sample.sh  # Mac / Linux
```

Then launch the GUI:

```bash
streamlit run app.py
```

### Full Mode *(full dataset)*

**Step 1 — Build the index**

```bash
python build_index.py
```

> Takes approximately 10–30 minutes. Requires several GB of RAM.

**Step 2 — Run the system**

```bash
streamlit run app.py
```

or

```bash
run_full.bat        # Windows
```

---

## GUI Usage

The Streamlit interface provides:

- Free-text queries
- TREC topic picker (auto-fills query from any of the 249 Robust04 topics)
- Result cards with source badges and qrels relevance badges (green = relevant)
- Query expansion details toggle
- On-demand full article loading and plain-text download
- Evaluation Results tab with interactive ablation table and bar charts

---

## Command-Line Usage

Interactive mode:

```bash
python search.py
```

Single query:

```bash
python search.py "child support enforcement"
```

Options:

```bash
python search.py --top-k 20 "international trade"
python search.py --no-expand "information retrieval"
python search.py --debug "jet aircraft flight"
```

---

## System Architecture

### Offline Pipeline (runs once)

1. **Parsing** — `parse_docs.py` extracts `docno`, `title`, and `body` from all five SGML collections (FT, FR94, CR, FBIS, LA Times) with Latin-1 encoding fallback
2. **Preprocessing** — `preprocess.py` applies lowercasing, tokenisation, stopword removal, and Porter stemming
3. **Indexing** — `build_index.py` builds a positional inverted index using SPIMI (Single-Pass In-Memory Indexing) in 20,000-document chunks with checkpoint/resume support
4. **Statistics** — Per-document field lengths and collection-level averages persisted alongside the index

### Online Pipeline (every query)

1. Query preprocessing (same pipeline as documents)
2. Optional WordNet expansion with drift filters
3. BM25F scoring + phrase and proximity bonuses
4. Ranked results returned via GUI or CLI

---

## Retrieval Model

### BM25F Parameters

| Parameter | Value |
|-----------|-------|
| K1 | 1.2 |
| B_TITLE | 0.75 |
| B_BODY | 0.75 |
| W_TITLE | 5.0 |
| W_BODY | 1.0 |

### Additional Scoring Signals

- **Phrase bonus** — +1.5 for each pair of consecutive query terms appearing adjacent in a field
- **Proximity bonus** — up to +0.5 per term pair appearing within an 8-word window
- **WordNet expansion** — synonyms added at weight γ = 0.3 with five drift filters (nouns only, IDF threshold, DF cap, co-occurrence requirement, max 3 per term)

### Scoring Formula

```
Score(d, q) =
    BM25F(original terms)
  + 0.3 × BM25F(expanded terms)
  + phrase_bonus
  + proximity_bonus
```

---

## Evaluation Results

Evaluated over **249 TREC Robust04 topics** across six ablation variants.

| System | MAP | P@10 | nDCG@10 | Recall@100 | R-Precision |
|--------|-----|------|---------|------------|-------------|
| BM25 Flattened (baseline) | 0.1832 | 0.3843 | 0.3852 | 0.3777 | 0.2509 |
| BM25 Separate Fields (unweighted) | 0.1603 | 0.3631 | 0.3685 | 0.3418 | 0.2273 |
| BM25F (field-weighted) | 0.1865 | 0.4012 | 0.3997 | 0.3804 | 0.2528 |
| BM25F + Phrase & Proximity ⭐ | **0.1961** | **0.4040** | **0.4033** | **0.3938** | 0.2655 |
| BM25F + Phrase/Prox + WordNet | 0.1958 | 0.4040 | 0.4014 | 0.3936 | **0.2657** |
| BM25F + Phrase/Prox + WordNet + Neural Rerank | 0.1795 | 0.3795 | 0.3794 | 0.3936 | 0.2449 |

**Key findings:**

- Phrase and proximity modelling produced the largest single performance gain
- WordNet expansion improved recall without significantly changing MAP
- Neural reranking did not improve MAP — likely due to truncated document representations limiting cross-encoder signal

---

## Key Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit GUI |
| `config.py` | Configuration and hyperparameters |
| `build_index.py` | Index construction (SPIMI) |
| `parse_docs.py` | SGML document parsing |
| `preprocess.py` | Text normalisation pipeline |
| `rank.py` | BM25F scoring + phrase/proximity |
| `query_expand.py` | WordNet expansion |
| `search.py` | CLI interface |
| `evaluate.py` | Evaluation pipeline |
| `metrics.py` | MAP, P@10, nDCG, Recall, R-Precision |
| `index_store.py` | Index access layer |
| `topics_parser.py` | TREC topic parsing |
| `qrels_parser.py` | Qrels parsing |

---

## Authors

- Blazej Olszta
- Muhamad Husaam Ateeq
- Max Monaghan
- Sulaiman Bhatti

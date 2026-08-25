if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research_extension.config import FROZEN_CONFIG
    from research_extension.datasets import load_dataset
    from research_extension.dense import DenseIndex, SentenceTransformerEncoder
    from research_extension.hybrid import fuse_runs
    from research_extension.io import save_run
    from research_extension.lexical import LexicalIndex, retrieve_all

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("scifact", "nfcorpus"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    root = Path(args.artifact_dir) / args.dataset
    documents, queries, _ = load_dataset(args.dataset, "test", args.data_dir)
    lexical = LexicalIndex.load(root / "lexical_index.pkl")
    encoder = SentenceTransformerEncoder(
        FROZEN_CONFIG.dense_model,
        revision=FROZEN_CONFIG.dense_revision,
        device=args.device,
        batch_size=FROZEN_CONFIG.dense_batch_size,
    )
    dense = DenseIndex.load(root / "dense_index.npz")
    lexical_run = retrieve_all(
        lexical,
        queries,
        "bm25f_phrase_proximity",
        FROZEN_CONFIG.lexical,
        FROZEN_CONFIG.retrieval_depth,
    )
    dense_run = dense.search_many(queries, encoder, FROZEN_CONFIG.retrieval_depth)
    hybrid_run = fuse_runs(
        [lexical_run, dense_run],
        rrf_k=FROZEN_CONFIG.rrf_k,
        top_k=FROZEN_CONFIG.retrieval_depth,
    )
    for name, run in (
        ("bm25f_phrase_proximity", lexical_run),
        ("dense", dense_run),
        ("hybrid_rrf", hybrid_run),
    ):
        save_run(run, root / "runs" / f"{name}.json")
        print(f"saved {name}")

if __name__ == "__main__":
    import argparse
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research_extension.datasets import load_dataset
    from research_extension.lexical import LexicalIndex

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("scifact", "nfcorpus"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    documents, _, _ = load_dataset(args.dataset, args.split, args.data_dir)
    start = time.perf_counter()
    index = LexicalIndex.build(documents)
    path = Path(args.output_dir) / args.dataset / "lexical_index.pkl"
    index.save(path)
    print(f"{path}: {len(documents)} documents in {time.perf_counter() - start:.3f}s")

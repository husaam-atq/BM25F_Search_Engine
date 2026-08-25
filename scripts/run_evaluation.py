if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research_extension.datasets import load_dataset
    from research_extension.evaluation import evaluate_run
    from research_extension.io import load_run

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("scifact", "nfcorpus"))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    args = parser.parse_args()
    _, _, qrels = load_dataset(args.dataset, "test", args.data_dir)
    run_dir = Path(args.artifact_dir) / args.dataset / "runs"
    summary = {}
    for path in sorted(run_dir.glob("*.json")):
        summary[path.stem] = evaluate_run(qrels, load_run(path))[0]
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research_extension.reporting import generate_figures

    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()
    generate_figures(args.artifact_dir, args.output_dir)
    print(f"figures written to {args.output_dir}")

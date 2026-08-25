if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from research_extension.experiments import main
    sys.argv.insert(1, "full")
    main()

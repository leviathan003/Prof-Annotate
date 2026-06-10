"""
profannotate/app.py
Package entry point called by the `profannotate` CLI command.
"""

import sys
from pathlib import Path


def main() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "main",
        Path(__file__).resolve().parent.parent / "main.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.exit(mod.main())


if __name__ == "__main__":
    main()

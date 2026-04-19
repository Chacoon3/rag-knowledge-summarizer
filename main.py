from __future__ import annotations

import sys
from pathlib import Path
from local_rag.cli import main as cli_main


def main() -> None:
    project_root = Path(__file__).resolve().parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    cli_main()


if __name__ == "__main__":
    main()

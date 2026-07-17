from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.product.operations import create_sqlite_backup, prune_sqlite_backups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an integrity-checked VitalsSight SQLite evidence backup."
    )
    parser.add_argument(
        "--db",
        default=os.getenv("VITALSSIGHT_DB_PATH", "runtime/vitalsight_console.db"),
        help="Evidence database path.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/controlled_trial_backups",
        help="Dedicated backup directory; raw upload directories are never copied.",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=14,
        help="Number of newest backup pairs to retain in the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = create_sqlite_backup(Path(args.db), Path(args.output_dir))
    result["pruned_backups"] = prune_sqlite_backups(args.output_dir, keep=args.keep)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

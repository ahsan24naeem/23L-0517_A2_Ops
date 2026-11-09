"""Point the local MLflow store (mlflow.db + mlruns/) at this folder after moving the project.

MLflow writes ABSOLUTE artifact paths into the SQLite database. The runs were
created in the training workspace (/home/claude/23L-0517_A2_Ops); after copying
the project to another machine those paths must be rewritten to the new location.
Usage: python scripts/relocate_mlflow_store.py [--db mlflow.db]
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "mlflow.db"))
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    locs = [r[0] for r in con.execute("select artifact_location from experiments")]
    old = None
    for loc in locs:
        m = re.match(r"(?:file://)?(.*)/mlruns/\d+$", loc or "")
        if m:
            old = m.group(1)
            break
    new = str(ROOT)
    if not old or old == new:
        print("nothing to do:", old)
        return
    updates = [("experiments", "artifact_location"), ("runs", "artifact_uri"),
               ("model_versions", "source"), ("model_versions", "storage_location")]
    for table, col in updates:
        try:
            n = con.execute(f"update {table} set {col} = replace({col}, ?, ?)", (old, new)).rowcount
            print(f"{table}.{col}: {n} rows")
        except sqlite3.OperationalError as e:
            print(f"skip {table}.{col}: {e}")
    con.commit()
    print(f"relocated {old} -> {new}")


if __name__ == "__main__":
    main()

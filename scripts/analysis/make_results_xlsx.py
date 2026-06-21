"""
Build a human-friendly results.xlsx from the master results.csv, with one sheet
per task (deeploc / meltome). The CSV stays the machine-readable source for the
plotting scripts; this workbook is just for eyeballing.

Each sheet drops columns that are empty for that task (e.g. the loc sheet has no
Spearman columns) and sorts by layer, method, d_c, seed.

Usage:
    python scripts/make_results_xlsx.py                       # results/results.csv -> results/results.xlsx
    python scripts/make_results_xlsx.py --csv path --out path
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents
                    if (p / 'configs').is_dir() and (p / 'models').is_dir())

# task value in the CSV -> sheet name
TASK_SHEETS = [('loc', 'deeploc'), ('meltome', 'meltome')]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, default=PROJECT_ROOT / 'results' / 'results.csv')
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f'{args.csv} not found. Run scripts/collect_results.py first.')
    out_path = args.out or args.csv.with_suffix('.xlsx')

    df = pd.read_csv(args.csv)
    sort_cols = [c for c in ('layer', 'method', 'proj_dim', 'seed') if c in df.columns]

    try:
        writer = pd.ExcelWriter(out_path, engine='openpyxl')
    except ModuleNotFoundError as exc:  # openpyxl missing
        raise SystemExit('Need openpyxl for .xlsx output: pip install openpyxl') from exc

    with writer:
        for task, sheet in TASK_SHEETS:
            sub = df[df['task'] == task].copy()
            if sub.empty:
                continue
            sub = sub.dropna(axis=1, how='all')              # drop all-empty cols for this task
            present = [c for c in sort_cols if c in sub.columns]
            for c in present:                                 # stable sort with empties first
                sub[c] = sub[c].fillna('' if sub[c].dtype == object else -1)
            sub = sub.sort_values(present)
            sub.to_excel(writer, sheet_name=sheet, index=False)
            print(f'sheet "{sheet}": {len(sub)} rows, {len(sub.columns)} cols')

    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()

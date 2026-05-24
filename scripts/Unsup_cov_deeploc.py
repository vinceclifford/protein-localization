#!/usr/bin/env python3
"""
Run one unsupervised covariance-pooling experiment.

Pipeline:
    1. Train L/R unsupervised with train_cov_unsup_deeploc.py.
    2. Create a downstream cov_unsup config pointing to the saved L/R checkpoint.
    3. Train subcellular localization with frozen L/R.

Usage:
    python scripts/Unsup_cov_deeploc.py

Smoke test:
    python scripts/Unsup_cov_deeploc.py \
        --dc 2 \
        --seed 123 \
        --tag smoke_cov_unsup \
        --pretrain-epochs 1 \
        --pretrain-batch-size 1 \
        --pretrain-max-length 512 \
        --pretrain-smoke-max-batches 2 \
        --train-num-epochs 1 \
        --train-batch-size 2 \
        --train-max-length 512 \
        --smoke

Full-ish run:
    python scripts/Unsup_cov_deeploc.py \
        --dc 32 \
        --seed 123 \
        --tag loc_cov_unsup_dc32_seed123
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRETRAIN_SCRIPT = "train_cov_unsup_deeploc.py"
DOWNSTREAM_SCRIPT = "train_subcellular_localization.py"

# Use the working supervised cov config as the DATA source for pretraining.
PRETRAIN_BASE_CONFIG = PROJECT_ROOT / "configs" / "subcellular_localization" / "cov.yaml"

# Use your cov_unsup config as the downstream template.
DOWNSTREAM_BASE_CONFIG = PROJECT_ROOT / "configs" / "subcellular_localization" / "cov_unsup.yaml"


def run_cmd(cmd, log_path: Path) -> int:
    print(f'  → {" ".join(str(c) for c in cmd)}')
    print(f"    log: {log_path}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
            logf.flush()

        proc.wait()

    return proc.returncode


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(cfg: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def make_downstream_config(
    base_config: Path,
    output_config: Path,
    seed: int,
    dc: int,
    experiment_name: str,
    checkpoint_path: Path,
    smoke: bool,
    train_num_epochs: int | None,
    train_batch_size: int | None,
    train_max_length: int | None,
) -> None:
    cfg = load_yaml(base_config)

    cfg["seed"] = seed
    cfg["experiment_name"] = experiment_name

    # Smoke-safe defaults. For full runs, leave existing config values unless CLI overrides.
    if smoke:
        cfg["num_epochs"] = 1
        cfg["batch_size"] = 2
        cfg["patience"] = 1
        cfg["log_iterations"] = 1
        cfg["eval_on_test"] = False

        # Use train as validation for smoke testing to avoid local val H5/FASTA mismatch.
        cfg["val_embeddings"] = cfg["train_embeddings"]
        cfg["val_remapping"] = cfg["train_remapping"]

    if train_num_epochs is not None:
        cfg["num_epochs"] = train_num_epochs
    if train_batch_size is not None:
        cfg["batch_size"] = train_batch_size
    if train_max_length is not None:
        cfg["max_length"] = train_max_length

    # These were needed in your local DeepLoc setup.
    cfg["key_format"] = cfg.get("key_format", "fasta_descriptor") or "fasta_descriptor"
    cfg["embedding_mode"] = cfg.get("embedding_mode", "lm") or "lm"

    if "model_parameters" not in cfg or cfg["model_parameters"] is None:
        cfg["model_parameters"] = {}

    cfg["model_parameters"]["pooling"] = "cov_unsup"
    cfg["model_parameters"]["proj_dim"] = dc
    cfg["model_parameters"]["cov_unsup_checkpoint"] = str(checkpoint_path)
    cfg["model_parameters"]["freeze_cov_projections"] = True

    if smoke:
        cfg["model_parameters"]["hidden_dim"] = 4
        cfg["model_parameters"]["dropout"] = 0.0
        cfg["model_parameters"]["n_hidden_layers"] = 0

    save_yaml(cfg, output_config)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dc", type=int, default=32)
    parser.add_argument("--tag", default=None)

    parser.add_argument("--pretrain-base-config", type=Path, default=PRETRAIN_BASE_CONFIG)
    parser.add_argument("--downstream-base-config", type=Path, default=DOWNSTREAM_BASE_CONFIG)

    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--pretrain-batch-size", type=int, default=8)
    parser.add_argument("--pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--pretrain-patience", type=int, default=10)
    parser.add_argument("--pretrain-max-length", type=int, default=None)
    parser.add_argument("--pretrain-smoke-max-batches", type=int, default=None)

    parser.add_argument("--train-num-epochs", type=int, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--train-max-length", type=int, default=None)

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke-test-safe downstream config edits.",
    )

    args = parser.parse_args()

    tag = args.tag or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "sweeps" / tag

    config_dir = run_dir / "configs"
    log_dir = run_dir / "logs"
    pretrain_dir = run_dir / "cov_unsup_pretrained"

    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    exp_name = f"loc_cov_unsup_dc{args.dc}_seed{args.seed}"

    print(f"Run dir: {run_dir}")
    print(f"Experiment: {exp_name}")
    print(f"seed: {args.seed}")
    print(f"d_c: {args.dc}")
    print(f"smoke: {args.smoke}")
    print()

    # ------------------------------------------------------------------
    # Stage 1: pretrain L/R
    # ------------------------------------------------------------------
    pretrain_log = log_dir / f"{exp_name}_pretrain.log"

    pretrain_cmd = [
        sys.executable,
        "-u",
        PRETRAIN_SCRIPT,
        "--config",
        args.pretrain_base_config,
        "--seed",
        args.seed,
        "--proj_dim",
        args.dc,
        "--num_epochs",
        args.pretrain_epochs,
        "--batch_size",
        args.pretrain_batch_size,
        "--lr",
        args.pretrain_lr,
        "--patience",
        args.pretrain_patience,
        "--embedding_mode",
        "lm",
        "--output_dir",
        pretrain_dir,
    ]

    if args.pretrain_max_length is not None:
        pretrain_cmd.extend(["--max_length", args.pretrain_max_length])

    if args.pretrain_smoke_max_batches is not None:
        pretrain_cmd.extend(["--smoke_max_batches", args.pretrain_smoke_max_batches])

    t0 = datetime.datetime.now()
    rc_pretrain = run_cmd(pretrain_cmd, pretrain_log)
    pretrain_minutes = (datetime.datetime.now() - t0).total_seconds() / 60.0

    # Your current pretrainer saves this filename.
    raw_checkpoint = pretrain_dir / f"cov_unsup_deeploc_dc{args.dc}.pt"

    # Optional fallback if you implemented latest saving.
    latest_checkpoint = pretrain_dir / f"cov_unsup_deeploc_dc{args.dc}_latest.pt"

    if not raw_checkpoint.exists() and latest_checkpoint.exists():
        raw_checkpoint = latest_checkpoint

    summary_lines = []

    if rc_pretrain != 0:
        summary_lines.append(
            f"{exp_name}  pretrain_rc={rc_pretrain}  train_rc=SKIP  "
            f"pretrain_time={pretrain_minutes:.1f}min"
        )
        save_summary(run_dir, summary_lines)
        print("\nPretraining failed; downstream training skipped.")
        sys.exit(rc_pretrain)

    if not raw_checkpoint.exists():
        summary_lines.append(
            f"{exp_name}  pretrain_rc=0  train_rc=SKIP  "
            f"ERROR=no_checkpoint_found"
        )
        save_summary(run_dir, summary_lines)
        print(f"\nERROR: expected checkpoint not found in {pretrain_dir}")
        print(f"Looked for: {pretrain_dir / f'cov_unsup_deeploc_dc{args.dc}.pt'}")
        print(f"Also tried: {pretrain_dir / f'cov_unsup_deeploc_dc{args.dc}_latest.pt'}")
        sys.exit(1)

    # Rename checkpoint to include seed so runs do not overwrite each other.
    checkpoint_path = pretrain_dir / f"cov_unsup_deeploc_dc{args.dc}_seed{args.seed}.pt"
    raw_checkpoint.replace(checkpoint_path)

    # ------------------------------------------------------------------
    # Stage 2: write downstream config
    # ------------------------------------------------------------------
    downstream_config = config_dir / f"{exp_name}.yaml"

    make_downstream_config(
        base_config=args.downstream_base_config,
        output_config=downstream_config,
        seed=args.seed,
        dc=args.dc,
        experiment_name=exp_name,
        checkpoint_path=checkpoint_path,
        smoke=args.smoke,
        train_num_epochs=args.train_num_epochs,
        train_batch_size=args.train_batch_size,
        train_max_length=args.train_max_length,
    )

    # ------------------------------------------------------------------
    # Stage 3: train downstream localization model
    # ------------------------------------------------------------------
    train_log = log_dir / f"{exp_name}_train.log"

    train_cmd = [
        sys.executable,
        "-u",
        DOWNSTREAM_SCRIPT,
        "--config",
        downstream_config,
    ]

    t1 = datetime.datetime.now()
    rc_train = run_cmd(train_cmd, train_log)
    train_minutes = (datetime.datetime.now() - t1).total_seconds() / 60.0

    total_minutes = pretrain_minutes + train_minutes

    summary_lines.append(
        f"{exp_name}  pretrain_rc={rc_pretrain}  train_rc={rc_train}  "
        f"pretrain_time={pretrain_minutes:.1f}min  "
        f"train_time={train_minutes:.1f}min  "
        f"total_time={total_minutes:.1f}min  "
        f"checkpoint={checkpoint_path}"
    )

    save_summary(run_dir, summary_lines)

    print("\nRun complete.")
    print("\n".join(summary_lines))

    if rc_train != 0:
        sys.exit(rc_train)


def save_summary(run_dir: Path, lines: list[str]) -> None:
    with open(run_dir / "summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
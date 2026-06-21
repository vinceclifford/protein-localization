"""
Run the bootstrap test evaluation against the best-val checkpoint of an existing run.

Usage:
    python scripts/eval_run.py runs/PoolingFFN_mean_pooling_ProtT5_12-05_18-00-15
    python scripts/eval_run.py        # picks the most recent run automatically
"""
import argparse
import glob
import os
import sys
from types import SimpleNamespace

_root = os.path.abspath(__file__)
while _root != os.path.dirname(_root):
    _root = os.path.dirname(_root)
    if os.path.isdir(os.path.join(_root, 'configs')) and os.path.isdir(os.path.join(_root, 'models')):
        break
sys.path.insert(0, _root)

import yaml
import torch
from torch.optim import Adam
from torchvision.transforms import transforms

from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.transforms import SolubilityToInt, ToTensor
from models import *  # makes model classes discoverable via globals()
from models.loss_functions import LocCrossEntropy
from solver import Solver


def find_config_in(run_dir: str) -> str:
    candidates = [f for f in os.listdir(run_dir) if f.endswith('.yaml')]
    if not candidates:
        raise FileNotFoundError(f'no .yaml found in {run_dir}')
    # prefer one that isn't named train_arguments
    preferred = [c for c in candidates if c != 'train_arguments.yaml']
    return os.path.join(run_dir, preferred[0] if preferred else candidates[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', nargs='?', default=None,
                        help='Path to runs/<run>. If omitted, the newest run under runs/ is used.')
    parser.add_argument('--filename', default='test_peek',
                        help='Suffix for evaluation output files (default: test_peek)')
    cli = parser.parse_args()

    run_dir = cli.run_dir or sorted(glob.glob('runs/*'), key=os.path.getmtime)[-1]
    cfg_path = find_config_in(run_dir)
    print(f'run dir : {run_dir}')
    print(f'config  : {cfg_path}')

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    defaults = dict(
        seed=cfg.get('seed', 123),
        optimizer='Adam',
        loss_function='LocCrossEntropy',
        target='loc',
        balanced_loss=False,
        solubility_loss=0,
        unknown_solubility=True,
        max_length=6000,
        embedding_mode='lm',
        n_draws=200,
        min_train_acc=0,
        log_iterations=100,
        checkpoint=run_dir,
        experiment_name=cfg.get('experiment_name', ''),
    )
    defaults.update(cfg)
    defaults['checkpoint'] = run_dir
    defaults['config'] = SimpleNamespace(name=cfg_path)
    args = SimpleNamespace(**defaults)

    tfm = transforms.Compose([SolubilityToInt(), ToTensor()])
    train_set = EmbeddingsLocalizationDataset(
        args.train_embeddings, args.train_remapping,
        args.unknown_solubility, key_format=args.key_format,
        max_length=args.max_length, embedding_mode=args.embedding_mode, transform=tfm,
    )
    test_set = EmbeddingsLocalizationDataset(
        args.test_embeddings, args.test_remapping,
        args.unknown_solubility, key_format=args.key_format,
        embedding_mode=args.embedding_mode, transform=tfm,
    )

    model_cls = globals()[args.model_type]
    model = model_cls(embeddings_dim=test_set[0][0].shape[-1], **args.model_parameters)
    solver = Solver(model, args, Adam, LocCrossEntropy, weight=train_set.class_weights)
    solver.evaluation(test_set, filename=cli.filename)

    out = os.path.join(run_dir, f'evaluation_{cli.filename}.txt')
    print(f'\nwrote {out}\n')
    with open(out) as f:
        print(f.read())


if __name__ == '__main__':
    main()

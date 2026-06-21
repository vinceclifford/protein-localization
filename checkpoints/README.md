# checkpoints

Trained artifacts for the second-order pooling experiments. Both the tiny frozen
projections (`projections/`, ~270 KB each) and the full trained probe heads
(`heads/`, ~370 MB total) are tracked in git, so a clone reproduces the
figures/tables in `results/` without re-training. The large per-residue `.h5`
embeddings are the only inputs distributed separately (Google Drive).

All embeddings are ProtT5-XL **layer-24** (final layer) per-residue features.
Canonical analysis seeds are **657 / 921 / 969**.

## Layout

```
checkpoints/
├── projections/                      frozen unsupervised covariance projections (L, R)
│   ├── union/                        cov_unsup_dc{8,16,24,32,48}.pt
│   │                                 L,R trained once on the UNION of both train splits;
│   │                                 reused frozen by every downstream cov_unsup head.
│   └── compare/                      projection-dataset comparison (Part A), d_c = 48
│       ├── {union,deeploc,meltome}_seed{657,921,969}/   each: cov_unsup_dc48.pt
│       └── reconstruction_summary.csv                   rel_err mean ± std per dataset
│
└── heads/                            trained downstream probes (PoolingFFN)
    ├── unsupervised/{deeploc,meltome}/   cov_unsup × d_c{8,16,24,32,48} × 3 seeds
    ├── supervised/{deeploc,meltome}/     mean + {cov,hybrid} × d_c{8,16,24,32,48} × 3 seeds
    └── layer_sweep/{deeploc,meltome}/    {mean,cov,hybrid} × layer{06,12,18,24,Stacked} × 3 seeds
```

## Run-directory naming

Each head run directory follows:

```
PoolingFFN_<task>[_layer<L>]_<method>[_dc<N>]_seed<S>_<DD-MM_HH-MM-SS>[_pretrained]
```

- `<task>`     — `loc` (DeepLoc subcellular localization) or `meltome` (FLIP human_cell).
- `layer<L>`   — only in the layer sweep: `06/12/18/24/Stacked` (`Stacked` = all layers concatenated).
- `<method>`   — `mean`, `cov`, `hybrid`, or `cov_unsup` (frozen unsupervised projections).
- `dc<N>`      — projection dimension d_c (absent for `mean`).
- `seed<S>`    — RNG seed (head init + shuffle + dropout); analysis uses 657/921/969.
- `_pretrained` — suffix on `cov_unsup` heads, marking that L/R were loaded frozen from `projections/union/`.

Each run dir holds `train_arguments.yaml`, the model checkpoint, and
`evaluation_test_set_after_train.txt` (parsed by `scripts/analysis/collect_results.py`).
Training logs and TensorBoard event files are excluded to keep the archive small.

## Notes

- **projections vs heads.** `projections/` holds the *unsupervised* L/R (the
  pretraining output, task-agnostic). `heads/` holds the *supervised* probe
  trained on top of a pooling method. The `cov_unsup` heads consume
  `projections/union/cov_unsup_dc<N>.pt` frozen.
- **Seed coverage.** All groups are 3 seeds (657/921/969). The meltome
  `supervised/hybrid` dirs additionally retain legacy seeds (123/456/789);
  restrict to the common 657/921/969 when comparing methods. The deeploc
  `layer_sweep/Stacked` `cov`/`hybrid` cells have 2 of 3 seeds.
- `projections/compare/` exists only to show the projections are ~seed-independent
  and that training on the union costs ~nothing vs per-task (see `results/`).

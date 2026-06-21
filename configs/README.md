# configs

YAML configs consumed by the training entry points (`train_subcellular_localization.py`,
`train_meltome.py`, `train_cov_unsup.py`) and by the sweep drivers in
`scripts/experiments/`, which clone a base config and override fields per run.

## Layout

```
configs/
├── subcellular_localization/   downstream DeepLoc (Q10 classification) head configs
│   └── mean / cov / cov_unsup / hybrid / la / la_cov .yaml
├── meltome/                    downstream Meltome (Tm regression) head configs
│   └── mean / cov / cov_unsup / hybrid / la / la_cov / ffn / light_attention .yaml
└── cov_unsup_pretrain/         unsupervised covariance-projection pretraining
    ├── union.yaml              train L,R on the UNION of both task train splits (used downstream)
    ├── deeploc.yaml            train on DeepLoc only      (Part A comparison)
    └── meltome.yaml            train on Meltome only      (Part A comparison)
```

## Downstream head configs (`subcellular_localization/`, `meltome/`)

One per **pooling method**. Key fields:

- `model_type` — `PoolingFFN` (mean/cov/cov_unsup/hybrid) or `LightAttention*` (la/la_cov).
- `model_parameters.pooling` — `mean` / `cov` / `cov_unsup` / `hybrid`.
- `model_parameters.proj_dim` — d_c (covariance bottleneck; absent for `mean`).
- For `cov_unsup`: `freeze_cov_projections: true` and a checkpoint path that the
  sweep driver rewrites per d_c to point at `checkpoints/projections/union/`.
- Data paths point at the layer-24 H5 files; loaders use `key_format: hash`.

The sweep drivers override `seed`, `proj_dim`, embedding paths, and the
`cov_unsup` checkpoint per run — so the committed YAMLs are the *base* recipes.

## Pretraining configs (`cov_unsup_pretrain/`)

Consumed by `train_cov_unsup.py` / `scripts/experiments/run_cov_unsup_pretrain.py`.
They list the **train-only** embedding sources to learn the projections `L, R`
by covariance reconstruction (no labels). `union.yaml` is the one used for the
frozen projections every downstream `cov_unsup` head reuses; `deeploc.yaml` and
`meltome.yaml` exist only for the projection-dataset comparison (Part A).

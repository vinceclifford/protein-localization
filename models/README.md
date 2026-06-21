# models

Model definitions. `__init__.py` auto-imports every module in this package and
hoists its classes into the package namespace, so a config can name a model by
string (`model_type: PoolingFFN`) and the solver resolves it via `globals()`.

| file | class(es) | role |
|---|---|---|
| `pooling_ffn.py` | `PoolingFFN` | The main model. A pooling layer (`mean` / `cov` / `cov_unsup` / `hybrid`) over per-residue embeddings, then an FFN head. `cov`/`hybrid` learn projections `L, R ∈ ℝ^{d×d_c}` and pool the bilinear covariance `C = (1/L)(XL)ᵀ(XR)`; `cov_unsup` loads frozen, pretrained `L, R`. |
| `light_attention.py` | `LightAttention` | LightAttention baseline (`la`): softmax attention pooling over residues. |
| `light_attention_cov.py` | `LightAttentionCov` | LightAttention with a covariance branch (`la_cov`). |
| `ffn.py` | `FFN` | Plain feed-forward head over a fixed-size input. |
| `loss_functions.py` | `JointCrossEntropy`, `LocCrossEntropy`, `MeltomeMSELoss` | Task losses (classification / regression). |

## Pooling methods (in `PoolingFFN`)

| `pooling` | output dim | learns L,R? | notes |
|---|---|---|---|
| `mean` | d (1024) | – | first-order baseline |
| `cov` | d_c² | yes (supervised) | bilinear covariance bottleneck |
| `cov_unsup` | d_c² | no (frozen) | projections pretrained by `train_cov_unsup.py`, loaded frozen |
| `hybrid` | d + d_c² | yes | concatenate mean and covariance |

The frozen `cov_unsup` projections come from `checkpoints/projections/union/`
(see the unsupervised pipeline in the top-level `README.md`, Experiment 2).

> The original DeepLoc fork's large `models/legacy/` tree has been removed; the
> auto-loader explicitly skipped it, so nothing here depended on it.

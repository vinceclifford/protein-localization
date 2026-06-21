# utils

| file | role |
|---|---|
| `general.py` | Shared constants and helpers used across the pipeline: `LOCALIZATION` (ordered Q10 class list), `AMINO_ACIDS`, `padded_permuted_collate`, and the TensorBoard / confusion-matrix / per-class-accuracy plotting helpers. |

Imported by `solver.py` and `datasets/embeddings_localization_dataset.py`, so
this is a real runtime dependency — not optional tooling.

> The original fork's exploratory utilities (`data_stats`, `explore_embeddings`,
> `extract_embeddings`, `parse_pssm_txt_file`, `preprocess`) were removed; none
> were imported by the active pipeline.

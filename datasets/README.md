# datasets

PyTorch `Dataset` wrappers around the per-residue ProtT5 H5 embeddings.

| file | class | role |
|---|---|---|
| `embeddings_localization_dataset.py` | `EmbeddingsLocalizationDataset` | DeepLoc: yields `(embedding[L, d], localization_label, metadata)`. Reads the Q10 class + solubility from the FASTA description. |
| `embeddings_meltome_dataset.py` | `EmbeddingsMeltomeDataset` | Meltome: yields `(embedding[L, d], Tm_target, metadata)`. Reads `TARGET=` from the FASTA description. |
| `transforms.py` | `ToTensor`, `SolubilityToInt`, … | Sample transforms applied by the loaders. |

## Key lookup

Each dataset pairs an H5 embedding file with a FASTA. `key_format` controls how
a FASTA record maps to an H5 key:

- `hash` — the H5 key is the record id verbatim (used by all `*_layer24.h5` files).
- `fasta_descriptor` — the descriptor with `.`/`/` replaced by `_`.

All current configs use `key_format: hash`. Per-residue tensors are
`[L, d]` (variable length `L`, `d = 1024` for ProtT5-XL); variable-length
batches are padded by the collate functions in the solvers.

`LOCALIZATION` (the ordered Q10 class list) and `AMINO_ACIDS` live in
`utils/general.py` and are imported here.

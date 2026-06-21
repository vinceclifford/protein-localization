# data_files

The per-residue ProtT5 / ProtX embeddings (`*.h5`) are large (tens of GB) and are
**not** tracked in git. This folder holds the small text artifacts (remapped FASTAs)
and is where the `.h5` embeddings must be placed to run the pipeline.

## Required files

DeepLoc (subcellular localization):
```
deeploc_our_train_set_layer24.h5   deeploc_our_train_set_remapped.fasta
deeploc_our_val_set_layer24.h5     deeploc_our_val_set_remapped.fasta
deeploc_test_set_layer24.h5        deeploc_test_set_remapped.fasta
```

Meltome (FLIP human_cell):
```
flip_meltome/prepared/human_cell/human_cell_train_layer24.h5  + *_remapped.fasta
flip_meltome/prepared/human_cell/human_cell_val_layer24.h5    + *_remapped.fasta
flip_meltome/prepared/human_cell/human_cell_test_layer24.h5   + *_remapped.fasta
```

`*_layer24.h5` are the final-layer (layer-24) ProtT5 embeddings; keys are integer
indices, so the dataset loaders use `key_format: hash`.

## How to obtain

- **Download** the `.h5` files from the project Google Drive (link in the submission
  notes) and place them at the paths above, **or**
- **Regenerate** from the FASTAs with the provided embedding script (ProtT5, GPU/MPS):
  ```bash
  python scripts/data/embed_layers_h5.py --fasta <input.fasta> --layers 24 --device mps
  ```
  This writes `<stem>_layer24.h5` and `<stem>_remapped.fasta` alongside the input.

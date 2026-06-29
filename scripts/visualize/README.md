# Unified Covariance Visualization Guide

This workflow provides two command-line tools:

- `scripts/visualize/visualize_covariance_sample.py`: one protein at a time.
- `scripts/visualize/visualize_covariance_dataset.py`: dataset-level statistical heatmaps.

Both tools accept either a run directory containing `checkpoint.pt` and
`train_arguments.yaml`, or the path to `checkpoint.pt` itself. They support the
project's `cov`, `hybrid`, `cov_n`, and `hybrid_n` pooling checkpoints.

## Requirements

The Python environment must provide PyTorch, NumPy, pandas, h5py, matplotlib,
requests, transformers, sentencepiece, and PyYAML. Direct sequence input uses
the local ProtT5 model under `embedder_model` by default. A missing local model
causes the Hugging Face loader to attempt a download.

## 1. Single-Protein Visualization

### UniProt ID input

This mode downloads or reads the cached canonical sequence and annotations:

```bash
python scripts/visualize/visualize_covariance_sample.py \
  PoolingFFN_loc_cov_dc32_seed969_21-05_07-59-09 \
  --uniprot-id P98088 \
  --device cuda \
  --out-dir figures/unified_sample
```

### Raw sequence input

```bash
python scripts/visualize/visualize_covariance_sample.py \
  runs/MY_RUN \
  --sequence "MKT...VLA" \
  --device cuda
```

Raw sequence alone produces covariance and sequence-window plots. Add a
matching UniProt ID to obtain feature alignment:

```bash
python scripts/visualize/visualize_covariance_sample.py \
  runs/MY_RUN \
  --sequence "MKT...VLA" \
  --uniprot-id Q9Y375 \
  --device cuda
```

The supplied sequence is used for inference; UniProt coordinates are mapped to
it when the sequence is exact or one sequence is a subsequence of the other.

### Reuse an existing embedding

This avoids running ProtT5 and is useful for exact reproducibility:

```bash
python scripts/visualize/visualize_covariance_sample.py \
  PoolingFFN_loc_cov_dc32_seed969_21-05_07-59-09 \
  --uniprot-id P98088 \
  --embedding-h5 data_files/deeploc_test_set.h5 \
  --embedding-key 0 \
  --device cpu
```

The sequence and embedding must have identical residue lengths.

### Main options

| Option | Default | Meaning |
|---|---:|---|
| `--top-pairs` | 5 | Covariance entries used for residue significance |
| `--pair-sign` | `positive` | Select positive, negative, or largest-absolute entries |
| `--top-positions` | 6 | Local windows shown for each selected entry |
| `--window-radius` | 12 | Residues shown on each side of a peak |
| `--top-features` | 10 | Highest-ranked UniProt feature instances |
| `--uniprot-cache` | `figures/unified_uniprot_cache` | UniProt JSON cache |

### Outputs

Each protein receives its own output directory:

- `covariance_matrix.png` and `.npy`: compressed matrix; selected entries are numbered.
- `significant_entries.png`: ranked selected-entry values and projection coordinates.
- `pair_XX_L*_R*.png`: strongest residue windows for each selected entry.
- `significant_pairs.csv`: selected covariance coordinates and values.
- `significant_sequence_windows.csv`: peak positions, residues, and windows.
- `residue_significance.csv`: score at every sequence position.
- `uniprot_feature_alignment_top10.png`: significance curve and best feature tracks.
- `uniprot_feature_matches.csv`: all mapped feature instances and overlap metrics.
- `metadata.json`: checkpoint, pooling method, and sequence-mapping status.

Feature instances are ranked by AP enrichment:

```text
AP enrichment = average precision / annotated residue fraction
```

The CSV also reports top-1% and top-5% precision, recall, and enrichment. Very
short annotations can obtain high enrichment from one exact peak, so always
interpret their annotation length and recall together with the rank.

## 2. Dataset Statistical Visualization

The minimum inputs are a checkpoint and an embedding H5:

```bash
python scripts/visualize/visualize_covariance_dataset.py \
  PoolingFFN_loc_cov_dc32_seed969_21-05_07-59-09 \
  data_files/deeploc_test_set.h5 \
  --device cuda \
  --out-dir figures/unified_deeploc_test
```

The script matches the H5 path to `train_arguments.yaml` and automatically
loads the corresponding remapping FASTA. For a custom H5, specify it directly:

```bash
python scripts/visualize/visualize_covariance_dataset.py \
  runs/MY_RUN custom_embeddings.h5 \
  --fasta custom_remapped.fasta \
  --device cuda
```

An H5 contains embeddings but not the amino-acid sequence, class, or UniProt
accession. Therefore, the remapping FASTA is required even when it is inferred
automatically.

### Meltome human-cell example

The following command runs the complete statistical analysis on the existing
Meltome human-cell test H5 with the `cov`, `dc=32`, seed-42 checkpoint used in
`meltome_functional_site_alignment_analysis.md`:

```bash
python scripts/visualize/visualize_covariance_dataset.py \
  runs/PoolingFFN_meltome_mixed_cov_pooling_ProtT5_18-05_23-37-20 \
  data_files/flip_meltome/prepared/human_cell/human_cell_test.h5 \
  --mapping-csv figures/meltome_uniprot_mapping_check/human_cell_test_uniprot_mapping.csv \
  --cache-only \
  --cache-fallback-dir figures/meltome_uniprot_site_overlap_cov_dc32_humancell_seed42_top5/uniprot_cache \
  --device cuda \
  --out-dir figures/unified_meltome_human_cell_test
```

The remapping FASTA is inferred as
`data_files/flip_meltome/prepared/human_cell/human_cell_test_remapped.fasta`
from the checkpoint configuration. The mapping CSV uses its native
`fasta_id` and `uniprot_accession` columns; these are accepted directly. With
`--cache-only`, proteins absent from both the primary and fallback caches are
excluded only from UniProt feature alignment, while their AA enrichment still
contributes to the dataset statistics. Remove `--cache-only` to retrieve any
missing UniProt records from the network.

### UniProt accessions

DeepLoc-style headers are parsed automatically. For another dataset, provide a
CSV mapping H5 keys to accessions:

```csv
h5_key,accession
0,P12345
1,Q9Y375
```

```bash
python scripts/visualize/visualize_covariance_dataset.py \
  runs/MY_RUN custom.h5 \
  --fasta custom_remapped.fasta \
  --mapping-csv h5_to_uniprot.csv
```

Use `--cache-only` to prohibit network requests. Additional existing caches can
be supplied repeatedly with `--cache-fallback-dir PATH`.

### Classes

Known DeepLoc labels are inferred from FASTA descriptions. If no recognized
class is present, the heatmap contains one row named `All`. AA CSV files also
contain an `All` aggregate for class-labeled datasets.

### Main options

| Option | Default | Meaning |
|---|---:|---|
| `--aa-top-pairs` | 1 | Positive entries used in AA enrichment |
| `--feature-top-pairs` | 5 | Positive entries summed for feature alignment |
| `--sample-limit` | all | Process only the first N FASTA records |
| `--workers` | 8 | UniProt retrieval concurrency setting |
| `--cache-only` | off | Never request missing UniProt records |
| `--mapping-csv` | none | H5-key to UniProt-accession mapping |

### AA enrichment outputs

- `aa_enrichment_by_class_heatmap.png`: 20 amino acids, grouped chemically.
- `aa_group_enrichment_by_class_heatmap.png`: six chemical groups.
- `aa_enrichment.csv` and `aa_group_enrichment.csv`: complete values and counts.

Each displayed value is a fold change, not a percentage:

```text
fold enrichment = absolute contribution share / sequence abundance share
```

`1.00x` is background expectation, values above 1 are enriched, and values
below 1 are depleted. Chemical groups are KRH, STNQC, DE, AVILM, FWY, and GP.

### UniProt feature outputs

- `official_feature_type_enrichment_heatmap.png`: all official feature types over top
  1%, 5%, 10%, 20%, 50%, and 80% residue thresholds.
- `official_feature_type_class_enrichment_top1.png`: class-specific top-1% enrichment.
- `official_feature_type_class_enrichment_top5.png`: class-specific top-5% enrichment.
- `functional_region_type_enrichment_heatmap.png`: focused view matching the
  earlier functional-region analysis.
- `functional_region_type_class_enrichment_top1.png` and `top5.png`: focused
  class-specific views.
- `uniprot_feature_overlap_per_protein.csv`, `overall.csv`, and `by_class.csv`:
  complete overlap statistics.

### Official Feature Definitions

| Category | Feature | Definition |
|---|---|---|
| Molecule processing | Initiator methionine | The first translated methionine, annotated when its retention or removal is biologically characterized. | 
| Molecule processing | Signal peptide / Signal | N-terminal peptide that directs a nascent protein into the secretory pathway and is usually cleaved after translocation into the ER. |
| Molecule processing | Transit peptide | N-terminal targeting peptide that directs a nuclear-encoded protein to an organelle such as a mitochondrion, plastid, or peroxisome and is often cleaved after import. |
| Molecule processing | Propeptide | Precursor segment removed during maturation or activation of the protein. |
| Molecule processing | Chain | Annotated mature protein chain remaining after precursor processing. | 
| Molecule processing | Peptide | Released or independently functional peptide produced by precursor cleavage. | 
| Membrane topology | Topological domain | Region of a membrane protein exposed to the cytosol, lumen, extracellular space, or an organelle compartment. | 
| Membrane topology | Transmembrane | Segment that spans a biological membrane, usually as an alpha helix. | 
| Membrane topology | Intramembrane | Segment embedded in a membrane without fully crossing the bilayer. |
| Region / domain | Domain | Conserved structural or functional unit that can often fold or function semi-independently. | 
| Region / domain | Repeat | Repeated sequence unit or tandemly recurring structural element. | 
| Region / domain | Calcium binding | Region specifically annotated as participating in calcium-ion binding. | 
| Region / domain | Zinc finger | Compact zinc-coordinating domain, commonly involved in DNA, RNA, or protein binding. | 
| Region / domain | DNA binding | Region experimentally or curator-annotated as binding DNA. | 
| Region / domain | Nucleotide binding | Region involved in binding ATP, GTP, NAD, FAD, or another nucleotide. | 
| Region / domain | Region | Broad UniProt-annotated sequence region with a described structural or functional role. | 
| Region / domain | Coiled coil | Extended helical region characterized by a repeating heptad pattern and inter-helix packing. | 
| Region / domain | Motif | Short conserved sequence pattern associated with a specific function, interaction, modification, or targeting event. | 
| Region / domain | Compositional bias | Region enriched in a restricted subset of amino acids, including low-complexity or residue-biased segments. | 
| Functional site | Active site | Residue directly involved in an enzyme's catalytic mechanism. | 
| Functional site | Binding site | Residue directly involved in binding a ligand, substrate, cofactor, ion, or another molecular partner. | 
| Functional site | Site | Other specifically annotated functional residue that is not represented by a more specialized official site type. | 

The focused functional-region view contains Transit peptide, Signal,
Propeptide, Peptide, Motif, Region, Transmembrane, Intramembrane, and
Topological domain.

For a feature mask and a top-residue threshold:

```text
precision = top residues inside the feature / all selected top residues
background = annotated residues / all residues
enrichment = precision / background
```

`P` in a feature heatmap is the number of proteins with that annotation in the
corresponding dataset or class. Gray cells have no annotated proteins and are
undefined; `0.00x` cells have annotations but no overlap.

## Covariance Attribution

For standard asymmetric covariance pooling:

```text
C[i,j] = (1/L) sum_p proj_L(x_p)[i] * proj_R(x_p)[j]
```

The residue contribution is the summand at position `p`. The significance
score is the sum of its absolute contribution over the selected entries.

For `cov_n`, the matrix is computed between learned sequence slots. Residue
scores use the symmetric marginal contribution through the two selected slots.
This is a faithful local attribution of the slot pair, but unlike standard
`cov` it is not a strictly additive decomposition because each slot pools
multiple residues before the covariance is formed.

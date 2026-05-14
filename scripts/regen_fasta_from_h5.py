"""
Regenerate FASTA files that exactly match the keys in the provided h5 embedding files.

The h5 keys are in the form:   '<accession> <Localization_with_underscore>-<Sol>'
                       (test):  '<accession> <Localization>-<Sol> test'

The dataset (key_format='fasta_descriptor') applies '.' -> '_' and '/' -> '_' to the
FASTA descriptor when looking up the h5 key. So we generate FASTAs with the dotted/
slashed localization form and let the dataset perform the substitution at lookup time.

Any trailing ' test' suffix in the h5 key is preserved verbatim in the FASTA.

Sequences are placeholders ('X' * embedding_length); the model never sees them, they
only feed length/frequency metadata.
"""
import os
import h5py
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


UNDERSCORE_TO_ORIGINAL = {
    'Cell_membrane': 'Cell.membrane',
    'Endoplasmic_reticulum': 'Endoplasmic.reticulum',
    'Golgi_apparatus': 'Golgi.apparatus',
    'Lysosome_Vacuole': 'Lysosome/Vacuole',
}


def restore_localization(loc_underscored: str) -> str:
    return UNDERSCORE_TO_ORIGINAL.get(loc_underscored, loc_underscored)


def regenerate(h5_path: str, fasta_path: str) -> None:
    records = []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            length = f[key].shape[0]
            tokens = key.split(' ')
            accession = tokens[0]
            loc_sol = tokens[1]
            suffix = ' ' + ' '.join(tokens[2:]) if len(tokens) > 2 else ''
            loc_underscored, sol = loc_sol.rsplit('-', 1)
            loc_dotted = restore_localization(loc_underscored)
            description = f'{accession} {loc_dotted}-{sol}{suffix}'

            rec = SeqRecord(Seq('X' * length), id=accession, description=description)
            records.append(rec)

    os.makedirs(os.path.dirname(fasta_path) or '.', exist_ok=True)
    SeqIO.write(records, fasta_path, 'fasta')
    print(f'Wrote {len(records)} records to {fasta_path}')


if __name__ == '__main__':
    regenerate('data_files/deeploc_our_train_set.h5', 'data_files/deeploc_our_train_set.fasta')
    regenerate('data_files/deeploc_our_val_set.h5',   'data_files/deeploc_our_val_set.fasta')
    regenerate('data_files/deeploc_test_set.h5',      'data_files/deeploc_test_set.fasta')

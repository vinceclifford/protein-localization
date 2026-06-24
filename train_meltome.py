import argparse
import sys

import torch
import yaml
from models import *  # For loading classes specified in config
from torch.optim import *  # For loading optimizer specified in config
from torch.utils.data import DataLoader

from datasets.embeddings_meltome_dataset import EmbeddingsMeltomeDataset
from solver_meltome import SolverMeltome, padded_permuted_meltome_collate
from utils.general import seed_all


class ToTensorMeltome():
    def __call__(self, sample):
        embedding, target = sample
        embedding = torch.tensor(embedding).float()
        target = torch.tensor(target).float()
        return embedding, target


def train(args):
    seed_all(args.seed)
    transform = ToTensorMeltome()
    train_set = EmbeddingsMeltomeDataset(args.train_embeddings, args.train_remapping,
                                         max_length=args.max_length, key_format=args.key_format,
                                         embedding_mode=args.embedding_mode,
                                         parti_weights_path=getattr(args, 'train_parti_weights', None),
                                         transform=transform)
    val_set = EmbeddingsMeltomeDataset(args.val_embeddings, args.val_remapping,
                                       key_format=args.key_format, max_length=args.max_length,
                                       embedding_mode=args.embedding_mode,
                                       parti_weights_path=getattr(args, 'val_parti_weights', None),
                                       transform=transform)

    if len(train_set[0][0].shape) == 2:
        collate_function = padded_permuted_meltome_collate
    else:
        collate_function = None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate_function)

    model = globals()[args.model_type](embeddings_dim=train_set[0][0].shape[-1], **args.model_parameters)
    print('trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    solver = SolverMeltome(model, args, globals()[args.optimizer], globals()[args.loss_function],
                           weight=train_set.class_weights)
    solver.train(train_loader, val_loader, eval_data=val_set)

    if args.eval_on_test:
        test_set = EmbeddingsMeltomeDataset(args.test_embeddings, args.test_remapping,
                                            key_format=args.key_format, max_length=args.max_length,
                                            embedding_mode=args.embedding_mode,
                                            parti_weights_path=getattr(args, 'test_parti_weights', None),
                                            transform=transform)
        solver.evaluation(test_set, filename='test_set_after_train')


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=128, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=50, help='stop training after no improvement in this many epochs')
    p.add_argument('--n_draws', type=int, default=200, help='number of times to sample for estimation of stderr')
    p.add_argument('--seed', type=int, default=123, help='seed for reproducibility')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_parameters', type=dict, default={'lr': 1.0e-4},
                   help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint')

    p.add_argument('--model_type', type=str, default='LightAttention', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, default={'output_dim': 1},
                   help='dictionary of model parameters')
    p.add_argument('--proj_dim', type=int, default=None,
                   help='override model_parameters.proj_dim from the command line for pooling ablations')
    p.add_argument('--loss_function', type=str, default='MeltomeMSELoss',
                   help='Classname of one of the loss functions models/loss_functions.py')
    p.add_argument('--max_length', type=int, default=6000, help='maximum length of sequences that will be used for '
                                                               'training when using embedddings of variable length')
    p.add_argument('--embedding_mode', type=str, default='lm',
                   help='type of embedding to use (lm means Language model) [lm, onehot, profile]')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--train_embeddings', type=str, default='data_files/flip_meltome/prepared/human_cell/human_cell_train.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--train_remapping', type=str, default='data_files/flip_meltome/prepared/human_cell/human_cell_train_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--val_embeddings', type=str, default='data_files/flip_meltome/prepared/human_cell/human_cell_val.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--val_remapping', type=str, default='data_files/flip_meltome/prepared/human_cell/human_cell_val_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--test_embeddings', type=str, default='data_files/flip_meltome/prepared/human_cell/human_cell_test.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--test_remapping', type=str, default='data_files/flip_meltome/prepared/human_cell/human_cell_test_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--key_format', type=str, default='hash',
                   help='the formatting of the keys in the h5 file [fasta_descriptor_old, fasta_descriptor, hash]')
    args = p.parse_args()
    cli_seed = args.seed if '--seed' in sys.argv else None
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    if cli_seed is not None:
        args.seed = cli_seed
    if args.proj_dim is not None:
        if args.model_parameters is None:
            args.model_parameters = {}
        args.model_parameters['proj_dim'] = args.proj_dim
    return args


if __name__ == '__main__':
    train(parse_arguments())

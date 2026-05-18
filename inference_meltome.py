import copy
import os
import argparse

import torch
import yaml
import torch.nn as nn
from models import *  # For loading classes specified in config
from torch.optim import *  # For loading optimizer class that was used in the checkpoint

from datasets.embeddings_meltome_dataset import EmbeddingsMeltomeDataset
from solver_meltome import SolverMeltome


class ToTensorMeltome():
    def __call__(self, sample):
        embedding, target = sample
        embedding = torch.tensor(embedding).float()
        target = torch.tensor(target).float()
        return embedding, target


def inference(args):
    transform = ToTensorMeltome()
    data_set = EmbeddingsMeltomeDataset(args.embeddings, args.remapping,
                                        key_format=args.key_format,
                                        max_length=args.max_length,
                                        embedding_mode=args.embedding_mode,
                                        transform=transform)

    model: nn.Module = globals()[args.model_type](embeddings_dim=data_set[0][0].shape[-1], **args.model_parameters)

    solver = SolverMeltome(model, args, globals()[args.optimizer], globals()[args.loss_function])
    return solver.evaluation(data_set, args.output_files_name)


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
    p.add_argument('--checkpoints_list', default=[],
                   help='if there are paths specified here, they all are evaluated')
    p.add_argument('--output_files_name', type=str, default='meltome_inference',
                   help='string that is appended to produced evaluation files in the run folder')
    p.add_argument('--batch_size', type=int, default=128, help='samples that will be processed in parallel')
    p.add_argument('--n_draws', type=int, default=100,
                   help='how often to bootstrap from the dataset for variance estimation')
    p.add_argument('--log_iterations', type=int, default=100, help='log every log_iterations (-1 for no logging)')
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint')
    p.add_argument('--embeddings', type=str, default='data_files/flip_meltome/prepared/human_cell/human_cell_test.h5',
                   help='.h5 or .h5py file with keys fitting the ids in the corresponding fasta remapping file')
    p.add_argument('--remapping', type=str, default='data_files/flip_meltome/prepared/human_cell/human_cell_test_remapped.fasta',
                   help='fasta file with remappings by bio_embeddings for the keys in the corresponding .h5 file')
    p.add_argument('--key_format', type=str, default='hash',
                   help='the formatting of the keys in the h5 file [fasta_descriptor_old, fasta_descriptor, hash]')
    p.add_argument('--max_length', type=int, default=6000)
    p.add_argument('--embedding_mode', type=str, default='lm')
    p.add_argument('--optimizer', type=str, default='Adam')
    p.add_argument('--optimizer_parameters', type=dict, default={'lr': 1.0e-4})
    p.add_argument('--loss_function', type=str, default='MeltomeMSELoss')
    p.add_argument('--model_type', type=str, default='LightAttention')
    p.add_argument('--model_parameters', type=dict, default={'output_dim': 1})
    p.add_argument('--proj_dim', type=int, default=None,
                   help='override model_parameters.proj_dim from the command line')

    args = p.parse_args()
    arg_dict = args.__dict__
    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    if args.proj_dim is not None:
        if args.model_parameters is None:
            args.model_parameters = {}
        args.model_parameters['proj_dim'] = args.proj_dim
    return args


if __name__ == '__main__':
    original_args = copy.copy(parse_arguments())
    spearmans = []
    mses = []
    maes = []
    checkpoints = original_args.checkpoints_list or ([original_args.checkpoint] if original_args.checkpoint else [])
    for checkpoint in checkpoints:
        args = copy.copy(original_args)
        arg_dict = args.__dict__
        arg_dict['checkpoint'] = checkpoint
        data = yaml.load(open(os.path.join(args.checkpoint, 'train_arguments.yaml'), 'r'), Loader=yaml.FullLoader)
        for key, value in data.items():
            if key not in args.__dict__.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
        spearman, mse, mae = inference(args)
        spearmans.append(spearman)
        mses.append(mse)
        maes.append(mae)

    print('checkpoints: \n', checkpoints)
    print('Spearman Rs: \n', spearmans)
    print('MSEs: \n', mses)
    print('MAEs: \n', maes)

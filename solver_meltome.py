import copy
import csv
import inspect
import os
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyaml
import torch
from models import *
from scipy.stats import spearmanr
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from models.loss_functions import MeltomeMSELoss


def padded_permuted_meltome_collate(batch: List[Tuple[torch.Tensor, torch.Tensor, dict]]) -> Tuple[
    torch.Tensor, torch.Tensor, dict]:
    """
    Takes list of meltome samples with variable length embeddings and pads them with zeros.
    Returns embeddings as [batch_size, embeddings_dim, sequence_length].
    """
    embeddings = [torch.as_tensor(item[0]).float() for item in batch]
    targets = torch.stack([torch.as_tensor(item[1]).float() for item in batch]).view(-1)
    metadata = [item[2] for item in batch]
    metadata = torch.utils.data.dataloader.default_collate(metadata)
    embeddings = pad_sequence(embeddings, batch_first=True)
    return embeddings.permute(0, 2, 1), targets, metadata


def _spearman(results: np.ndarray) -> float:
    if len(results) < 2:
        return float('nan')
    rho, _ = spearmanr(results[:, 1], results[:, 0])
    return float(rho)


class SolverMeltome():
    def __init__(self, model, args, optim=torch.optim.Adam, loss_func=MeltomeMSELoss, weight=None, eval=False):
        self.optim = optim(list(model.parameters()), **args.optimizer_parameters)
        self.args = args
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model = model.to(self.device)
        if args.checkpoint and not eval:
            checkpoint = torch.load(os.path.join(args.checkpoint, 'checkpoint.pt'), map_location=self.device,
                                    weights_only=False)
            self.writer = SummaryWriter(args.checkpoint)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            with open(os.path.join(self.writer.log_dir, 'epoch.txt'), "r") as f:
                self.start_epoch = int(f.read()) + 1
            self.max_val_spearman = checkpoint.get('maximum_spearman', checkpoint.get('maximum_accuracy', -np.inf))
            self.weight = checkpoint.get('weight', torch.ones(1)).to(self.device)
        elif not eval:
            self.start_epoch = 0
            self.max_val_spearman = -np.inf
            self.writer = SummaryWriter(
                'runs/{}_{}_{}'.format(args.model_type, args.experiment_name,
                                       datetime.now().strftime('%d-%m_%H-%M-%S')))
            self.weight = weight.to(self.device) if weight is not None else torch.ones(1, device=self.device)

        self.loss_func = loss_func()

    def train(self, train_loader: DataLoader, val_loader: DataLoader, eval_data=None):
        args = self.args
        epochs_no_improve = 0
        for epoch in range(self.start_epoch, args.num_epochs):
            self.model.train()
            train_loss, train_results = self.predict(train_loader, epoch + 1, optim=self.optim)

            self.model.eval()
            with torch.no_grad():
                val_loss, val_results = self.predict(val_loader, epoch + 1)

            train_spearman = _spearman(train_results)
            val_spearman = _spearman(val_results)
            train_mae = np.abs(train_results[:, 0] - train_results[:, 1]).mean()
            val_mae = np.abs(val_results[:, 0] - val_results[:, 1]).mean()

            print('[Epoch %d] VAL Spearman: %.4f train Spearman: %.4f VAL MSE: %.7f train MSE: %.7f' % (
                epoch + 1, val_spearman, train_spearman, val_loss, train_loss))

            self.writer.add_scalars('MSE', {'train': train_loss, 'val': val_loss}, epoch + 1)
            self.writer.add_scalars('MAE', {'train': train_mae, 'val': val_mae}, epoch + 1)
            self.writer.add_scalars('SpearmanR', {'train': train_spearman, 'val': val_spearman}, epoch + 1)

            if not np.isnan(val_spearman) and val_spearman >= self.max_val_spearman:
                epochs_no_improve = 0
                self.max_val_spearman = val_spearman
                self.save_checkpoint(epoch + 1)
            else:
                epochs_no_improve += 1

            with open(os.path.join(self.writer.log_dir, 'epoch.txt'), 'w') as file:
                file.write(str(epoch))

            if epochs_no_improve >= args.patience:
                break

        if eval_data:
            checkpoint = torch.load(os.path.join(self.writer.log_dir, 'checkpoint.pt'), map_location=self.device,
                                    weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.evaluation(eval_data, filename='val_data_after_training')

    def predict(self, data_loader: DataLoader, epoch: int = None, optim: torch.optim.Optimizer = None) -> \
            Tuple[float, np.ndarray]:
        args = self.args
        results = []
        running_loss = 0
        for i, batch in enumerate(data_loader):
            embedding, target, metadata = batch
            embedding, target = embedding.to(self.device), target.to(self.device)
            sequence_lengths = metadata['length'][:, None].to(self.device)
            frequencies = metadata['frequencies'].to(self.device)

            mask = torch.arange(metadata['length'].max())[None, :] < metadata['length'][:, None]
            prediction = self.model(embedding, mask=mask.to(self.device), sequence_lengths=sequence_lengths,
                                    frequencies=frequencies)
            loss = self.loss_func(prediction, target, args)
            if optim:
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            prediction = prediction.squeeze(-1)
            results.append(torch.stack((prediction, target), dim=1).detach().cpu().numpy())
            running_loss += loss.item()
            if i % args.log_iterations == args.log_iterations - 1:
                if epoch:
                    print('Epoch %d ' % (epoch), end=' ')
                print('[Iter %5d/%5d] %s: MSE loss: %.7f' % (
                    i + 1, len(data_loader), 'Train' if optim else 'Val', loss.item()))

        running_loss /= len(data_loader)
        return running_loss, np.concatenate(results)

    def evaluation(self, eval_dataset: Dataset, filename: str = ''):
        self.model.eval()
        if len(eval_dataset[0][0].shape) == 2:
            collate_function = padded_permuted_meltome_collate
        else:
            collate_function = None

        data_loader = DataLoader(eval_dataset, batch_size=self.args.batch_size, collate_fn=collate_function)
        loss, predictions = self.predict(data_loader)

        np.save(os.path.join(self.writer.log_dir, 'results_array_' + filename), predictions)
        self.save_predictions_csv(eval_dataset, predictions, filename)

        mse = np.mean((predictions[:, 0] - predictions[:, 1]) ** 2)
        mae = np.abs(predictions[:, 0] - predictions[:, 1]).mean()
        rho, pvalue = spearmanr(predictions[:, 1], predictions[:, 0])

        spearman_samples = []
        with torch.no_grad():
            for i in range(self.args.n_draws):
                samples = np.random.choice(range(0, len(eval_dataset)), len(eval_dataset))
                spearman_samples.append(_spearman(predictions[samples]))
        spearman_stderr = np.nanstd(np.array(spearman_samples))

        results_string = 'Number of draws: {} \n' \
                         'Spearman R: {:.4f}\n' \
                         'Spearman p-value: {:.4e}\n' \
                         'Spearman stderr: {:.4f}\n' \
                         'MSE: {:.7f}\n' \
                         'MAE: {:.7f}\n' \
                         'Number of proteins: {}\n'.format(
                             self.args.n_draws, rho, pvalue, spearman_stderr, mse, mae, len(eval_dataset))

        with open(os.path.join(self.writer.log_dir, 'evaluation_' + filename + '.txt'), 'w') as file:
            file.write(results_string)
        print(results_string)
        return rho, mse, mae

    def save_predictions_csv(self, eval_dataset: Dataset, predictions: np.ndarray, filename: str):
        path = os.path.join(self.writer.log_dir, 'predictions_' + filename + '.csv')
        with open(path, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['id', 'target', 'prediction', 'length', 'set', 'validation'])
            for sample, result in zip(eval_dataset.meltome_metadata_list, predictions):
                metadata = sample['metadata']
                writer.writerow([
                    metadata['id'],
                    result[1],
                    result[0],
                    metadata['length'],
                    metadata.get('set', ''),
                    metadata.get('validation', ''),
                ])

    def save_checkpoint(self, epoch: int):
        run_dir = self.writer.log_dir
        torch.save({
            'epoch': epoch,
            'weight': self.weight,
            'maximum_spearman': self.max_val_spearman,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, os.path.join(run_dir, 'checkpoint.pt'))
        train_args = copy.copy(self.args)
        config_name = None
        if hasattr(train_args.config, 'name'):
            config_name = train_args.config.name
        elif train_args.config:
            config_name = str(train_args.config)
        train_args.config = config_name
        pyaml.dump(train_args.__dict__, open(os.path.join(run_dir, 'train_arguments.yaml'), 'w'))
        if config_name and Path(config_name).exists():
            shutil.copyfile(config_name, os.path.join(run_dir, os.path.basename(config_name)))

        model_class = globals()[type(self.model).__name__]
        source_code = inspect.getsource(model_class)
        file_name = os.path.basename(inspect.getfile(model_class))
        with open(os.path.join(run_dir, file_name), "w") as f:
            f.write(source_code)


Solver = SolverMeltome

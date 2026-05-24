#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.general import seed_all, padded_permuted_collate
from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset


class ToTensorLocalization:
    def __call__(self, sample):
        """
        Dataset passes:
            embedding, localization, solubility

        For unsupervised covariance pretraining, labels are ignored later,
        so keep localization/solubility unchanged.
        """
        embedding, loc, sol = sample
        embedding = torch.tensor(embedding).float()
        return embedding, loc, sol


class CovarianceProjectionPretrainer(nn.Module):
    """
    Trains L and R unsupervised by reconstructing the full second moment.

    Input x: [B, d, L_seq]
    """

    def __init__(self, embeddings_dim: int, proj_dim: int):
        super().__init__()
        self.embeddings_dim = embeddings_dim
        self.proj_dim = proj_dim

        self.proj_L = nn.Linear(embeddings_dim, proj_dim, bias=False)
        self.proj_R = nn.Linear(embeddings_dim, proj_dim, bias=False)

    def bilinear_cov(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(-2, -1)           # [B, L, d]

        x_L = self.proj_L(x_t)              # [B, L, d_c]
        x_R = self.proj_R(x_t)              # [B, L, d_c]

        m = mask.unsqueeze(-1).to(x.dtype)  # [B, L, 1]
        x_L = x_L * m
        x_R = x_R * m

        counts = m.sum(-2, keepdim=True).clamp(min=1.0)  # [B, 1, 1]

        C = x_L.transpose(-2, -1) @ x_R / counts          # [B, d_c, d_c]
        return C

    @staticmethod
    def masked_second_moment(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(1).to(x.dtype)                 # [B, 1, L]
        x_masked = x * m                                  # [B, d, L]
        counts = m.sum(-1, keepdim=True).clamp(min=1.0)   # [B, 1, 1]
        return x_masked @ x_masked.transpose(-2, -1) / counts

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.float()

        C = self.bilinear_cov(x, mask)                    # [B, d_c, d_c]

        W_L = self.proj_L.weight                          # [d_c, d]
        W_R = self.proj_R.weight                          # [d_c, d]

        recon = W_L.T.unsqueeze(0) @ C @ W_R.unsqueeze(0) # [B, d, d]
        target = self.masked_second_moment(x, mask)       # [B, d, d]

        return F.mse_loss(recon, target)

def padded_permuted_embedding_only_collate(batch):
    """
    DeepLoc dataset returns:
        embedding, localization, solubility, metadata

    For unsupervised covariance pretraining, ignore labels and keep only:
        embedding: [B, d, L]
        metadata['length']
    """
    embeddings = [torch.as_tensor(item[0]).float() for item in batch]
    metadata = [item[3] for item in batch]

    metadata = torch.utils.data.dataloader.default_collate(metadata)

    embeddings = pad_sequence(embeddings, batch_first=True)  # [B, L, d]
    embeddings = embeddings.permute(0, 2, 1)                 # [B, d, L]

    return embeddings, metadata

def train(args):
    seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = ToTensorLocalization()

    print("train_embeddings:", args.train_embeddings)
    print("train_remapping:", args.train_remapping)
    print("key_format:", args.key_format)
    print("embedding_mode:", args.embedding_mode)
    print("max_length:", args.max_length)

    train_set = EmbeddingsLocalizationDataset(
        args.train_embeddings,
        args.train_remapping,
        max_length=args.max_length,
        key_format=args.key_format,
        embedding_mode=args.embedding_mode,
        transform=transform,
    )

    print("dataset size:", len(train_set))

    if len(train_set[0][0].shape) == 2:
        collate_function = padded_permuted_embedding_only_collate
    else:
        raise ValueError("Unsupervised covariance pretraining requires per-residue embeddings.")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_function,
    )

    embeddings_dim = train_set[0][0].shape[-1]

    model = CovarianceProjectionPretrainer(
        embeddings_dim=embeddings_dim,
        proj_dim=args.proj_dim,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    print("device:", device)
    print("embeddings_dim:", embeddings_dim)
    print("proj_dim:", args.proj_dim)
    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            embedding, metadata = batch
            embedding = embedding.to(device)

            mask = torch.arange(metadata["length"].max())[None, :] < metadata["length"][:, None]
            mask = mask.to(device)

            loss = model(embedding, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if args.log_iterations > 0 and i % args.log_iterations == args.log_iterations - 1:
                print(
                    f"[Epoch {epoch + 1} Iter {i + 1}/{len(train_loader)}] "
                    f"reconstruction loss: {loss.item():.7f}"
                )

            if args.smoke_max_batches is not None and i + 1 >= args.smoke_max_batches:
                break

        epoch_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] reconstruction loss: {epoch_loss:.7f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0

            checkpoint = {
                "proj_L_state_dict": model.proj_L.state_dict(),
                "proj_R_state_dict": model.proj_R.state_dict(),
                "embeddings_dim": embeddings_dim,
                "proj_dim": args.proj_dim,
                "epoch": epoch + 1,
                "loss": best_loss,
            }

            save_path = output_dir / f"cov_unsup_deeploc_dc{args.proj_dim}.pt"
            torch.save(checkpoint, save_path)
            print(f"saved checkpoint: {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("early stopping")
            break


def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=argparse.FileType(mode="r"), default=None)

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--proj_dim", type=int, default=32)
    p.add_argument("--log_iterations", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="runs/cov_unsup_deeploc_pretrained")

    p.add_argument("--max_length", type=int, default=6000)
    p.add_argument("--embedding_mode", type=str, default=None)
    p.add_argument("--key_format", type=str, default=None)

    p.add_argument("--train_embeddings", type=str, default=None)
    p.add_argument("--train_remapping", type=str, default=None)

    p.add_argument("--smoke_max_batches", type=int, default=None)

    args = p.parse_args()

    # Record only arguments the user explicitly typed.
    cli_overrides = {}
    argv = sys.argv[1:]
    for i, token in enumerate(argv):
        if token.startswith("--"):
            key = token.lstrip("--").replace("-", "_")
            if key != "config":
                # Boolean flags are not used here, so this assumes next token is the value.
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    cli_overrides[key] = argv[i + 1]

    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        for key, value in data.items():
            setattr(args, key, value)

    # Re-apply only explicit CLI overrides, with type conversion.
    for key, value in cli_overrides.items():
        current = getattr(args, key, None)

        if key in {"seed", "num_epochs", "batch_size", "patience", "proj_dim", "log_iterations", "max_length", "smoke_max_batches"}:
            value = int(value)
        elif key == "lr":
            value = float(value)

        setattr(args, key, value)

    return args


if __name__ == "__main__":
    train(parse_arguments())
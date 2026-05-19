import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets.embeddings_meltome_dataset import EmbeddingsMeltomeDataset
from solver_meltome import padded_permuted_meltome_collate
from train_meltome import ToTensorMeltome
from utils.general import seed_all


class CovarianceProjectionPretrainer(nn.Module):
    """
    Trains L and R unsupervised by reconstructing the full second moment:

        A = X X^T / L        in R^{d x d}
        C = (XL)^T (XR) / L  in R^{d_c x d_c}
        A_hat = L C R^T      in R^{d x d}

    Input x has shape [B, d, L_seq].
    """

    def __init__(self, embeddings_dim: int, proj_dim: int):
        super().__init__()
        self.embeddings_dim = embeddings_dim
        self.proj_dim = proj_dim

        self.proj_L = nn.Linear(embeddings_dim, proj_dim, bias=False)
        self.proj_R = nn.Linear(embeddings_dim, proj_dim, bias=False)

    def bilinear_cov(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, d, L]
        x_t = x.transpose(-2, -1)          # [B, L, d]

        x_L = self.proj_L(x_t)             # [B, L, d_c]
        x_R = self.proj_R(x_t)             # [B, L, d_c]

        m = mask.unsqueeze(-1).to(x.dtype) # [B, L, 1]
        x_L = x_L * m
        x_R = x_R * m

        counts = m.sum(-2, keepdim=True).clamp(min=1.0)  # [B, 1, 1]

        C = x_L.transpose(-2, -1) @ x_R / counts          # [B, d_c, d_c]
        return C

    @staticmethod
    def masked_second_moment(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, d, L]
        m = mask.unsqueeze(1).to(x.dtype)                 # [B, 1, L]
        x_masked = x * m                                  # [B, d, L]
        counts = m.sum(-1, keepdim=True).clamp(min=1.0)   # [B, 1, 1]
        return x_masked @ x_masked.transpose(-2, -1) / counts  # [B, d, d]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.float()

        C = self.bilinear_cov(x, mask)                    # [B, d_c, d_c]

        W_L = self.proj_L.weight                          # [d_c, d]
        W_R = self.proj_R.weight                          # [d_c, d]

        recon = W_L.T.unsqueeze(0) @ C @ W_R.unsqueeze(0) # [B, d, d]
        target = self.masked_second_moment(x, mask)       # [B, d, d]

        # Normalized squared Frobenius reconstruction loss.
        # Equivalent to mean squared error over the full second-moment matrix.
        loss = F.mse_loss(recon, target)
        return loss


def train(args):
    seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    transform = ToTensorMeltome()

    train_set = EmbeddingsMeltomeDataset(
        args.train_embeddings,
        args.train_remapping,
        max_length=args.max_length,
        key_format=args.key_format,
        embedding_mode=args.embedding_mode,
        transform=transform,
    )

    if len(train_set[0][0].shape) == 2:
        collate_function = padded_permuted_meltome_collate
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

    best_loss = float("inf")
    epochs_no_improve = 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            embedding, target, metadata = batch
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

            save_path = output_dir / f"cov_unsup_dc{args.proj_dim}.pt"
            torch.save(checkpoint, save_path)
            print(f"saved checkpoint: {save_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print("early stopping")
            break


def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--proj_dim", type=int, default=32)
    p.add_argument("--log_iterations", type=int, default=50)
    p.add_argument("--output_dir", type=str, default="runs/cov_unsup_pretrained")

    p.add_argument("--max_length", type=int, default=6000)
    p.add_argument("--embedding_mode", type=str, default="lm")
    p.add_argument("--key_format", type=str, default="hash")

    p.add_argument(
        "--train_embeddings",
        type=str,
        default="data_files/flip_meltome/prepared/human_cell/human_cell_train.h5",
    )
    p.add_argument(
        "--train_remapping",
        type=str,
        default="data_files/flip_meltome/prepared/human_cell/human_cell_train_remapped.fasta",
    )

    p.add_argument("--config", type=argparse.FileType(mode="r"), default=None)

    args = p.parse_args()

    if args.config:
        data = yaml.load(args.config, Loader=yaml.FullLoader)
        for key, value in data.items():
            setattr(args, key, value)

    return args


if __name__ == "__main__":
    train(parse_arguments())
#!/usr/bin/env python3
"""
Unsupervised covariance-projection pretraining on the union of all task train splits.

The projections L, R in R^{d x d_c} are trained once, frozen, and reused across tasks
(spec section 3.1 / 5.1). They are learned by reconstructing the full second-moment
matrix X X^T / L, exploiting the Frobenius-norm equivalence ||X^T X||_F = ||X X^T||_F
so that no sequence-by-sequence (L x L) matrix is materialized.

Data sources are listed in the config under `sources`, one block per train split.
Only train splits are used; val/test of either task are never touched.

Usage:
    python train_cov_unsup.py --config configs/cov_unsup_pretrain.yaml
    python train_cov_unsup.py --config configs/cov_unsup_pretrain.yaml --proj_dim 8
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader

from datasets.embeddings_localization_dataset import EmbeddingsLocalizationDataset
from datasets.embeddings_meltome_dataset import EmbeddingsMeltomeDataset
from utils.general import seed_all


DATASET_CLASSES = {
    'localization': EmbeddingsLocalizationDataset,
    'meltome': EmbeddingsMeltomeDataset,
}


class CovarianceProjectionPretrainer(nn.Module):
    """
    Trains L and R unsupervised by reconstructing the full second moment:

        A     = X X^T / L        in R^{d x d}
        C     = (XL)^T (XR) / L  in R^{d_c x d_c}
        A_hat = L C R^T          in R^{d x d}

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        x = x.float()

        C = self.bilinear_cov(x, mask)                    # [B, d_c, d_c]

        W_L = self.proj_L.weight                          # [d_c, d]
        W_R = self.proj_R.weight                          # [d_c, d]

        recon = W_L.T.unsqueeze(0) @ C @ W_R.unsqueeze(0) # [B, d, d]
        target = self.masked_second_moment(x, mask)       # [B, d, d]

        # Normalized squared Frobenius reconstruction loss.
        # Equivalent to mean squared error over the full second-moment matrix.
        loss = F.mse_loss(recon, target)

        # Scale-free diagnostic: relative Frobenius error per sample, mean over batch.
        # ~1.0 means "predicting zero" (no learning); ->0 means perfect reconstruction.
        with torch.no_grad():
            num = (recon - target).flatten(1).norm(dim=1)
            den = target.flatten(1).norm(dim=1).clamp(min=1e-12)
            rel_err = (num / den).mean()

        return loss, rel_err


def padded_permuted_embedding_only_collate(batch):
    """
    Both task datasets return tuples that end in a metadata dict containing 'length',
    with the per-residue embedding as item[0]. For unsupervised pretraining the labels
    are ignored, so keep only the padded embeddings and the sequence lengths.

    Returns embeddings as [batch_size, embeddings_dim, sequence_length].
    """
    embeddings = [torch.as_tensor(item[0]).float() for item in batch]
    lengths = torch.tensor([item[-1]['length'] for item in batch])
    embeddings = pad_sequence(embeddings, batch_first=True)  # [B, L, d]
    embeddings = embeddings.permute(0, 2, 1)                 # [B, d, L]
    return embeddings, lengths


def build_union_dataset(args) -> ConcatDataset:
    datasets = []
    for source in args.sources:
        dataset_type = source['dataset']
        if dataset_type not in DATASET_CLASSES:
            raise ValueError(
                f"Unknown source dataset {dataset_type!r}; "
                f"expected one of {sorted(DATASET_CLASSES)}"
            )
        dataset_cls = DATASET_CLASSES[dataset_type]

        print(f"source: {dataset_type}")
        print(f"  embeddings: {source['embeddings']}")
        print(f"  remapping:  {source['remapping']}")
        print(f"  key_format: {source['key_format']}")

        dataset = dataset_cls(
            source['embeddings'],
            source['remapping'],
            max_length=args.max_length,
            key_format=source['key_format'],
            embedding_mode=args.embedding_mode,
        )
        print(f"  size:       {len(dataset)}")
        datasets.append(dataset)

    union = ConcatDataset(datasets)
    print(f"union dataset size: {len(union)}")
    return union


@torch.no_grad()
def compute_avg_second_moment(loader, embeddings_dim, device):
    """One pass over the data: mean over proteins of X X^T / L  ->  [d, d]."""
    Mbar = torch.zeros(embeddings_dim, embeddings_dim, device=device)
    total = 0
    for embedding, lengths in loader:
        embedding = embedding.to(device)
        mask = torch.arange(lengths.max())[None, :] < lengths[:, None]
        mask = mask.to(device)
        m = mask.unsqueeze(1).to(embedding.dtype)         # [B, 1, L]
        xm = embedding * m                                # [B, d, L]
        counts = m.sum(-1, keepdim=True).clamp(min=1.0)   # [B, 1, 1]
        M = xm @ xm.transpose(-2, -1) / counts            # [B, d, d]
        Mbar += M.sum(0)
        total += embedding.shape[0]
    return Mbar / max(total, 1)


def pca_init_projections(model, loader, embeddings_dim, proj_dim, device):
    """Initialize proj_L, proj_R from the top-proj_dim eigenvectors of the average
    second moment, so recon = (V V^T) M (V V^T) holds the closed-form floor at init.
    proj_L.weight has shape [proj_dim, d], so we copy V^T (V is [d, proj_dim])."""
    print("computing PCA init basis (one pass over data)...")
    Mbar = compute_avg_second_moment(loader, embeddings_dim, device)
    # eigh on CPU in double precision (MPS lacks linalg.eigh); eigenvalues ascending.
    evecs = torch.linalg.eigh(Mbar.cpu().double())[1]
    V = evecs[:, -proj_dim:].to(torch.float32)            # [d, proj_dim] top eigenvectors
    with torch.no_grad():
        model.proj_L.weight.copy_(V.t().to(device))
        model.proj_R.weight.copy_(V.t().to(device))
    print(f"initialized L, R from top-{proj_dim} eigenvectors of avg second moment")


@torch.no_grad()
def evaluate(model, loader, device):
    """Average reconstruction loss and rel_err over the loader."""
    model.eval()
    total_loss, total_rel, n = 0.0, 0.0, 0
    for embedding, lengths in loader:
        embedding = embedding.to(device)
        mask = torch.arange(lengths.max())[None, :] < lengths[:, None]
        mask = mask.to(device)
        loss, rel_err = model(embedding, mask)
        total_loss += loss.item()
        total_rel += rel_err.item()
        n += 1
    return total_loss / max(n, 1), total_rel / max(n, 1)


def save_checkpoint(model, embeddings_dim, proj_dim, epoch, loss, rel_err, output_dir):
    checkpoint = {
        "proj_L_state_dict": model.proj_L.state_dict(),
        "proj_R_state_dict": model.proj_R.state_dict(),
        "embeddings_dim": embeddings_dim,
        "proj_dim": proj_dim,
        "epoch": epoch,
        "loss": loss,
        "rel_err": rel_err,
    }
    save_path = Path(output_dir) / f"cov_unsup_dc{proj_dim}.pt"
    torch.save(checkpoint, save_path)
    return save_path


def train(args):
    seed_all(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("max_length:", args.max_length)
    print("embedding_mode:", args.embedding_mode)

    train_set = build_union_dataset(args)

    if len(train_set[0][0].shape) != 2:
        raise ValueError("Unsupervised covariance pretraining requires per-residue embeddings.")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=padded_permuted_embedding_only_collate,
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

    pca_init = bool(getattr(args, "pca_init", None)) or bool(getattr(args, "pca_only", None))
    if pca_init:
        pca_init_projections(model, train_loader, embeddings_dim, args.proj_dim, device)
        # Score and save the pristine PCA solution, and seed best_loss so subsequent
        # SGD refinement can never leave us with a worse checkpoint than PCA init.
        best_loss, init_rel_err = evaluate(model, train_loader, device)
        save_path = save_checkpoint(model, embeddings_dim, args.proj_dim, 0, best_loss, init_rel_err, output_dir)
        print(f"[PCA init] reconstruction loss: {best_loss:.7f}  rel_err: {init_rel_err:.4f}")
        print(f"saved checkpoint: {save_path}")

        if bool(getattr(args, "pca_only", None)):
            print("pca_only set — skipping SGD refinement.")
            return

    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        running_rel_err = 0.0

        for i, batch in enumerate(train_loader):
            embedding, lengths = batch
            embedding = embedding.to(device)

            mask = torch.arange(lengths.max())[None, :] < lengths[:, None]
            mask = mask.to(device)

            loss, rel_err = model(embedding, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_rel_err += rel_err.item()

            if args.log_iterations > 0 and i % args.log_iterations == args.log_iterations - 1:
                print(
                    f"[Epoch {epoch + 1} Iter {i + 1}/{len(train_loader)}] "
                    f"reconstruction loss: {loss.item():.7f}  rel_err: {rel_err.item():.4f}"
                )

        epoch_loss = running_loss / len(train_loader)
        epoch_rel_err = running_rel_err / len(train_loader)
        print(f"[Epoch {epoch + 1}] reconstruction loss: {epoch_loss:.7f}  rel_err: {epoch_rel_err:.4f}")

        # Only count an improvement that beats the best by more than min_delta (relative).
        # Without this, microscopic late-stage gains keep resetting patience forever.
        significant = epoch_loss < best_loss * (1.0 - args.min_delta)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = save_checkpoint(model, embeddings_dim, args.proj_dim,
                                        epoch + 1, best_loss, epoch_rel_err, output_dir)
            print(f"saved checkpoint: {save_path}")
        if significant:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"early stopping (no >{args.min_delta:.1%} improvement for {args.patience} epochs)")
            break


def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=argparse.FileType(mode="r"), required=True)

    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--proj_dim", type=int, default=None)
    p.add_argument("--log_iterations", type=int, default=None)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--max_length", type=int, default=None)
    p.add_argument("--embedding_mode", type=str, default=None)
    p.add_argument("--min_delta", type=float, default=None,
                   help="Min relative loss improvement to reset early-stopping patience "
                        "(default 0.005 = 0.5%%). Prevents microscopic gains from blocking stop.")
    # store_const with default=None so config can set these and the None-filter below
    # only treats them as CLI overrides when the flag is actually passed.
    p.add_argument("--pca_init", action="store_const", const=True, default=None,
                   help="Initialize L, R from top-d_c eigenvectors, then refine with SGD.")
    p.add_argument("--pca_only", action="store_const", const=True, default=None,
                   help="PCA-initialize and save immediately, no SGD refinement.")

    args = p.parse_args()

    # Config supplies the defaults; explicit CLI flags override them.
    cli_overrides = {k: v for k, v in vars(args).items()
                     if k not in ("config",) and v is not None}

    data = yaml.load(args.config, Loader=yaml.FullLoader)
    for key, value in data.items():
        setattr(args, key, value)

    for key, value in cli_overrides.items():
        setattr(args, key, value)

    if getattr(args, "min_delta", None) is None:
        args.min_delta = 0.005

    return args


if __name__ == "__main__":
    train(parse_arguments())

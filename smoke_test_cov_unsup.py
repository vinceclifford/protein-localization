from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from models import PoolingFFN


class CovarianceProjectionPretrainer(nn.Module):
    def __init__(self, embeddings_dim: int, proj_dim: int):
        super().__init__()
        self.proj_L = nn.Linear(embeddings_dim, proj_dim, bias=False)
        self.proj_R = nn.Linear(embeddings_dim, proj_dim, bias=False)

    def bilinear_cov(self, x, mask):
        # x: [B, d, L]
        x_t = x.transpose(-2, -1)          # [B, L, d]
        x_L = self.proj_L(x_t)             # [B, L, d_c]
        x_R = self.proj_R(x_t)             # [B, L, d_c]

        m = mask.unsqueeze(-1).to(x.dtype)
        x_L = x_L * m
        x_R = x_R * m

        counts = m.sum(-2, keepdim=True).clamp(min=1.0)
        return x_L.transpose(-2, -1) @ x_R / counts

    @staticmethod
    def masked_second_moment(x, mask):
        # x: [B, d, L]
        m = mask.unsqueeze(1).to(x.dtype)
        x_masked = x * m
        counts = m.sum(-1, keepdim=True).clamp(min=1.0)
        return x_masked @ x_masked.transpose(-2, -1) / counts

    def forward(self, x, mask):
        C = self.bilinear_cov(x, mask)

        W_L = self.proj_L.weight
        W_R = self.proj_R.weight

        recon = W_L.T.unsqueeze(0) @ C @ W_R.unsqueeze(0)
        target = self.masked_second_moment(x, mask)

        return F.mse_loss(recon, target)


def main():
    torch.manual_seed(123)

    out_dir = Path("runs/smoke_cov_unsup_synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = out_dir / "cov_unsup_dc2.pt"

    batch_size = 2
    embeddings_dim = 16
    seq_len = 8
    proj_dim = 2

    # ------------------------------------------------------------------
    # Stage 1: synthetic unsupervised pretraining
    # ------------------------------------------------------------------
    pretrainer = CovarianceProjectionPretrainer(
        embeddings_dim=embeddings_dim,
        proj_dim=proj_dim,
    )

    optimizer = Adam(pretrainer.parameters(), lr=1e-3)

    x = torch.randn(batch_size, embeddings_dim, seq_len)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    loss = pretrainer(x, mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch.save(
        {
            "proj_L_state_dict": pretrainer.proj_L.state_dict(),
            "proj_R_state_dict": pretrainer.proj_R.state_dict(),
            "embeddings_dim": embeddings_dim,
            "proj_dim": proj_dim,
            "loss": loss.item(),
        },
        checkpoint_path,
    )

    print(f"saved checkpoint: {checkpoint_path}")
    print(f"pretrain loss: {loss.item():.7f}")

    # ------------------------------------------------------------------
    # Stage 2: load frozen L/R into PoolingFFN
    # ------------------------------------------------------------------
    model = PoolingFFN(
        embeddings_dim=embeddings_dim,
        output_dim=1,
        hidden_dim=4,
        n_hidden_layers=0,
        dropout=0.0,
        pooling="cov_unsup",
        proj_dim=proj_dim,
        cov_unsup_checkpoint=str(checkpoint_path),
        freeze_cov_projections=True,
    )

    print("\nParameter freeze check:")
    for name, p in model.named_parameters():
        print(name, p.requires_grad)

    y = model(x, mask)
    print("\nforward output shape:", tuple(y.shape))

    assert y.shape == (batch_size, 1)
    assert model.proj_L.weight.requires_grad is False
    assert model.proj_R.weight.requires_grad is False

    # ------------------------------------------------------------------
    # Stage 3: synthetic downstream supervised step
    # ------------------------------------------------------------------
    target = torch.randn(batch_size)

    downstream_optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    pred = model(x, mask).squeeze(-1)
    downstream_loss = F.mse_loss(pred, target)

    downstream_optimizer.zero_grad()
    downstream_loss.backward()
    downstream_optimizer.step()

    print(f"\ndownstream loss: {downstream_loss.item():.7f}")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
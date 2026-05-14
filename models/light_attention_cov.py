import torch
import torch.nn as nn


class LightAttentionCov(nn.Module):
    """
    LightAttention augmented with a parallel bilinear covariance pooling branch.

    Three pooling branches all operate on the same per-residue input X in R^{d x L}:
        o1     = sum_t  softmax(a_t) * f_t                   (LA attention-weighted mean, R^d)
        o2     = max_t  f_t                                  (LA max pool,                R^d)
        C      = (1/L)  (X L_mat)^T (X R_mat)                (bilinear covariance,        R^{d_c x d_c})

    where f = feature_conv(X), a = attention_conv(X), and L_mat, R_mat in R^{d x d_c}
    are two independent learnable projections (asymmetric bilinear pooling).

    The flattened covariance is concatenated with o1 and o2, yielding a pooled
    vector of size 2*d + d_c^2 fed through the same compact head LightAttention uses.

    All branches respect the sequence mask so zero-padded positions never enter
    the statistics.
    """

    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 10,
                 dropout: float = 0.25, kernel_size: int = 9,
                 conv_dropout: float = 0.25, proj_dim: int = 32,
                 hidden_dim: int = 32):
        super().__init__()

        # LightAttention branches (kept verbatim w.r.t. shape/params for fair comparison)
        self.feature_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size,
            stride=1, padding=kernel_size // 2,
        )
        self.attention_convolution = nn.Conv1d(
            embeddings_dim, embeddings_dim, kernel_size,
            stride=1, padding=kernel_size // 2,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(conv_dropout)

        # New bilinear covariance branch (Linear over the channel dim — MPS-friendly).
        self.proj_L = nn.Linear(embeddings_dim, proj_dim, bias=False)
        self.proj_R = nn.Linear(embeddings_dim, proj_dim, bias=False)
        self.proj_dim = proj_dim

        pooled_dim = 2 * embeddings_dim + proj_dim * proj_dim
        self.linear = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.output = nn.Linear(hidden_dim, output_dim)

    def _bilinear_cov(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, d, L] -> C: [B, d_c, d_c] flattened to [B, d_c^2]
        x_t = x.transpose(-2, -1)                         # [B, L, d]
        x_L = self.proj_L(x_t)                            # XL : [B, L, d_c]
        x_R = self.proj_R(x_t)                            # XR : [B, L, d_c]
        m = mask.unsqueeze(-1).to(x.dtype)                # [B, L, 1]
        x_L = x_L * m
        x_R = x_R * m
        counts = m.sum(-2, keepdim=True).clamp(min=1.0)   # [B, 1, 1]
        C = x_L.transpose(-2, -1) @ x_R / counts          # [B, d_c, d_c]
        return C.flatten(start_dim=1)                     # [B, d_c^2]

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, d, L]
        x = x.float()

        # ---- LightAttention branches ----
        f = self.feature_convolution(x)                   # [B, d, L]
        f = self.dropout(f)
        a = self.attention_convolution(x)                 # [B, d, L]
        a = a.masked_fill(mask[:, None, :] == False, -1e9)
        o1 = torch.sum(f * self.softmax(a), dim=-1)       # [B, d]
        o2, _ = torch.max(f, dim=-1)                      # [B, d]

        # ---- Covariance branch ----
        flat_C = self._bilinear_cov(x, mask)              # [B, d_c^2]

        # ---- Concat + head ----
        pooled = torch.cat([o1, o2, flat_C], dim=-1)      # [B, 2d + d_c^2]
        h = self.linear(pooled)
        return self.output(h)

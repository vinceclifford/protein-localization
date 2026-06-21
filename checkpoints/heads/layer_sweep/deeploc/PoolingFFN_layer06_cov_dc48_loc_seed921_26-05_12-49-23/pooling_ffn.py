class PoolingFFN(nn.Module):
    """
    Per-residue protein embeddings X in R^{L x d} -> pooled vector -> FFN probe head.

    Pooling methods (matching the project spec):
        'mean'   : mu = (1/L) X^T 1_L                    in R^d
        'cov'    : C  = (1/L) (X L_mat)^T (X R_mat)      in R^{d_c x d_c}, flattened
        'hybrid' : [mu ; flat(C)]                        in R^{d + d_c^2}

    L_mat, R_mat in R^{d x d_c} are two independent learnable projections (asymmetric
    bilinear pooling). They are implemented as 1x1 Conv1d layers without bias for
    efficiency. Total projection parameters: 2 * d * d_c.

    Padded positions are masked out of all sums and from the length normalizer.
    """

    def __init__(self, embeddings_dim: int = 1024, output_dim: int = 10,
                 hidden_dim: int = 32, n_hidden_layers: int = 0, dropout: float = 0.25,
                 pooling: str = 'mean', proj_dim: int = 32):
        super().__init__()
        if pooling not in ('mean', 'cov', 'hybrid'):
            raise ValueError(f"pooling must be 'mean'|'cov'|'hybrid', got {pooling!r}")
        self.pooling = pooling
        self.embeddings_dim = embeddings_dim
        self.proj_dim = proj_dim

        if pooling in ('cov', 'hybrid'):
            # L_mat and R_mat in R^{d x d_c}. Implemented as nn.Linear over the
            # channel dim (faster on MPS than Conv1d kernel_size=1). Applied to
            # x.transpose(-2, -1) of shape [B, L, d] -> [B, L, d_c].
            self.proj_L = nn.Linear(embeddings_dim, proj_dim, bias=False)
            self.proj_R = nn.Linear(embeddings_dim, proj_dim, bias=False)

        if pooling == 'mean':
            pooled_dim = embeddings_dim
        elif pooling == 'cov':
            pooled_dim = proj_dim * proj_dim
        else:
            pooled_dim = embeddings_dim + proj_dim * proj_dim

        self.input = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.hidden = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            )
            for _ in range(n_hidden_layers)
        ])
        self.output = nn.Linear(hidden_dim, output_dim)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, d, L], mask: [B, L] (True for real, False for padding)
        m = mask.unsqueeze(1).to(x.dtype)            # [B, 1, L]
        counts = m.sum(-1).clamp(min=1.0)            # [B, 1]
        return (x * m).sum(-1) / counts              # [B, d]

    def _bilinear_cov(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, d, L] -> C: [B, d_c, d_c] flattened to [B, d_c^2]
        x_t = x.transpose(-2, -1)                    # [B, L, d]
        x_L = self.proj_L(x_t)                       # XL : [B, L, d_c]
        x_R = self.proj_R(x_t)                       # XR : [B, L, d_c]
        m = mask.unsqueeze(-1).to(x.dtype)           # [B, L, 1]
        x_L = x_L * m
        x_R = x_R * m
        counts = m.sum(-2, keepdim=True).clamp(min=1.0)   # [B, 1, 1]
        C = x_L.transpose(-2, -1) @ x_R / counts     # [B, d_c, d_c]
        return C.flatten(start_dim=1)                # [B, d_c^2]

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        # x: [B, d, L] (per-residue, after padded_permuted_collate)
        x = x.float()
        if self.pooling == 'mean':
            pooled = self._masked_mean(x, mask)
        elif self.pooling == 'cov':
            pooled = self._bilinear_cov(x, mask)
        else:
            pooled = torch.cat(
                [self._masked_mean(x, mask), self._bilinear_cov(x, mask)],
                dim=-1,
            )

        o = self.input(pooled)
        for layer in self.hidden:
            o = layer(o)
        return self.output(o)

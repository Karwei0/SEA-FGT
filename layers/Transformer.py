from layers.FGT import FGTEncoderLayer, TemporalEmbedding
import torch
import torch.nn as nn


class Transformer(nn.Module):
    """
    Frequency-Guided Transformer (FGT) without seg
    Input:  x [B, N, T]
    Output: h [B, T, D]
    """
    def __init__(self,
                 num_channels: int,
                 T: int,
                 d_model: int,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 ffn_dim: int = 2048,
                 position_embedding: str = 'rotate'):
        super(Transformer, self).__init__()

        self.num_channels = num_channels
        self.T = T
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # ✅ Removed: SpectralGate / SEG

        # time embedding
        self.temporal_embedding = TemporalEmbedding(T, num_channels, d_model)

        # position embedding
        self.position_embedding = None
        # if position_embedding is not None and position_embedding != 'rotate':
        #     self.position_embedding = RotaryPositionEmbedding(d_model)

        # encoder stack
        self.encoder_layers = nn.ModuleList([
            FGTEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # output
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, num_channels)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, N, T]
        :return: [B, T, D]
        """
        B, N, T = x.shape

        # ✅ Removed: spectral gate
        # channel_gate = self.spectral_gate(x)
        # x = x * channel_gate.unsqueeze(-1)

        # temporal embedding
        z = self.temporal_embedding(x)

        # position embedding
        if self.position_embedding is not None:
            pos_emb = self.position_embedding(z)
            z = z + pos_emb

        # transformer encoder
        for encoder in self.encoder_layers:
            z = encoder(z)

        # projection（保留你原来的写法；不过你现在 return 的是 z）
        h = self.layer_norm(z)
        h = self.output_projection(z)

        return z


class DualTransformer(nn.Module):
    def __init__(self,
                 num_channels: int,
                 T: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 dropout: float = 0.1,
                 ffn_dim: int = 2048,
                 position_embedding: str = 'rotate'):
        super(DualTransformer, self).__init__()
        self.shared_fgt = Transformer(
            num_channels=num_channels,
            T=T,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            ffn_dim=ffn_dim,
            position_embedding=position_embedding
        )

    def forward(self, x_ori: torch.Tensor, x_aug: torch.Tensor):
        h_ori = self.shared_fgt(x_ori)
        h_aug = self.shared_fgt(x_aug)
        return h_ori, h_aug
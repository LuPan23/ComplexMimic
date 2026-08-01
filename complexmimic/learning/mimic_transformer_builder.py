import torch
import torch.nn as nn
from complexmimic.learning.amp_network_builder import AMPBuilder


class Encoder_TRANSFORMER(nn.Module):
    def __init__(self,
                 feat_splits=[358,216,400],   #  358 + 216 + 400 feat_splits=[358,216,400],
                 latent_dim=512,
                 ff_size=2048,
                 num_layers=4,
                 num_heads=4,
                 dropout=0.0,                       # RL里建议0
                 activation="relu",
                 use_post_norm=True):
        super().__init__()

        self.feat_splits = feat_splits
        self.latent_dim = latent_dim

        # 每个分片 -> latent
        self.token_embeds = nn.ModuleList([nn.Linear(n, latent_dim) for n in feat_splits])

        # learnable weight token + pos embed
        self.weight_token = nn.Parameter(torch.zeros(1, 1, latent_dim))
        nn.init.trunc_normal_(self.weight_token, std=0.02)

        num_tokens = len(feat_splits) + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 可选归一化（常能更稳）
        self.post_norm = nn.LayerNorm(latent_dim) if use_post_norm else nn.Identity()

        # 显式初始化线性层（可选但推荐）
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        batch: [B, sum(feat_splits)] 例如 [B, 2296]
        return: latent [B, latent_dim]
        """
        B, D = batch.shape
        assert D == sum(self.feat_splits), f"got {D}, expected {sum(self.feat_splits)}"

        # 1) split -> tokens
        xs = torch.split(batch, self.feat_splits, dim=1)                 # tuple of [B, split]
        tokens = torch.stack([emb(x) for emb, x in zip(self.token_embeds, xs)], dim=1)  # [B, 4, latent]

        # 2) prepend weight token
        wt = self.weight_token.expand(B, -1, -1)                         # [B,1,latent]
        x = torch.cat([wt, tokens], dim=1)                               # [B, 1+4, latent]

        # 3) add pos embed (按实际长度裁剪，稳一点)
        x = x + self.pos_embed[:, :x.size(1), :]

        # 4) Transformer
        x = self.transformer_encoder(x)                                  # [B, 1+4, latent]

        # 5) 取 weight token 并可选归一化
        z = self.post_norm(x[:, 0])                                      # [B, latent]

        return z
    
class MimicTransformerBuilder(AMPBuilder):

    class Network(AMPBuilder.Network):

        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            self.actor_mlp = Encoder_TRANSFORMER()

    def build(self, name, **kwargs):
        return MimicTransformerBuilder.Network(self.params, **kwargs)

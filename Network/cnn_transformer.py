"""CNN-Transformer hybrid model for image classification.

Designed to be a lightweight, configurable reference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class CNNTransformerConfig:
    image_size: int = 224
    in_channels: int = 3
    num_classes: int = 1000
    cnn_channels: Tuple[int, ...] = (64, 128, 256)
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNStem(nn.Module):
    def __init__(self, config: CNNTransformerConfig) -> None:
        super().__init__()
        layers = []
        in_channels = config.in_channels
        for idx, out_channels in enumerate(config.cnn_channels):
            stride = 2 if idx == 0 else 1
            layers.append(ConvBlock(in_channels, out_channels, stride=stride))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.projection = nn.Conv2d(in_channels, config.embed_dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.projection(x)


class TransformerEncoder(nn.Module):
    def __init__(self, config: CNNTransformerConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=int(config.embed_dim * config.mlp_ratio),
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CNNTransformerClassifier(nn.Module):
    def __init__(self, config: CNNTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.cnn = CNNStem(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self._num_patches(config.image_size) + 1, config.embed_dim)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.transformer = TransformerEncoder(config)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _num_patches(self, image_size: int) -> int:
        reduced = image_size
        for idx in range(len(self.config.cnn_channels)):
            if idx == 0:
                reduced = reduced // 2
            reduced = reduced // 2
        return reduced * reduced

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        batch_size, channels, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embedding[:, : x.size(1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def build_cnn_transformer(config: CNNTransformerConfig | None = None) -> CNNTransformerClassifier:
    if config is None:
        config = CNNTransformerConfig()
    return CNNTransformerClassifier(config)


if __name__ == "__main__":
    model = build_cnn_transformer()
    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    print(logits.shape)

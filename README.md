# cnn-transformer

A minimal CNN-Transformer hybrid model for image classification.

## Overview

This repository provides a reference implementation that combines a lightweight CNN stem with a Transformer encoder. The CNN reduces spatial resolution and projects features into a token sequence, while the Transformer captures global context before classification.

## Model design

- **CNN stem**: stacked convolutional blocks with max pooling to downsample features.
- **Token projection**: 1x1 convolution projects CNN features into Transformer embeddings.
- **Transformer encoder**: multi-head self-attention layers with GELU feed-forward blocks.
- **Classifier head**: uses a learnable `[CLS]` token for image-level predictions.

## Usage

```python
from Network.cnn_transformer import build_cnn_transformer, CNNTransformerConfig

config = CNNTransformerConfig(
    image_size=224,
    num_classes=100,
    cnn_channels=(64, 128, 256),
    embed_dim=256,
    num_heads=4,
    num_layers=4,
)
model = build_cnn_transformer(config)
```

## Quick check

```bash
python Network/cnn_transformer.py
```

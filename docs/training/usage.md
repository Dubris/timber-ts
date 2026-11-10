# Usage

The examples of this page use the same small model as the [README](https://github.com/Dubris/timber-ts#readme), so that
they run on a laptop. Real settings are in the [configuration reference](../configuration.md) and in the paper.

## Build a model

A TIMBER model is described by a `TimberConfig`, made of one sub-configuration per module. Any field you leave out takes
the default of the paper's pretraining setting. The arguments of each module are:

- `tokenizer`: patch length, number of channels, channel embedding size (`TokenizerConfig`);
- `contextualizer`: width, number of layers, heads, causal or not (`ContextualizerConfig`);
- `encoder`: number of latent queries and their width, number of blocks (`EncoderConfig`);
- `decoders`: one entry per pretraining objective, such as TACO (`DecoderConfig`);
- `head`: for finetuning only (`HeadConfig`).

```python
import torch
from timber import TimberConfig, TimberForPretraining

config = TimberConfig(
    tokenizer={"patch_length": 16, "num_channels": 64, "channel_embedding_dim": 64},
    contextualizer={
        "hidden_dim": 128,        # width of the contextualizer layers
        "num_layers": 3,          # alternating channel and time attention layers
        "num_heads": 4,
        "is_causal": True,        # a patch only sees the past along time (required by TACO)
    },
    encoder={
        "num_latents": 32,        # latent queries, the summary of an entity
        "latent_dim": 256,
        "num_blocks": 2,
    },
    decoders=[{"objective": "forecasting_1"}],  # TACO: predict the next patch of each channel
    patch_mask_ratio=0.7,                         # hide 70% of the patches before the encoder
)
model = TimberForPretraining(config)
output = model(torch.randn(4, 64, 128))  # (entity, channel, time)
output.loss.backward()
```

## Get representations for downstream tasks

The backbone, `TimberModel`, returns two representations, both without masking:

- the **latent embedding** (`last_hidden_state`, `(entity, latent, latent_dim)`): a compact summary of all the patches
  of an entity, used by linear probes and finetuning heads;
- the **contextualized patches** (`contextualized`, `(entity, channel, patch, dim)`): one representation per channel and
  per time patch, before the encoder.

```python
from timber import TimberModel

model.save_pretrained("my-timber")
backbone = TimberModel.from_pretrained("my-timber").eval()
with torch.inference_mode():
    out = backbone(torch.randn(4, 64, 128), mask_patches=False)
latent_embedding = out.last_hidden_state   # (4, 33, 256): 32 latent queries + 1 CLS token, if enabled
patches = out.contextualized                # (4, 64, 8, 128): 8 patches of 16 samples per channel
```

To embed a subset of the channels, pass `channel_indices` as explained in the [README](https://github.com/Dubris/timber-ts#readme).

For a whole dataset, the data pipeline gives the batches. Here the recordings are split into two channel shards, and
each training sample takes a random 128-sample time window:

```python
import numpy as np
from timber import TimberConfig
from timber.data import TimberDataModule

files = {  # (entity, recording, shard) -> (time, channel) array of the shard
    ("subject-01", 0, 0): "subject-01_run-0_shard-0.npy", ("subject-01", 0, 1): "subject-01_run-0_shard-1.npy",
    ("subject-02", 0, 0): "subject-02_run-0_shard-0.npy", ("subject-02", 0, 1): "subject-02_run-0_shard-1.npy",
}
channel_index = {0: np.arange(0, 8), 1: np.arange(8, 16)}  # channels of each shard -> global channels
datamodule = TimberDataModule(
    {
        "sampler": {
            "partition_shape": [2, 1, 8, 300, 2],        # entities, recordings, channels per shard, time, shards
            "chunk_shape": [1, 1, 4, 128, 1],            # 4 channels and a 128-sample window per chunk
            "chunks_per_batch": [2, 1, 1, 1, 1],         # 2 entities per batch, one chunk along each other axis
            "shuffle_before_chunking": [True, True, True, False, True],  # the time window is contiguous
        },
        "num_train_entities": 2, "num_val_entities": 0, "num_test_entities": 0,
        "resample_time_window": True,                   # a new random window at every epoch
    },
    files,
    channel_index=channel_index,
)
datamodule.setup()
signal, positions, _ = next(iter(datamodule.train_dataloader()))
# signal: (2, 1, 1, 1, 4, 128): the shard axis is merged into the channel-chunk axis, each chunk comes from one shard
```

## Finetune

Finetuning uses a model with a head and no pretraining decoder, loaded from the pretrained backbone. The head is plugged
after the encoder; the backbone parameters keep their pretrained values and can be frozen:

```python
from timber import TimberForPrediction

model = TimberForPrediction.from_pretrained(
    "my-timber",
    head={"task": "classification", "num_outputs": 2},  # the head, after the encoder
    decoders=[],                                          # drop the pretraining decoder
    patch_mask_ratio=0.0,                                 # no masking at finetuning time
)
```

For a training loop with labels, see [Training with Lightning](lightning.md).

## Load from the Hub

Models are saved in the Hugging Face format. After `import timber`, `AutoModel.from_pretrained` loads a model from the
Hub or from a folder.

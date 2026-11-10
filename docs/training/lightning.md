# Training with Lightning

`timber.lightning` wraps the Hugging Face models in [PyTorch Lightning](https://lightning.ai) modules. The wrappers
add the training loop, the optimiser and schedule, the metrics, and the loading and export of weights. The model
itself stays a plain `transformers` model, so you can always take it out with `module.model`.

## The modules

| Class | Wraps | Use it to |
|---|---|---|
| `TimberPretrainingModule` | `TimberForPretraining` | learn the representation with the pretraining objectives |
| `TimberFinetuningModule` | `TimberForPrediction` | train a head, optionally with pretraining decoders |

Both take the model configuration, the optimisation settings and, when training from scratch, the initial channel
embeddings:

```python
from timber.lightning import TimberFinetuningModule, TimberPretrainingModule

module = TimberPretrainingModule(
    config,                                   # TimberConfig: the architecture
    training={"learning_rate": 1e-4, "max_epochs": 30},  # TimberTrainingConfig fields
    channel_embedding_init=None,              # or an array of shape (num_channels, channel_embedding_dim)
)
```

## Optimisation

`TimberTrainingConfig` holds the optimisation settings. It is not saved with the model; it is kept in the Lightning
checkpoint.

| Field | Default | Meaning |
|---|---|---|
| `learning_rate` | 1e-4 | AdamW learning rate of the backbone. |
| `weight_decay` | 1e-2 | AdamW weight decay; biases and norm weights are not decayed. |
| `head_learning_rate` | `None` | Learning rate of the head, if different. |
| `latents_learning_rate` | `None` | Learning rate of the latents, if different. |
| `warmup_ratio` | 0.1 | Fraction of the steps with a linear warm-up, then a cosine decay. |
| `max_epochs` | 30 | Number of epochs. |
| `grad_norm_log_interval` | 50 | How often the gradient norm is logged, in steps. |
| `compile` | `False` | Compile the attention layers with `torch.compile`. |

Gradient clipping is set on the trainer (`gradient_clip_val`). Lightning refuses to clip the gradients of a fused
optimiser under mixed precision, so the optimiser is not fused when clipping is on.

## Training

```python
from lightning.pytorch import Trainer
from timber.device import default_precision

trainer = Trainer(max_epochs=30, precision=default_precision(), gradient_clip_val=5.0)
trainer.fit(module, datamodule=datamodule)
```

The default precision is `bf16-mixed` on CUDA and MPS and `32-true` on CPU. Losses and metrics are computed in
float32.

## Finetuning

`TimberFinetuningModule` can start from a pretrained backbone, from a Hugging Face folder or a Hub id, or from a
Lightning checkpoint of these modules:

```python
module = TimberFinetuningModule(finetune_config, training={"learning_rate": 2e-5, "head_learning_rate": 2e-4})
module.load_pretrained("my-timber")                  # a Hugging Face folder or Hub id
module.freeze_parameters(["timber.tokenizer", "timber.contextualizer", "timber.encoder.blocks.0"])
```

- `load_pretrained(source, discard=(...))` loads the weights that match. `discard` lists the prefixes not to load, such
  as the pretraining decoders, which a finetuning model does not have.
- `freeze_parameters(patterns)` freezes the parameters whose names start with a pattern. Frozen modules are kept in
  `eval()` mode during training, so their dropout is off and they run without autograd when the whole front end is
  frozen.

## Saving

```python
module.export_pretrained("my-timber")   # config.json and model.safetensors
```

The folder loads back with `TimberModel.from_pretrained("my-timber")` (or `TimberForPrediction`, `TimberForPretraining`
for the other tasks), and with `AutoModel.from_pretrained` after `import timber`.

## Reference

::: timber.lightning.base
    options:
      members:
        - TimberTrainingConfig
        - TimberLightningModule
        - cosine_schedule_with_warmup
        - load_pretrained_state_dict

::: timber.lightning.pretraining
    options:
      members:
        - TimberPretrainingModule

::: timber.lightning.finetuning
    options:
      members:
        - TimberFinetuningModule

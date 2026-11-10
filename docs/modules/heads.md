# Finetuning heads

The head (paper module **H**) turns the latents of an entity into a prediction: a regression vector or class logits.
It is what changes between tasks. The paper's method pools the latents with attention and applies a small MLP; the
fMRI experiments use mean pooling (see the table below).

## What it does

The head reads the latent embedding `e` of each view and predicts the target. It has three stages:

- **Pooling** summarises the latent queries of a view into one vector:
  - `"attention"` (the paper's method): a softmax over a learned linear score of each latent query, so that the queries
    that matter most for the target get the largest weights; the weights are returned for inspection;
  - `"mean"`: the average of the latent queries (the fMRI experiments);
  - `"weighted_mean"`: a softmax-weighted average with one learned weight per latent query;
  - `"cls_token"`: the last token, which needs `use_cls_token=True` in the encoder.
  An optional `Linear` + GELU projection (`projection_dim`) and input dropout (`pooling_input_dropout`) apply to every
  token before pooling.
- **View fusion** combines the views of an entity (recordings, channel chunks, time chunks). `fusion_stage` decides
  where: `"early"` averages the pooled views before the MLP, so there is one prediction per entity; `"late"` predicts for
  each view and averages the predictions; `"loss"` keeps one prediction per view and computes the loss on each, with the
  entity's label repeated.
- **MLP** maps the fused vector to `num_outputs` values, through `num_hidden_layers` hidden layers (0 in the experiments:
  a single linear layer) with `dropout`.

The loss depends on the task. Regression uses the **Huber** loss, as in the paper. Classification uses cross-entropy,
with `label_smoothing` (0.05 in experiment 4).

## Options that matter

| Option | Default | Experiments | Effect |
|---|---|---|---|
| `task` | `"regression"` | | `"regression"` (Huber) or `"classification"` (cross-entropy). |
| `num_outputs` | 1 | | Size of the prediction: regression targets, or classes. |
| `pooling` | `"mean"` | `"mean"` | How the latents of a view are summarised. |
| `fusion_stage` | `"late"` | | Where the views of an entity are combined. |
| `num_hidden_layers` | 0 | 0 | Hidden layers of the MLP; `0` is a linear layer. |
| `dropout` | 0.5 | 0.3 (experiments 3 to 5) | Dropout of the MLP. |
| `label_smoothing` | 0.0 | 0.05 (experiment 4) | Smoothing of the classification targets. |

## Reference

::: timber.modules.heads
    options:
      members:
        - PredictionHead
        - LatentPooling
        - MLPHead
        - fuse_views

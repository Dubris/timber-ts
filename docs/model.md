# Model reference

The models are Hugging Face `PreTrainedModel`s. `TimberModel` is the backbone that produces the embeddings; the two
task models add the pretraining decoders or the finetuning head around it. Their forward methods return the typed
output objects listed below.

::: timber.modeling_timber
    options:
      members:
        - TimberModel
        - TimberForPretraining
        - TimberForPrediction
        - TimberModelOutput
        - TimberPretrainingOutput
        - TimberPredictionOutput

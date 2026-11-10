# Configuration reference

`TimberConfig` describes the architecture of a model. It is saved as `config.json` with the weights. Each sub-configuration
is a dataclass, so unknown option names raise an error and every default is visible in the code.

::: timber.configuration_timber
    options:
      members:
        - TimberConfig
        - TokenizerConfig
        - ContextualizerConfig
        - EncoderConfig
        - DecoderConfig
        - HeadConfig

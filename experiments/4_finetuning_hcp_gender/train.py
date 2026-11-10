"""Training of experiment 4_finetuning_hcp_gender (settings in config/; see experiments.timber_fmri.training).

Run from the repository root::

    uv run python -m experiments.4_finetuning_hcp_gender.train [hydra overrides, e.g. training.max_epochs=2]
"""

import hydra
from omegaconf import DictConfig

from experiments.timber_fmri.training import train


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()

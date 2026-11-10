# %%
# Load the packages
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from experiments.timber_fmri.paths import PATHS
from experiments.timber_fmri.presets import paper_config

# %%
# Model configuration: the paper's pretraining setting (TACO), see experiments.timber_fmri.presets
config_model = {
    key: value
    for key, value in paper_config().to_dict().items()
    if key in ("tokenizer", "contextualizer", "encoder", "decoders", "head", "patch_mask_ratio")
}
config_model = OmegaConf.create(config_model)

# %%
# Define the configuration for the data module
# Filter the subjects with complete data (length 1200 for all 4 runs)
n_runs = 4
list_files = list(PATHS.hcp_data.glob("*dtseries.npy"))
list_files = [f for f in tqdm(list_files) if np.load(f, mmap_mode="r").shape[0] == 1200]
subjects, counts = np.unique(
    [path.parts[-1].split("_")[0][-6:] for path in list_files],
    return_counts=True,
)
subjects = subjects[counts == n_runs]
list_files = [str(path) for path in list_files if path.parts[-1].split("_")[0][-6:] in subjects]

# Configure the sampler
config_sampler = {
    "partition_shape": [200, n_runs, 59412, 1200],
    "chunk_shape": [1, 1, 1024, 1200],
    "chunks_per_batch": [2, 2, 2, 1],
    "shuffle_before_chunking": [True, True, True, False],
    "crop_shape": [0, 0, 0, 300],
    # labels=[None, None, labels, None],
    "shuffle_batches": True,
}

# Generate the list of datamodule configurations for different training set sizes
dict_config_datamodule = {}
seed = 42
list_size_training_set = [50, 100, 200, 400, 500]
size_validation_set = 48
size_test_set = len(subjects) - size_validation_set - max(list_size_training_set)
for size_training_set in list_size_training_set:
    # The test set is fixed for all training set sizes
    rng = np.random.RandomState(seed)
    test_subjects = rng.choice(
        subjects,
        size=size_test_set,
        replace=False,
    )
    remaining_subjects = np.setdiff1d(subjects, test_subjects)
    val_subjects = rng.choice(
        remaining_subjects,
        size=size_validation_set,
        replace=False,
    )
    train_subjects = np.setdiff1d(remaining_subjects, val_subjects)
    train_subjects = rng.choice(
        train_subjects,
        size=size_training_set,
        replace=False,
    )

    config_datamodule = {
        "num_train_subjects": size_training_set,
        "num_val_subjects": size_validation_set,
        "num_test_subjects": size_test_set,
        "sampler": config_sampler,
        "path_data": "${path:hcp_data}",
        "train_subjects": train_subjects.tolist(),
        "val_subjects": val_subjects.tolist(),
        "test_subjects": test_subjects.tolist(),
        "keep_subcortex": False,
        "path_measures": None,
        "measures": None,
        "drop_last": True,
        "num_workers": 8,
        "prefetch_factor": 2,
        "seed": 42,
    }
    dict_config_datamodule[f"size_{size_training_set}"] = OmegaConf.create(config_datamodule)

# %%
# Define the configuration of the trainer
config_trainer = {
    "accelerator": "auto",
    "devices": 1,
    "strategy": "auto",
    "precision": "bf16-mixed",
    "gradient_clip_val": 5.0,
    "accumulate_grad_batches": 1,
}

config_trainer = OmegaConf.structured(config_trainer)

# %%
# Save the config files
path_config = Path(__file__).parent
OmegaConf.save(config_model, path_config / "model" / "config_model.yaml")
OmegaConf.save(config_trainer, path_config / "trainer" / "config_trainer.yaml")
for key, config_datamodule in dict_config_datamodule.items():
    OmegaConf.save(config_datamodule, path_config / "datamodule" / f"config_datamodule_{key}.yaml")

# %%

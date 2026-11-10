# %%
# Load the packages
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.timber_fmri.loading import pretraining_subjects
from experiments.timber_fmri.paths import PATHS

df_people = pd.read_csv(PATHS.hcp_people)
df_people.Subject = df_people.Subject.astype(str)
df_people = df_people.set_index("Subject")

# %%
# Define the seed and the path to the pretrained checkpoint from the main config file
cfg = OmegaConf.load(Path(__file__).parent / "config.yaml")
SEED = cfg.seed
PATH_PRETRAINED_CHECKPOINT = cfg.pretrained

# %%
# Define the configuration for the data module
# Filter the subjects with complete data (length 1200 for all 4 runs)
n_files_per_subject = 4  # one file per run (whole cortex)
list_files = list(PATHS.hcp_data.glob("sub-*.npy"))

subjects, counts = np.unique(
    [path.parts[-1].split("_")[0][-6:] for path in list_files],
    return_counts=True,
)
subjects = subjects[counts == n_files_per_subject]
list_files = [str(path) for path in list_files if path.parts[-1].split("_")[0][-6:] in subjects]

# %%
# Configure the sampler
n_runs = 4
n_vertices = 59412
n_chunks_batch = [4, 2, 2, 2, 1]
accumulate_grad_batches = 4  # 16 subjects per step
n_steps_per_epoch = 500
max_batches = n_steps_per_epoch * accumulate_grad_batches

config_sampler = {
    "partition_shape": [1, n_runs, n_vertices, 1200, 1],
    "chunk_shape": [1, 1, 1024, 1200, 1],
    "chunks_per_batch": n_chunks_batch,
    "shuffle_before_chunking": [True, True, True, False, True],
    "crop_shape": [0, 0, 0, 300, 0],
    # labels=[None, None, labels, None],
    "shuffle_batches": True,
    "max_batches": max_batches,
    "seed": SEED,
}

# %%
# Generate the list of datamodule configurations for different splits
# We fix the ration train/validation/test to 70/15/15
# The train set includes the subjects from the pre-training
n_splits = 3
ratio_train = 0.7
ratio_val = 0.15
ratio_test = 0.15
list_measures = ["Gender"]

subjects = df_people.loc[subjects][list_measures].dropna().index.tolist()
size_subjects = len(subjects)
size_training_set = int(size_subjects * ratio_train)
size_validation_set = int(size_subjects * ratio_val)
size_test_set = size_subjects - size_training_set - size_validation_set

subjects_pretraining = pretraining_subjects(PATH_PRETRAINED_CHECKPOINT)
subjects_pretraining = [sub for sub in subjects_pretraining if sub in subjects]
subjects_available = np.setdiff1d(subjects, subjects_pretraining)

dict_config_datamodule = {}
for i in range(n_splits):
    rng = np.random.RandomState(seed=SEED)
    subjects_test = rng.choice(
        subjects_available,
        size=size_test_set,
        replace=False,
    )
    subjects_available = np.setdiff1d(subjects_available, subjects_test)
    subjects_val = rng.choice(
        np.setdiff1d(subjects, subjects_test),
        size=size_validation_set,
        replace=False,
    )
    subjects_train = np.setdiff1d(subjects, np.concatenate((subjects_test, subjects_val)))

    config_datamodule = {
        "num_train_subjects": size_training_set,
        "num_val_subjects": size_validation_set,
        "num_test_subjects": size_test_set,
        "sampler": config_sampler,
        "path_data": "${path:hcp_data}",
        "train_subjects": subjects_train.tolist(),
        "val_subjects": subjects_val.tolist(),
        "test_subjects": subjects_test.tolist(),
        "keep_subcortex": False,
        "path_measures": "${path:hcp_people}",
        "measures": list_measures,
        "drop_last": False,
        "use_batch_sampler": False,
        "num_workers": 8,
        "prefetch_factor": 2,
        "seed": SEED,
    }
    dict_config_datamodule[f"split_{i + 1}"] = OmegaConf.create(config_datamodule)

# %%
# Define the configuration of the trainer
config_trainer = {
    "accelerator": "auto",
    "devices": 1,
    "strategy": "auto",
    "precision": "bf16-mixed",
    "gradient_clip_val": 5.0,
    "accumulate_grad_batches": accumulate_grad_batches,
}

config_trainer = OmegaConf.structured(config_trainer)

# %%
# Save the config files
path_config = Path(__file__).parent
(path_config / "trainer").mkdir(exist_ok=True)
(path_config / "datamodule").mkdir(exist_ok=True)

OmegaConf.save(config_trainer, path_config / "trainer" / "config_trainer.yaml")
for key, config_datamodule in dict_config_datamodule.items():
    OmegaConf.save(config_datamodule, path_config / "datamodule" / f"config_datamodule_{key}.yaml")

# %%

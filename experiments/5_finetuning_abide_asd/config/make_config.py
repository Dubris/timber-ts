# %%
# Load the packages
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from experiments.timber_fmri.paths import PATHS

df_people = pd.read_csv(PATHS.abide_measures, dtype={"Subject": str})
df_people = df_people.set_index("Subject")

path_data_key = "abide_data"  # whole files, listed explicitly
path_data = PATHS[path_data_key]
n_vertices = 59412
n_files_per_subject = 1  # one file per run (whole cortex)
list_measures = ["DX_GROUP"]

# %%
# Define the seed and the path to the pretrained checkpoint from the main config file
cfg = OmegaConf.load(Path(__file__).parent / "config.yaml")
SEED = cfg.seed
PATH_PRETRAINED_CHECKPOINT = cfg.pretrained

# %%
# Define the configuration for the data module
# Filter the subjects with complete data (length 1200 for all 4 runs)
list_files = list(path_data.glob("sub-*.npy"))
list_files = sorted(list_files)

subjects, counts = np.unique(
    [path.parts[-1].split("_")[0][4:] for path in list_files],
    return_counts=True,
)
list_files = [str(path) for path in list_files if path.parts[-1].split("_")[0][4:] in subjects]
df_people = df_people.loc[subjects]

# %%
# Configure the sampler
n_chunks_batch = [8, 1, 2, 2, 1]
accumulate_grad_batches = 2  # 16 subjects per step
n_steps_per_epoch = 500
max_batches = n_steps_per_epoch * accumulate_grad_batches

config_sampler = {
    "partition_shape": [1, 1, n_vertices, 1200, n_files_per_subject],
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
# We generate 10-fold splits stratified on the site and condition
# The validation set contains 20% of the non-test fold,
n_folds = 5
ratio_val = 0.25
groups_condition = df_people["DX_GROUP"].to_numpy()
groups_condition_site = (df_people["SITE_ID"] + "_" + df_people["DX_GROUP"].astype(str)).to_numpy()

outer_loop = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
inner_loop = StratifiedShuffleSplit(n_splits=1, test_size=ratio_val, random_state=SEED)

dict_config_datamodule = {}
for i, split in enumerate(outer_loop.split(subjects, groups_condition_site)):
    subjects_train_val = subjects[split[0]]
    subjects_test = subjects[split[1]]
    split_inner = next(inner_loop.split(subjects_train_val, groups_condition[split[0]]))
    subjects_train = subjects_train_val[split_inner[0]]
    subjects_val = subjects_train_val[split_inner[1]]

    config_datamodule = {
        "num_train_subjects": len(subjects_train),
        "num_val_subjects": len(subjects_val),
        "num_test_subjects": len(subjects_test),
        "sampler": config_sampler,
        "path_data": f"${{path:{path_data_key}}}",
        "files": [str(p) for p in list_files] if path_data_key == "abide_data" else None,
        "train_subjects": subjects_train.tolist(),
        "val_subjects": subjects_val.tolist(),
        "test_subjects": subjects_test.tolist(),
        "keep_subcortex": False,
        "path_measures": "${path:abide_measures}",
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
    "gradient_clip_val": 1.0,
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

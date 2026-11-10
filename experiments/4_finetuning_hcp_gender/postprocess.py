# %%
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.timber_fmri import FmriDataset, find_files, load_measures
from experiments.timber_fmri.analysis import get_schaefer_atlas, sample_model
from experiments.timber_fmri.loading import load_model
from experiments.timber_fmri.paths import PATHS
from timber.data import SamplerConfig, build_dataloader

PATH_DATA = PATHS.hcp_data
N_FILES_PER_RUN = 1
N_VERTICES_PER_FILE = 59_412
N_CHUNKS_PER_BATCH = [1, 2, 7, 2, 1]


# %%
def postprocess(path_ckpt: str, path_save: str | None = None) -> None:
    # Check path_save
    if path_save is None:
        path_ckpt = Path(path_ckpt)
        path_dir = next(iter(path_ckpt.parents))
        file_name = path_ckpt.parts[-1].replace("checkpoint_", "samples_")
        path_save = (path_dir / file_name).with_suffix(".pkl")

    # Load the time-series
    list_dtseries = list(PATH_DATA.glob("sub-*.npy"))
    # list_dtseries = [
    #     f for f in tqdm(list_dtseries)
    #     if np.load(f, mmap_mode="r").shape[0] == 1200
    # ]
    subjects, counts = np.unique(
        [path.parts[-1].split("_")[0][-6:] for path in list_dtseries],
        return_counts=True,
    )
    subjects = subjects[counts == 4 * N_FILES_PER_RUN]
    list_dtseries = [path for path in list_dtseries if path.parts[-1].split("_")[0][-6:] in subjects]
    print(f"Number of subjects: {len(subjects)}.")

    # Load the model (HF folder or Lightning checkpoint); the analysis does not mask patches
    model = load_model(path_ckpt)
    print(f"Model type: {type(model).__name__}")

    # Create the dataloader (all HCP subjects)
    list_measures = pd.read_csv(PATHS.hcp_measures, sep=";")["hcp_field"].tolist()
    list_measures += [
        "Gender",
        "CogFluidComp_AgeAdj",
        "CogCrystalComp_AgeAdj",
        "CogTotalComp_AgeAdj",
    ]

    config_sampler = {
        "partition_shape": [len(subjects), 4, N_VERTICES_PER_FILE, 1200, N_FILES_PER_RUN],
        "chunk_shape": [1, 1, 1024, 1200, 1],
        "chunks_per_batch": N_CHUNKS_PER_BATCH,
        "shuffle_before_chunking": [True, True, True, False, True],
        "crop_shape": [0, 0, 0, 300, 0],
        "shuffle_batches": False,
    }

    files = find_files(files=list_dtseries)
    labels, _ = load_measures(PATHS.hcp_people, list_measures, drop_incomplete=False)
    dataset = FmriDataset(files, labels=labels, measure_names=list_measures)
    dataloader = build_dataloader(
        dataset,
        SamplerConfig(**config_sampler),
        num_workers=4,
        prefetch_factor=2,
    )

    # Sample from the model
    labels_vertex, _ = get_schaefer_atlas(n_rois_atlas=800)
    return sample_model(
        model,
        dataloader,
        save_path=path_save,
        include=[
            "signal_space",
            # "latent_embeddings_1",
            "latent_embeddings_2",
            "head_classification",
        ],
        labels_vertex=labels_vertex,
        dims_to_aggregate={
            "timepatch": "mean",
        },
    )


# %%
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Slurm batch Python script example")
    parser.add_argument("--path_ckpt", type=str, help="Path to the checkpoint file")
    parser.add_argument("--path_save", type=str, help="Saving path", default=None)
    args = parser.parse_args()
    postprocess(args.path_ckpt, args.path_save)

    # path_ckpt = PATHS.results / "remote_data/3_finetuning_intelligence/checkpoint_split-1_epoch-7.ckpt"
    # path_save = PATHS.results / "remote_data/3_finetuning_intelligence/samples_test.pkl"
    # foo = postprocess(path_ckpt, path_save)

# %%

# %%
import argparse
import logging
from pathlib import Path

import numpy as np

from experiments.timber_fmri import NUM_CORTICAL_VERTICES, FmriDataset, find_files
from experiments.timber_fmri.analysis import get_schaefer_atlas, sample_model
from experiments.timber_fmri.loading import load_model
from experiments.timber_fmri.paths import PATHS
from timber.data import SamplerConfig, build_dataloader

PATH_DATA = PATHS.abide_data
N_FILES_PER_RUN = 1
N_VERTICES_PER_FILE = 59412
N_CHUNKS_PER_BATCH = [1, 1, 29, 1, 1]


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
        [path.parts[-1].split("_")[0][4:] for path in list_dtseries],
        return_counts=True,
    )
    subjects = subjects[counts == N_FILES_PER_RUN]  # 1 run per subject
    list_dtseries = [path for path in list_dtseries if path.parts[-1].split("_")[0][4:] in subjects]
    print(f"Number of subjects: {len(subjects)}.")

    # Load the model (HF folder or Lightning checkpoint); the analysis does not mask patches
    model = load_model(path_ckpt)
    print(f"Model type: {type(model).__name__}")

    # Create the dataloader (all ABIDE subjects)
    config_sampler = {
        "partition_shape": [len(subjects), 1, N_VERTICES_PER_FILE, 1200, N_FILES_PER_RUN],
        "chunk_shape": [1, 1, 1024, 1200, 1],
        "chunks_per_batch": N_CHUNKS_PER_BATCH,
        "shuffle_before_chunking": [True, True, True, False, True],
        "crop_shape": [0, 0, 0, 300, 0],
        "shuffle_batches": False,
    }

    files = find_files(files=list_dtseries)
    dataset = FmriDataset(
        files,
        resample_time_window=True,
        resample_channels=True,
        num_resample_channels=NUM_CORTICAL_VERTICES,  # keep_subcortex=False
    )
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
            "latent_embeddings_1",
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

    # path_ckpt = PATHS.results / "remote_data/1_learning_curve/checkpoint_epoch-29_train-200.ckpt"
    # path_save = PATHS.results / "remote_data/1_learning_curve/samples_test.pkl"
    # foo = postprocess(path_ckpt, path_save)

# %%

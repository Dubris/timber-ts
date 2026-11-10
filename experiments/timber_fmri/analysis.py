"""Sample embeddings, attention maps and predictions of a TIMBER model over a whole fMRI dataset.

Results are ``xarray`` datasets indexed by subject, run, vertex (optionally pooled into atlas regions)
and latent token, saved as pickles for the analysis notebooks.

Differences with the former post-processing code:

- the model runs without patch masking (``mask_patches=False``), as the former post-processing did;
- attention maps are softmax *probabilities* (the former monkey patch of transformers returned the
  pre-softmax scores);
- the device is CUDA, then MPS, then CPU.
"""

# %%
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import hcp_utils as hcp
import nibabel as nib
import numpy as np
import torch
import xarray as xr
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from timber.data import SamplePositions
from timber.device import get_accelerator

from .paths import PATHS

logger = logging.getLogger(__name__)


# %%
def get_schaefer_atlas(
    n_rois_atlas: int = 400,
    schaefer_atlas_path: Path | None = None,
):
    """Load the Schaefer atlas with the specified number of ROIs.

    Parameters
    ----------
    n_rois_atlas (int): Number of ROIs in the Schaefer atlas.
    schaefer_atlas_path (Path): Directory of the Schaefer atlas files (default: ``PATHS.schaefer_atlas``).

    Returns
    -------
    np.ndarray: The Schaefer atlas data for the specified number of ROIs.

    """
    if schaefer_atlas_path is None:
        schaefer_atlas_path = PATHS.schaefer_atlas
    data = nib.load(
        schaefer_atlas_path / f"Schaefer2018_{n_rois_atlas}Parcels_17Networks_order.dlabel.nii",
    )

    mask_medial = hcp.cortex_data(59_412 * [1])
    atlas = data.get_fdata().squeeze()[mask_medial.astype(bool)]
    labels = (n_rois_atlas + 1) * [""]
    for k, v in data.header.get_axis(0).label[0].items():
        labels[k] = v[0]
    return atlas, labels


# %%
@dataclass
class Activations:
    """What the samplers read from one forward pass (views are flattened into the batch axis).

    Patch tensors are ``(batch, channel * patch, dim)``; the encoder tuples hold one entry per block:
    the latents ``(batch, latent, dim)``, the last self-attention layer and the cross-attention of the block.
    The head fields are ``None`` unless the model has a head (see ``pooling_weights`` for its attention).
    """

    inputs: torch.Tensor  # contextualised patches
    signal_space: torch.Tensor  # contextualised patches before the output projection
    latents: tuple[torch.Tensor, ...]
    self_attentions: tuple[torch.Tensor, ...]
    cross_attentions: tuple[torch.Tensor, ...]
    prediction: torch.Tensor | None = None  # regression head
    classification: torch.Tensor | None = None  # classification head
    pooled: torch.Tensor | None = None  # pooled latents of the head (``cls_token`` in the results)
    pooling_weights: torch.Tensor | None = None


# Sampled result name -> attribute of `Activations`
HEAD_ATTRIBUTES = {
    "head_classification": "classification",
    "head_prediction": "prediction",
    "cls_token": "pooled",
    "head_attention_weights": "pooling_weights",
}


@torch.no_grad()
def model_outputs(model, signal: torch.Tensor, positions: SamplePositions) -> Activations:
    """One forward pass without patch masking, returning the `Activations` the samplers read."""
    timber = getattr(model, "timber", model)
    captured = {}
    projection = timber.contextualizer.output_projection
    hook = projection.register_forward_pre_hook(lambda _, args: captured.update(hidden=args[0]))
    try:
        out = timber(signal, positions.channel, positions.timestep, mask_patches=False, output_attentions=True)
    finally:
        hook.remove()

    activations = Activations(
        inputs=out.contextualized.flatten(1, 2),
        signal_space=captured["hidden"].flatten(1, 2),
        latents=tuple(out.block_latents or ()),
        self_attentions=tuple(out.self_attentions or ()),
        cross_attentions=tuple(out.cross_attentions or ()),
    )
    head = getattr(model, "head", None)
    if head is not None:
        latents = out.block_latents[model.latent_block - 1]
        predicted = head(latents, num_entities=signal.shape[0])
        if model.config.head.task == "classification":
            activations.classification = predicted["prediction"]
        else:
            activations.prediction = predicted["prediction"]
        activations.pooled = predicted["pooled"]
        activations.pooling_weights = predicted["pooling_weights"]
    return activations


def _to_device(positions: SamplePositions, device: str) -> SamplePositions:
    return SamplePositions(*(p.to(device) for p in positions))


def _public_attributes(obj) -> dict:
    return {k: v for k, v in vars(obj).items() if not k.startswith("_") and not callable(v)}


def _measure_coords(dataset) -> dict:
    names = getattr(dataset, "measure_names", None)
    if dataset.labels is None or not names:
        return {}
    values = np.stack([dataset.labels[subject] for subject in dataset.subjects])
    return {name: ("subject", values[:, i]) for i, name in enumerate(names)}


# %%
@torch.no_grad()
def sample_model(
    model,
    dataloader: DataLoader,
    save_path: Path | None = None,
    include: tuple[str, ...] = ("latent_embeddings_2",),
    labels_vertex: list[int] | None = None,
    dims_to_aggregate: dict[str, str] | None = None,
):
    device = get_accelerator()
    model = model.to(device)
    model.eval()

    # Setting up the save path for results
    save_path = Path(save_path if save_path is not None else ".")
    if save_path.is_dir():
        save_path = save_path / "sampled_data.nc"
        logger.info("Computing %s...", save_path)

    # Initializing the variables for partitioning and chunking
    size_partition = dataloader.sampler.partition_shape
    n_spacepoints = dataloader.sampler.chunk_shape[2]
    n_chunks_batch = dataloader.sampler.chunks_per_batch
    n_sub, n_run, n_spacechunks = n_chunks_batch[:3]
    if len(n_chunks_batch) == 5:
        n_spacechunks *= n_chunks_batch[4]  # Multiplied by the number of files per run

    # Initializing the results dictionary for various outputs
    results = {
        name: None
        for name in [
            "signal_space",
            "inputs",
            "latent_embeddings_1",
            "latent_embeddings_2",
            "self_attentions_1",
            "self_attentions_2",
            "cross_attentions_1",
            "cross_attentions_2",
            "cls_token",
            "head_classification",
            "head_prediction",
        ]
        if name in include
    }
    iterations = {}

    # If labels_vertex is provided, convert to numpy array and get unique count
    if labels_vertex is not None:
        labels_vertex = np.array(labels_vertex, dtype=np.int32)
        n_labels_vertex = np.unique(labels_vertex).size

    # Launch the sampling loop over the dataloader
    for data, positions, _ in tqdm(dataloader):
        data = data.to(device)
        idx = _to_device(positions, device)

        # Forward pass through the model
        outputs = model_outputs(model, data, idx)

        # Compute the indices at once and transfer them to cpu
        indices = {
            "sub": idx[0][:, 0, 0, 0, 0].cpu().numpy(),
            "run": idx[1][0, :, 0, 0, 0].cpu().numpy(),
            "mesh": idx[2][0, 0, :, 0].cpu().numpy(),
        }

        # Sampling from the contextualizer
        for key in ["signal_space", "inputs"]:
            if key in results:
                data = getattr(outputs, key)
                if results[key] is None:
                    _, length_sequence, n_channels = data.shape
                    results[key] = xr.DataArray(
                        np.zeros(
                            (
                                *size_partition[:2],
                                size_partition[3] if labels_vertex is None else n_labels_vertex,
                                length_sequence // n_spacepoints,
                                n_channels,
                            ),
                            dtype=np.float32,
                        ),
                        dims=[
                            "subject",
                            "run",
                            "vertex",
                            "timepatch",
                            f"channel_{key}",
                        ],
                    )

                    if labels_vertex is not None:
                        iterations[key] = xr.DataArray(
                            np.zeros(
                                (
                                    *size_partition[:2],
                                    n_labels_vertex,
                                ),
                                dtype=np.float32,
                            ),
                            dims=["subject", "run", "vertex"],
                        )

                # Rearranging the data and the indices
                data = (
                    rearrange(
                        data,
                        "(sub run sch tch) (spo tpa) d -> sub run (sch spo) tch tpa d",
                        sub=n_sub,
                        run=n_run,
                        sch=n_spacechunks,
                        spo=n_spacepoints,
                    )
                    .cpu()
                    .numpy()
                )
                data = data.mean(-3)

                idx_sub = indices["sub"]
                idx_run = indices["run"]
                idx_mesh = indices["mesh"].reshape(-1)

                # Apply vertex labels transformation if needed
                if labels_vertex is not None:
                    idx_mesh = labels_vertex[idx_mesh]

                # Create indexing tuple once and reuse
                idx_tuple = np.ix_(idx_sub, idx_run, idx_mesh)

                # Update arrays
                if labels_vertex is not None:
                    iterations[key].values[idx_tuple] += 1
                results[key].values[idx_tuple] += data

        # Sampling from the heads
        for head_label in ["head_classification", "head_prediction", "cls_token"]:
            # Process the data for this head if it exists in the outputs
            data = getattr(outputs, HEAD_ATTRIBUTES[head_label])
            if data is None or head_label not in results:  # only the requested outputs
                continue
            data = (
                rearrange(
                    data,
                    "(sub views) dim -> sub views dim",
                    sub=n_sub,
                )
                .cpu()
                .numpy()
            )

            if results[head_label] is None:
                # Initialize the results and iterations DataArrays for this head
                results[head_label] = xr.DataArray(
                    np.zeros(
                        (
                            size_partition[0],
                            data.shape[-1],
                        ),
                        dtype=np.float32,
                    ),
                    dims=[
                        "subject",
                        f"channel_{head_label}",
                    ],
                )

                iterations[head_label] = xr.DataArray(
                    np.zeros(
                        (size_partition[0],),
                        dtype=np.float32,
                    ),
                    dims=["subject"],
                )

            idx_sub = indices["sub"]
            idx_run = indices["run"]

            # Sum over the views
            data_processed = data.sum(1)

            # Use broadcasting-friendly indexing
            # Create index tuples for all combinations
            sub_idx_expanded = idx_sub[:, None]  # Shape: (n_sub, 1)

            # Vectorized assignment using broadcasting
            results[head_label].values[sub_idx_expanded,] += data_processed

            # Update counters for all (sub, run) pairs at once
            iterations[head_label].values[sub_idx_expanded,] += data.shape[1]

        # Sampling from the encoder
        num_blocks = model.config.encoder.num_blocks
        for idx_block in range(1, num_blocks + 1):
            # Computing the latent embeddings
            if f"latent_embeddings_{idx_block}" in results and idx_block <= len(outputs.latents):
                data = outputs.latents[idx_block - 1]

                if results[f"latent_embeddings_{idx_block}"] is None:
                    results[f"latent_embeddings_{idx_block}"] = xr.DataArray(
                        np.zeros(
                            (*size_partition[:2], *data.shape[-2:]),
                            dtype=np.float32,
                        ),
                        dims=[
                            "subject",
                            "run",
                            f"latent_{idx_block}",
                            "channel_latent",
                        ],
                    )
                    iterations[f"latent_embeddings_{idx_block}"] = xr.DataArray(
                        np.zeros(
                            (*size_partition[:2],),
                            dtype=np.float32,
                        ),
                        dims=["subject", "run"],
                    )

                # Rearranging the data and indices
                data = (
                    rearrange(
                        data,
                        "(sub run sch tch) l d -> sub run sch tch l d",
                        sub=n_sub,
                        run=n_run,
                        sch=n_spacechunks,
                    )
                    .cpu()
                    .numpy()
                )
                data = data.mean(3)  # Average over time chunks (tch)

                idx_sub = indices["sub"]
                idx_run = indices["run"]
                n_spatial_chunks = data.shape[2]

                # Sum over spatial chunks and reshape for vectorized operations
                data_processed = data.sum(2)  # Sum spatial chunks: (n_sub, n_run, h, l1, l2)

                # Use broadcasting-friendly indexing
                # Create index tuples for all combinations
                sub_idx_expanded = idx_sub[:, None]  # Shape: (n_sub, 1)
                run_idx_expanded = idx_run[None, :]  # Shape: (1, n_run)

                # Vectorized assignment using broadcasting
                results[f"latent_embeddings_{idx_block}"].values[sub_idx_expanded, run_idx_expanded] += data_processed

                # Update counters for all (sub, run) pairs at once
                iterations[f"latent_embeddings_{idx_block}"].values[sub_idx_expanded, run_idx_expanded] += (
                    n_spatial_chunks
                )

            # Computing the self-attention for latent embeddings (middle layer)
            if f"self_attentions_{idx_block}" in results and idx_block <= len(outputs.latents):
                data = outputs.self_attentions[idx_block - 1]  # Select the middle layer attention

                if results[f"self_attentions_{idx_block}"] is None:
                    results[f"self_attentions_{idx_block}"] = xr.DataArray(
                        np.zeros(
                            (*size_partition[:2], *data.shape[-3:]),
                            dtype=np.float32,
                        ),
                        dims=[
                            "subject",
                            "run",
                            "head_self",
                            f"latent_{idx_block}",
                            f"latent_{idx_block}_bis",
                        ],
                    )
                    iterations[f"self_attentions_{idx_block}"] = xr.DataArray(
                        np.zeros(
                            (*size_partition[:2],),
                            dtype=np.float32,
                        ),
                        dims=["subject", "run"],
                    )

                # Rearranging the data and indices
                data = (
                    rearrange(
                        data,
                        "(sub run sch tch) h l1 l2 -> sub run sch tch h l1 l2",
                        sub=n_sub,
                        run=n_run,
                        sch=n_spacechunks,
                    )
                    .cpu()
                    .numpy()
                )
                data = data.mean(3)  # Average over time chunks (tch)

                idx_sub = indices["sub"]
                idx_run = indices["run"]
                n_spatial_chunks = data.shape[2]

                # Sum over spatial chunks and reshape for vectorized operations
                data_processed = data.sum(2)  # Sum spatial chunks: (n_sub, n_run, h, l1, l2)

                # Use broadcasting-friendly indexing
                # Create index tuples for all combinations
                sub_idx_expanded = idx_sub[:, None]  # Shape: (n_sub, 1)
                run_idx_expanded = idx_run[None, :]  # Shape: (1, n_run)

                # Vectorized assignment using broadcasting
                results[f"self_attentions_{idx_block}"].values[sub_idx_expanded, run_idx_expanded] += data_processed

                # Update counters for all (sub, run) pairs at once
                iterations[f"self_attentions_{idx_block}"].values[sub_idx_expanded, run_idx_expanded] += (
                    n_spatial_chunks
                )

            # Compute the cross-attentions
            if f"cross_attentions_{idx_block}" in results and idx_block <= len(outputs.latents):
                data = outputs.cross_attentions[idx_block - 1]

                if results[f"cross_attentions_{idx_block}"] is None:
                    results[f"cross_attentions_{idx_block}"] = xr.DataArray(
                        np.zeros(
                            (
                                *size_partition[:2],
                                size_partition[3] if labels_vertex is None else n_labels_vertex,
                                data.shape[-1] // n_spacepoints,
                                *data.shape[1:3],
                            ),
                            dtype=np.float32,
                        ),
                        dims=[
                            "subject",
                            "run",
                            "vertex",
                            "timepatch",
                            "head_cross",
                            f"latent_{idx_block}",
                        ],
                    )

                    if labels_vertex is not None:
                        iterations[f"cross_attentions_{idx_block}"] = xr.DataArray(
                            np.zeros(
                                (
                                    *size_partition[:2],
                                    size_partition[3] if labels_vertex is None else n_labels_vertex,
                                ),
                                dtype=np.float32,
                            ),
                            dims=["subject", "run", "vertex"],
                        )

                # Rearranging and averaging the cross-attentions
                data = (
                    rearrange(
                        data,
                        "(sub run sch tch) h l (spo tpa) -> sub run (sch spo) tch tpa h l",
                        sub=n_sub,
                        run=n_run,
                        sch=n_spacechunks,
                        spo=n_spacepoints,
                    )
                    .cpu()
                    .numpy()
                )
                data = data.mean(3)  # Average over time chunks (tch)

                idx_sub = indices["sub"]
                idx_run = indices["run"]
                idx_mesh = indices["mesh"].reshape(-1)

                # Apply vertex labels transformation if needed
                if labels_vertex is not None:
                    idx_mesh = labels_vertex[idx_mesh]

                idx_tuple = np.ix_(idx_sub, idx_run, idx_mesh)

                # Update arrays
                if labels_vertex is not None:
                    iterations[f"cross_attentions_{idx_block}"].values[idx_tuple] += 1

                results[f"cross_attentions_{idx_block}"].values[idx_tuple] += data

    # Averaging the values within the regions defined by `labels_vertex`
    for name, n_iterations in iterations.items():
        results[name] = results[name] / n_iterations

    # Reducing the results along iteration dimension if requested
    dims_to_aggregate = dims_to_aggregate if dims_to_aggregate is not None else {}
    for dim, aggregate in dims_to_aggregate.items():
        for key, value in results.items():
            if value is None:
                continue
            if dim in value.dims:
                if aggregate == "mean":
                    results[key] = value.mean(dim=dim)
                elif aggregate == "mean_std":
                    mean = value.mean(dim=dim)
                    std = value.std(dim=dim)
                    results[key] = xr.concat([mean, std], dim="aggregation")
                    results[key] = results[key].assign_coords(
                        aggregation=["mean", "std"],
                    )
                elif aggregate == "max":
                    results[key] = value.max(dim=dim)

    # Merging the results into a Dataset
    logger.info("Merging and saving results...")
    results = xr.Dataset(
        results,
        coords={
            "subject": dataloader.dataset.subjects,
        },
    )

    # Adding behavioral measure coordinates (if they exist)
    results = results.assign_coords(_measure_coords(dataloader.dataset))
    results = results.assign_attrs(**_public_attributes(dataloader.sampler))
    results = results.assign_attrs(**_public_attributes(dataloader.dataset))
    results = results.assign_attrs(labels_vertex=labels_vertex)

    # Saving the results to file if save_path is provided
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        # res.to_netcdf(save_path)

    return results


# %%
@torch.no_grad()
def sample_model_finetuning(
    model,
    dataloader: DataLoader,
    save_path: Path | None = None,
    include: tuple[str, ...] = ("latent_embeddings_2",),
    dims_to_aggregate: dict[str, str] | None = None,
):
    device = get_accelerator()
    model = model.to(device)
    model.eval()

    # Switching off the batch shuffling to make the sampling deterministic and easier wrt the indices
    dataloader.sampler.shuffle_batches = False

    # Setting up the save path for results
    save_path = Path(save_path if save_path is not None else ".")
    if save_path.is_dir():
        save_path = save_path / "sampled_data.nc"
        logger.info("Computing %s...", save_path)

    # Initializing the variables for partitioning and chunking
    size_partition = dataloader.sampler.partition_shape
    n_subjects, n_runs = size_partition[:2]
    n_chunks_batch = dataloader.sampler.chunks_per_batch
    n_sub, n_run = n_chunks_batch[:2]
    n_views_batch = np.prod(n_chunks_batch[2:]).item()
    n_views = (np.array(dataloader.sampler.groups_per_axis[2:]) * np.array(n_chunks_batch[2:])).prod().item()

    # Initializing the results dictionary for various outputs
    results = {
        name: None
        for name in [
            "latent_embeddings_1",
            "latent_embeddings_2",
            "cls_token",
            "head_attention_weights",
            "head_classification",
            "head_prediction",
        ]
        if name in include
    }

    # Initializing the view index DataArray to keep track of the current view index for each (subject, run) pair
    da_idx_view = xr.DataArray(
        np.zeros((n_subjects, n_runs), dtype=np.float32),
        dims=["subject", "run"],
    )

    # Launch the sampling loop over the dataloader
    for data, positions, _ in tqdm(dataloader):
        data = data.to(device)
        idx = _to_device(positions, device)

        # Forward pass through the model
        outputs = model_outputs(model, data, idx)

        # Compute the indices at once and transfer them to cpu
        idx_sub = idx[0][:, 0, 0, 0, 0].cpu().numpy()
        idx_run = idx[1][0, :, 0, 0, 0].cpu().numpy()
        idx_view = da_idx_view.loc[idx_sub, idx_run].values
        idx_view = np.unique(idx_view).astype(int).item()  # Requires shuffle_batches = False

        # Use broadcasting-friendly indexing
        idx_sub_expanded = idx_sub[:, None, None]  # Shape: (n_sub, 1, 1)
        idx_run_expanded = idx_run[None, :, None]  # Shape: (1, n_run, 1)
        idx_view_expanded = np.arange(idx_view, idx_view + n_views_batch)[None, None, :]  # Shape: (1, 1, n_views)

        # Sampling from the heads
        for head_label in ["head_classification", "head_prediction", "cls_token", "head_attention_weights"]:
            # Process the data for this head if it exists in the outputs
            data = getattr(outputs, HEAD_ATTRIBUTES[head_label])
            if data is None or head_label not in results:  # only the requested outputs
                continue

            data = rearrange(
                data.cpu().numpy(),
                "(s r v) ... -> s r v ...",
                s=n_sub,
                r=n_run,
            )

            if results[head_label] is None:
                # Initialize the results for this head
                results[head_label] = xr.DataArray(
                    np.zeros(
                        (
                            n_subjects,
                            n_runs,
                            n_views,
                            data.shape[-1],
                        ),
                        dtype=np.float32,
                    ),
                    dims=[
                        "subject",
                        "run",
                        "view",
                        f"channel_{head_label}",
                    ],
                )

            # Vectorized assignment using broadcasting
            results[head_label].values[
                idx_sub_expanded,
                idx_run_expanded,
                idx_view_expanded,
            ] = data

        # Sampling from the encoder
        num_blocks = model.config.encoder.num_blocks
        for idx_block in range(1, num_blocks + 1):
            # Computing the latent embeddings
            if f"latent_embeddings_{idx_block}" in results and idx_block <= len(outputs.latents):
                data = outputs.latents[idx_block - 1]

                if results[f"latent_embeddings_{idx_block}"] is None:
                    results[f"latent_embeddings_{idx_block}"] = xr.DataArray(
                        np.zeros(
                            (n_subjects, n_runs, n_views, *data.shape[-2:]),
                            dtype=np.float32,
                        ),
                        dims=[
                            "subject",
                            "run",
                            "view",
                            f"latent_{idx_block}",
                            "channel_latent",
                        ],
                    )

                # Rearranging the data and indices
                data = (
                    rearrange(
                        data,
                        "(s r v) ... -> s r v ...",
                        s=n_sub,
                        r=n_run,
                    )
                    .cpu()
                    .numpy()
                )

                # Vectorized assignment using broadcasting
                results[f"latent_embeddings_{idx_block}"].values[
                    idx_sub_expanded, idx_run_expanded, idx_view_expanded
                ] = data

        # Update the view index for this (sub, run) pair
        da_idx_view.loc[idx_sub, idx_run] += n_views_batch

    # Reducing the results along iteration dimension if requested
    dims_to_aggregate = dims_to_aggregate if dims_to_aggregate is not None else {}
    for dim, aggregate in dims_to_aggregate.items():
        for key, value in results.items():
            if value is None:
                continue
            if dim in value.dims:
                if aggregate == "mean":
                    results[key] = value.mean(dim=dim)
                elif aggregate == "mean_std":
                    mean = value.mean(dim=dim)
                    std = value.std(dim=dim)
                    results[key] = xr.concat([mean, std], dim="aggregation")
                    results[key] = results[key].assign_coords(
                        aggregation=["mean", "std"],
                    )
                elif aggregate == "max":
                    results[key] = value.max(dim=dim)

    # Merging the results into a Dataset
    logger.info("Merging and saving results...")
    results = xr.Dataset(
        results,
        coords={
            "subject": dataloader.dataset.subjects,
        },
    )

    # Adding behavioral measure coordinates (if they exist)
    results = results.assign_coords(_measure_coords(dataloader.dataset))
    results = results.assign_attrs(**_public_attributes(dataloader.sampler))
    results = results.assign_attrs(**_public_attributes(dataloader.dataset))

    # Saving the results to file if save_path is provided
    if save_path is not None:
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        # res.to_netcdf(save_path)

    return results

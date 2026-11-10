# %%
from pathlib import Path

import brainspace.mesh as mesh_utils
import hcp_utils as hcp
import numpy as np
from sklearn.manifold import SpectralEmbedding


# %%
def compute_and_save_spectral_embeddings(
    adjacency: np.ndarray,
    output_path: str | Path = "spectral_embeddings.npy",
    n_components: int = 64,
    n_neighbors: int = 10,
) -> None:
    """Compute spectral embeddings from a given adjacency matrix and saves them to a file.

    Args:
        adjacency (array-like or sparse matrix): Precomputed adjacency matrix representing the graph structure. Shape
        (n_samples, n_samples).
        output_path (str, optional): Path to the file where the computed spectral embeddings will be saved (in NumPy
        .npy format). Defaults to "spectral_embeddings.npy".
        n_components (int, optional): Number of dimensions for the spectral embedding. Defaults to 64.
        n_neighbors (int, optional): Number of nearest neighbors to use when constructing the affinity matrix. Defaults
        to 10.

    Returns:
        None: The function saves the computed spectral embeddings to the specified output file.

    Notes:
        The embeddings are normalized to have unit norm along the last axis before saving.

    """
    spectral = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        n_neighbors=n_neighbors,
    )
    embeddings = spectral.fit_transform(adjacency)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    np.save(output_path, embeddings)


def create_random_parcellation(
    save_path: str | Path,
    n_regions: int = 1024,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Create a random parcellation of the cortical surface into a specified number of regions and saves the
    resulting labels to a file.

    Args:
        save_path (Union[str, Path]): Path to save the resulting parcellation labels as a `.npy` file.
        n_regions (int, optional): Total number of regions to create across both hemispheres. Defaults to 1024.
        random_state (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to 42.
        **kwargs: Additional keyword arguments passed to the mesh clustering function.

    Returns:
        np.ndarray: Array of region labels for all vertices in both hemispheres.

    Notes:
        - Assumes the existence of HCP mesh and structure data, as well as utility functions for mesh creation and
        clustering.
        - The medial wall is masked out to exclude non-cortical vertices from clustering.
        - The resulting labels are saved to the specified path in NumPy `.npy` format.

    """
    labels = []
    for hemi in ["left", "right"]:
        n_clusters = n_regions // 2  # per hemisphere

        # Create mesh
        mesh = hcp.mesh[f"sphere_{hemi}"]
        mesh = mesh_utils.mesh_creation.build_polydata(mesh[0], mesh[1])

        # Create mask for the medial wall
        size_hemi = hcp.struct[f"cortex_{hemi}"].stop - hcp.struct[f"cortex_{hemi}"].start
        if hemi == "left":
            mask = hcp.left_cortex_data(np.ones(size_hemi), 0).astype(bool)
        else:
            mask = hcp.right_cortex_data(np.ones(size_hemi), 0).astype(bool)

        # Create the regions
        cluster_labels, _ = mesh_utils.mesh_cluster.cluster_points(
            mesh,
            mask=mask,
            n_clusters=n_clusters,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )
        cluster_labels = cluster_labels[cluster_labels != 0] - 1
        labels.append(
            (cluster_labels + n_clusters) if hemi == "right" else cluster_labels,
        )

    # Merge both hemispheres and save the parcellation
    labels = np.concatenate(labels)
    np.save(save_path, labels)
    return labels


# %%

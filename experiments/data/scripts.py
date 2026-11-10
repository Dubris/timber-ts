# %%
import hcp_utils as hcp

from experiments.data.utils import compute_and_save_spectral_embeddings, create_random_parcellation

# %%
n_components = 64
compute_and_save_spectral_embeddings(
    adjacency=hcp.cortical_adjacency,
    output_path=f"spectral_embeddings_{n_components}.npy",
    n_components=n_components,
    n_neighbors=10,
)

# %%
n_regions = 1024
create_random_parcellation(
    save_path=f"random_parcellation_{n_regions}.npy",
    n_regions=n_regions,
    random_state=42,
)

# %%

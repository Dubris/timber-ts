Assets of the fMRI application (cortical surface fsLR 32k, 59412 vertices without the medial wall).
In the `timber` package vocabulary, cortical vertices are *channels* and subjects are *entities*.

Files (generated with the functions of `utils.py`, see `scripts.py`):
- `random_parcellation_1024.npy`: cortex atlas with 1,024 regions (`create_random_parcellation`).
- `random_parcellation_4096.npy`: cortex atlas with 4,096 regions (`create_random_parcellation`).
- `spectral_embeddings_64.npy`: spectral embedding of the cortical mesh (D=64), used to initialise the
  channel embedding (`compute_and_save_spectral_embeddings`; `experiments.timber_fmri.spectral_embeddings_path`).
- `spectral_embeddings_128.npy`: same, D=128.
- `HCP-MMP1_UniqueRegionList.csv`: description of the HCP-MMP1 atlas (interpretability).

Scripts: `utils.py` (atlas and spectral embedding computation, needs `brainspace` and `hcp-utils`),
`scripts.py` (generates the files above).

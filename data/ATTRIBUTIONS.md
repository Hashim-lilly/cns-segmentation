# Dataset Attributions

Required/recommended attribution text for each external dataset materialized under this
directory by `src/cns_segmentation/data/adapters/`, registered in `dataset_registry.py`.

## SPIDER (`spider_canal`)

License: **CC-BY-4.0** (permits commercial use with attribution).

> van der Graaf, J.W., van Hooff, M.L., Buckens, C.F.M., et al. (2024). *SPIDER: A comprehensive
> dataset for spinal disorders diagnosis and severity assessment*. Zenodo.
> DOI: [10.5281/zenodo.10159290](https://doi.org/10.5281/zenodo.10159290)

Materialized here: only the 210 true-T2 series' spinal-canal masks (of 447 total series across
218 patients) — vertebra/IVD masks and the T1w/T2-SPACE series are not used by this project.

## Al-Kafri / Sudirman lumbar dataset (`alkafri_mendeley_thecal_sac`)

License: **CC BY 4.0** (permits commercial use with attribution).

> Al-Kafri, A.S., Sudirman, S., Hussain, A., et al. (2019). *Boundary Delineation of MRI Images
> for Lumbar Spinal Stenosis Detection Through Semantic Segmentation Using Deep Neural Networks*.
> IEEE Access, 7, 43487–43501. DOI: [10.1109/ACCESS.2019.2908002](https://doi.org/10.1109/ACCESS.2019.2908002)
>
> Ground-truth labels dataset: Sudirman, S., Al Kafri, A., Natalia, F., et al. (2019). *Label
> Image Ground Truth Data for Lumbar Spine MRI Dataset*. Mendeley Data, V2.
> DOI: [10.17632/zbf6b4pttk.2](https://doi.org/10.17632/zbf6b4pttk.2)

Materialized here: 1545 single-slice pseudo-subjects (515 patients × 3 lumbar disk levels
each), each a binary thecal-sac mask thresholded from the ground-truth label PNGs (pixel value
150). The raw-DICOM record (DOI 10.17632/k57fr854j2.2) was not used — see
`src/cns_segmentation/data/adapters/alkafri_mendeley.py`'s module docstring for why.

## OpenNeuro ds004507 (`openneuro_ds004507`)

License: **CC0** (public domain dedication — no attribution legally required, credited here as
good practice).

> OpenNeuro dataset ds004507, *"Spinal Cord Head Positions"*.
> [https://openneuro.org/datasets/ds004507](https://openneuro.org/datasets/ds004507)

Materialized here: a 7-subject subset's `rootlets_dseg` derivative (one canonical session per
subject, preferring `ses-headNormal`). This is a PMJ-angle/CSA head-position study; rootlet
labels are a secondary derivative, not the dataset's primary purpose — do not describe it
externally as "a rootlet dataset."

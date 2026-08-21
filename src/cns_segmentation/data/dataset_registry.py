"""Registry of ground-truth datasets available for training and pipeline wiring.

Mirrors the `ModelSpec`/`MODELS` pattern in
`cns_cfd.segmentation_bridge.model_registry` on the dataset side: each entry
tracks where a structure's ground-truth labels live on disk and under what
BIDS derivative suffix, so both `cns_segmentation`'s trainer and the sibling
`cns-cfd-simulation` repo's pipeline can resolve label paths from one place.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Repo root: .../cns-segmentation/src/cns_segmentation/data/dataset_registry.py -> .../cns-segmentation
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SPINE_GENERIC_ROOT = _REPO_ROOT / "data" / "spine-generic"
# Sass et al. 2017 has no imaging data of its own — it's a single idealized
# CAD geometry, represented in this codebase purely as hardcoded numeric
# constants (volumes, CSAs, hydrodynamic parameters). `root` below points at
# that constants module (in the sibling cns-cfd-simulation repo) rather than
# a BIDS directory, documenting where the real values live instead of
# implying a dataset tree that doesn't exist.
_SASS_2017_SOURCE = (
    _REPO_ROOT.parent
    / "cns-cfd-simulation"
    / "src"
    / "cns_cfd"
    / "domain_prep"
    / "geometry_validation.py"
)
_SPIDER_ROOT = _REPO_ROOT / "data" / "spider"
_ALKAFRI_MENDELEY_ROOT = _REPO_ROOT / "data" / "alkafri_mendeley"
_OPENNEURO_DS004507_ROOT = _REPO_ROOT / "data" / "openneuro_ds004507"


@dataclass
class DatasetSpec:
    """Specification for a ground-truth dataset entry.

    Args:
        name: Registry key.
        root: BIDS dataset root directory.
        format: Data layout convention (e.g. "bids-derivatives").
        label_keys: Maps structure name (e.g. "cord") to its BIDS derivative
            suffix (e.g. "SC_seg"), used to build
            `<root>/derivatives/<derivatives_subdir>/<subject>/anat/<subject>_<contrast>_label-<suffix>.nii.gz`.
        spinal_region: Anatomical coverage, e.g. "cervical-thoracic".
        subject_count: Subject count actually reachable via `create_datalist()`
            with this spec's `label_keys` — not the raw on-disk file count.
        sites: Number of distinct acquisition sites reachable.
        license: Dataset license string.
        role: Intended use, e.g. "train" or "eval".
        adapter: Optional callable to convert this dataset's label format into
            the convention consumed by downstream code. None for datasets
            already in BIDS-derivatives form.
        derivatives_subdir: BIDS derivatives subdirectory name under `root`.
            Defaults to "labels" (hard binary masks). spine-generic also
            ships "labels_softseg" (probabilistic soft-segmentation masks).
    """

    name: str
    root: Path
    format: str
    label_keys: dict[str, str]
    spinal_region: str
    subject_count: int
    sites: int
    license: str
    role: str = "train"
    adapter: Optional[Callable] = None
    derivatives_subdir: str = "labels"


DATASETS: dict[str, DatasetSpec] = {
    "spine_generic_cord": DatasetSpec(
        name="spine_generic_cord",
        root=_SPINE_GENERIC_ROOT,
        format="bids-derivatives",
        label_keys={"cord": "SC_seg"},
        spinal_region="cervical-thoracic",
        subject_count=254,
        sites=42,
        license="CC0",
        role="train",
    ),
    "spine_generic_canal": DatasetSpec(
        name="spine_generic_canal",
        root=_SPINE_GENERIC_ROOT,
        format="bids-derivatives",
        label_keys={"canal": "canal_seg"},
        spinal_region="cervical-thoracic",
        subject_count=254,
        sites=42,
        license="CC0",
        role="train",
    ),
    "spine_generic_csf": DatasetSpec(
        name="spine_generic_csf",
        root=_SPINE_GENERIC_ROOT,
        format="bids-derivatives",
        label_keys={"csf": "CSF_seg"},
        spinal_region="cervical-thoracic",
        subject_count=10,
        sites=7,
        license="CC0",
        role="train",
    ),
    "spine_generic_rootlets": DatasetSpec(
        name="spine_generic_rootlets",
        root=_SPINE_GENERIC_ROOT,
        format="bids-derivatives",
        label_keys={"rootlets": "rootlets_dseg"},
        spinal_region="cervical",
        # 21 subjects ship a plain rootlets_dseg label directly; 3 more
        # (sub-amu02, sub-barcelona01, sub-brnoUhb03) only ship rater-variant
        # (desc-staple/desc-rater1..4) labels. Run
        # `rootlet_rater_selection.resolve_rootlet_rater_labels(spec.root)`
        # once to materialize the canonical file each of those 3 needs
        # before this count is reachable via `create_datalist()`.
        subject_count=24,
        sites=10,
        license="CC0",
        role="train",
    ),
    "spine_generic_softseg_cord": DatasetSpec(
        name="spine_generic_softseg_cord",
        root=_SPINE_GENERIC_ROOT,
        format="bids-derivatives",
        label_keys={"cord_soft": "SC_softseg"},
        spinal_region="cervical-thoracic",
        subject_count=230,
        sites=39,
        license="CC0",
        role="comparison_only",
        derivatives_subdir="labels_softseg",
    ),
    "sass_2017_reference": DatasetSpec(
        name="sass_2017_reference",
        # Not a BIDS directory — see `_SASS_2017_SOURCE` above. Nothing reads
        # this spec's `root`/`label_keys` for real I/O; `create_datalist()`
        # and `label_path()` are never called against this entry, and
        # `TestRealDataCounts` skips role="comparison_only" specs for the
        # same reason. Kept for registry-completeness and cross-repo
        # discoverability (Fluids Barriers CNS 14:36, foramen magnum to S5,
        # healthy 23-year-old female).
        root=_SASS_2017_SOURCE,
        format="reference-constants",
        label_keys={},
        spinal_region="full-spine",
        subject_count=1,
        sites=1,
        license="CC-BY-SA 4.0",
        role="comparison_only",
    ),
    "spider_canal": DatasetSpec(
        name="spider_canal",
        # Materialized by `data.adapters.spider.prepare()` from Zenodo
        # 10.5281/zenodo.10159290 (218 patients / 447 series; this entry
        # keeps only the 210 true-T2 series). Site caveat: the paper (Table
        # 1) documents 4 source hospitals, but the released files carry no
        # per-subject hospital identifier, so every subject here resolves to
        # a single site string ("spider") via `get_site_from_subject()`.
        # `sites` below is 1 (what `create_datalist()` actually reaches, per
        # this field's own docstring), not the paper's 4 — do not use SPIDER
        # subjects as a training site under a leave-one-site-out scheme.
        # role is "validation", not "train", for that same reason.
        root=_SPIDER_ROOT,
        format="bids-derivatives",
        label_keys={"canal": "canal_seg"},
        spinal_region="lumbar",
        subject_count=210,
        sites=1,
        license="CC-BY-4.0",
        role="validation",
    ),
    "alkafri_mendeley_thecal_sac": DatasetSpec(
        name="alkafri_mendeley_thecal_sac",
        # Materialized by `data.adapters.alkafri_mendeley.prepare()` from
        # Mendeley DOI 10.17632/zbf6b4pttk.2 (515 patients, last 3 lumbar
        # IVD levels only — not full lumbar coverage). Each of the 1545
        # subjects here is a single 2D axial slice promoted to a
        # 1-slice-thick pseudo-volume, not a full 3D study; `sites=1` since
        # the source metadata carries no per-patient site identifier. role
        # is "validation" pending a human review of the adapter's slice-
        # correspondence logic before any promotion to "train".
        root=_ALKAFRI_MENDELEY_ROOT,
        format="bids-derivatives",
        label_keys={"thecal_sac": "thecal_sac_seg"},
        spinal_region="lumbar",
        subject_count=1545,
        sites=1,
        license="CC BY 4.0",
        role="validation",
    ),
    "openneuro_ds004507": DatasetSpec(
        name="openneuro_ds004507",
        # Materialized by `data.adapters.openneuro_ds004507.prepare()` — a
        # 7-subject rootlets-validation subset of OpenNeuro ds004507
        # ("Spinal Cord Head Positions", CC0), one canonical session per
        # subject (prefer ses-headNormal, fall back to headUp/headDown).
        # Same `rootlets_dseg` label scheme as spine_generic_rootlets, but
        # kept as its own site (role="validation") rather than pooled into
        # that entry, to preserve site-holdout integrity.
        root=_OPENNEURO_DS004507_ROOT,
        format="bids-derivatives",
        label_keys={"rootlets": "rootlets_dseg"},
        spinal_region="cervical-thoracic-junction",
        subject_count=7,
        sites=1,
        license="CC0",
        role="validation",
    ),
}


def label_path(
    spec: DatasetSpec, subject_id: str, structure: str, contrast: str = "T2w"
) -> Path:
    """Build the expected label file path for a subject/structure in this spec.

    Args:
        spec: Dataset spec containing `label_keys`.
        subject_id: BIDS subject directory name, e.g. "sub-amu01".
        structure: Structure name, must be a key in `spec.label_keys`.
        contrast: Image contrast used in the label filename. Defaults to "T2w".

    Returns:
        Expected path to the label NIfTI (may not exist on disk).

    Raises:
        KeyError: If `structure` is not in `spec.label_keys`.
    """
    if structure not in spec.label_keys:
        available = ", ".join(spec.label_keys.keys())
        raise KeyError(
            f"Structure '{structure}' not in dataset '{spec.name}'. "
            f"Available: {available}"
        )
    suffix = spec.label_keys[structure]
    return (
        spec.root
        / "derivatives"
        / spec.derivatives_subdir
        / subject_id
        / "anat"
        / f"{subject_id}_{contrast}_label-{suffix}.nii.gz"
    )


def merge_label_keys(*specs: DatasetSpec) -> dict[str, str]:
    """Merge label_keys from multiple dataset specs into one lookup dict.

    Args:
        *specs: One or more DatasetSpec instances to merge.

    Returns:
        Combined structure -> BIDS suffix mapping. Later specs' keys win on
        a structure-name collision (in practice spine-generic's cord/canal/
        csf/rootlets specs each define exactly one distinct structure, so
        collisions are not expected).

    Raises:
        ValueError: If no specs are given.
    """
    if not specs:
        raise ValueError("merge_label_keys requires at least one DatasetSpec")
    merged: dict[str, str] = {}
    for spec in specs:
        merged.update(spec.label_keys)
    return merged


def get_dataset(name: str) -> DatasetSpec:
    """Retrieve a dataset specification by name.

    Args:
        name: Registry key for the dataset.

    Returns:
        DatasetSpec for the requested dataset.

    Raises:
        KeyError: If the dataset name is not in the registry.
    """
    if name not in DATASETS:
        available = ", ".join(DATASETS.keys())
        raise KeyError(f"Dataset '{name}' not found. Available: {available}")
    return DATASETS[name]


def list_datasets() -> list[str]:
    """List all registered dataset names."""
    return list(DATASETS.keys())

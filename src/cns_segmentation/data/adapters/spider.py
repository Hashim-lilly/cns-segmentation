"""Adapter for the SPIDER lumbar-spine dataset (Zenodo 10.5281/zenodo.10159290).

SPIDER ships paired raw MRI series and multi-label vertebra/disc/canal
segmentations as one `.mha` file per series (447 total across `images.zip`
and `masks.zip`), named `<patient>_<sequence>.mha` (e.g. `1_t1.mha`,
`1_t2.mha`, `107_t2_SPACE.mha`). Confirmed against the released
`overview.csv` (447 rows) and both zip listings: `new_file_name` matches the
`.mha` stem exactly in both `masks/` and `images/`.

Scope decision (already made — see task description, not relitigated here):
this adapter keeps ONLY true T2 series, excluding both T1 and the T2-SPACE
variant. That is decidable from the filename suffix alone with no need to
open every file or even download `overview.csv`: splitting the stem on the
last "_" gives "t2"/"t1"/"SPACE" respectively, so a stem's last underscore-
delimited token equalling "t2" (case-insensitive) is both necessary and
sufficient. Confirmed by direct count against the real files: 210 stems end
in "_t2", 196 in "_t1", 41 in "_t2_SPACE" (447 total) — matches "roughly
half" from the task description.

The per-series mask is a multi-label int16 `.mha` volume: 0=background,
1..N=vertebrae (not true anatomical numbering — ignored here), 100=spinal
canal (confirmed constant across every inspected series), 201+=intervertebral
discs (ignored here). Only the canal label is kept, thresholded to a binary
uint8 mask before conversion to NIfTI.

Site-holdout caveat (mirrors the `sass_2017_reference` caveat already
documented in `dataset_registry.py`): the paper (Table 1) reports SPIDER was
collected across 4 hospitals, and `sites=4` below records that as ground
truth. But the *released* files carry no per-subject hospital identifier —
only a sequential patient index — so every subject materialized by this
adapter gets the single tag prefix "spider" via `subject_dirname`, and
`spine_generic.get_site_from_subject()` will therefore only ever resolve 1
site ("spider") for these subjects, not 4. Any site-holdout split that
assumes `sites` reachable subjects-with-distinct-site-strings will silently
see 1, not 4, for this dataset. Do not use SPIDER subjects as a training
site under a leave-one-site-out scheme for that reason (hence `role`
recommendation of "validation", not "train").
"""

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests
import SimpleITK as sitk

from cns_segmentation.data.adapters.base import (
    is_prepared,
    subject_dirname,
    write_bids_subject,
)
from cns_segmentation.data.dataset_registry import DatasetSpec

logger = logging.getLogger(__name__)

# .../adapters/spider.py -> parents[0]=adapters, [1]=data, [2]=cns_segmentation,
# [3]=src, [4]=cns-segmentation (repo root)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TARGET_ROOT = _REPO_ROOT / "data" / "spider"

_ZENODO_RECORD_ID = "10159290"
_ZENODO_BASE = f"https://zenodo.org/api/records/{_ZENODO_RECORD_ID}/files"
MASKS_URL = f"{_ZENODO_BASE}/masks.zip/content"
IMAGES_URL = f"{_ZENODO_BASE}/images.zip/content"

# Ground truth from the SPIDER paper's Table 1 — see the site-holdout caveat
# in the module docstring above: this adapter's own subjects only ever
# resolve to 1 site string ("spider"), not these 4 hospitals.
SITES_PER_PAPER = 4

# True-T2 series count confirmed by direct inspection of the live Zenodo
# release's masks.zip listing (210 of 447 total .mha files; the rest are
# 196 T1 + 41 T2-SPACE series, excluded by scope decision). Used only as the
# `is_prepared()` short-circuit target for `prepare(force=False)` — the
# `subject_count` this module actually reports always comes from what is
# counted on disk, not from this constant.
EXPECTED_SUBJECT_COUNT = 210

LICENSE = "CC-BY-4.0"
SPINAL_REGION = "lumbar"


def _is_t2_series(stem: str) -> bool:
    """True if `stem` (a `.mha` filename minus extension) is a true T2 series.

    Excludes both T1 ("<patient>_t1") and T2-SPACE ("<patient>_t2_SPACE")
    series — see the scope-decision note in the module docstring.

    Args:
        stem: Filename stem, e.g. "1_t2", "1_t1", "107_t2_SPACE".

    Returns:
        True iff the last "_"-delimited token of `stem` is "t2" (case-insensitive).
    """
    return stem.rsplit("_", 1)[-1].lower() == "t2"


def select_t2_series(member_names: list[str]) -> list[str]:
    """Filter a zip member-name listing down to true-T2 series stems.

    Pure/offline: operates on filenames only, no file contents are read.

    Args:
        member_names: Zip member names, e.g. "masks/1_t2.mha" or "1_t2.mha".
            Any leading directory and the ".mha" extension are stripped.

    Returns:
        Sorted (lexicographic, for determinism) list of unique series stems
        (e.g. "1_t2") for members that are true T2 series and end in ".mha".
    """
    stems = set()
    for name in member_names:
        if not name.lower().endswith(".mha"):
            continue
        stem = Path(name).stem
        if _is_t2_series(stem):
            stems.add(stem)
    return sorted(stems)


def _mha_bytes_to_nifti_bytes(mha_bytes: bytes, canal_only: bool = False) -> bytes:
    """Convert one `.mha` volume's raw bytes into `.nii.gz` bytes.

    Pure format conversion — no network, no dataset-specific file naming.
    Uses SimpleITK for both the read and the write so voxel geometry
    (spacing/origin/direction) is preserved exactly as stored in the source
    `.mha`.

    Args:
        mha_bytes: Raw bytes of a `.mha` file (SimpleITK/GDCM-readable).
        canal_only: If True, treat the volume as SPIDER's multi-label mask
            and replace its content with a binary `(array == 100)` spinal-
            canal mask (uint8) before writing, preserving the original
            geometry via `CopyInformation`. If False, write the volume
            through unchanged (a plain format conversion).

    Returns:
        Raw bytes of an equivalent `.nii.gz` file.
    """
    with tempfile.TemporaryDirectory(prefix="spider_mha_") as tmp:
        tmp_dir = Path(tmp)
        mha_path = tmp_dir / "in.mha"
        nifti_path = tmp_dir / "out.nii.gz"
        mha_path.write_bytes(mha_bytes)

        image = sitk.ReadImage(str(mha_path))
        if canal_only:
            array = sitk.GetArrayFromImage(image)
            canal_array = (array == 100).astype("uint8")
            canal_image = sitk.GetImageFromArray(canal_array)
            canal_image.CopyInformation(image)
            image = canal_image

        sitk.WriteImage(image, str(nifti_path))
        return nifti_path.read_bytes()


def materialize_series(
    image_mha_bytes: bytes,
    mask_mha_bytes: bytes,
    target_root: Path,
    subject_id: str,
) -> None:
    """Convert one series' paired raw image + multi-label mask into a BIDS subject.

    The pure, network-free half of this adapter: given the two `.mha` files'
    raw bytes already in hand (however they were obtained), derives the
    binary spinal-canal label from the mask and writes both files out in
    spine-generic BIDS-derivatives shape via `write_bids_subject`. Feed this
    function small synthetic `.mha` bytes in tests instead of exercising
    `prepare()`'s network path.

    Args:
        image_mha_bytes: Raw bytes of the series' raw MRI `.mha` file.
        mask_mha_bytes: Raw bytes of the series' multi-label mask `.mha` file.
        target_root: Dataset root directory to materialize into.
        subject_id: BIDS subject dirname, e.g. "sub-spider0001".
    """
    image_bytes = _mha_bytes_to_nifti_bytes(image_mha_bytes, canal_only=False)
    canal_bytes = _mha_bytes_to_nifti_bytes(mask_mha_bytes, canal_only=True)
    write_bids_subject(target_root, subject_id, image_bytes, {"canal_seg": canal_bytes})


def _download_to_file(url: str, dest: Path) -> Path:
    """Stream-download `url` to `dest` in full (source has no Range support).

    Args:
        url: Zenodo file-content endpoint to download.
        dest: Destination path to write the downloaded bytes to.

    Returns:
        `dest`, for chaining.
    """
    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, timeout=1800, stream=True) as response:
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 22):
                f.write(chunk)
    return dest


def _count_materialized_subjects(root: Path) -> int:
    """Count `sub-spider*` subject dirs already materialized under `root`."""
    labels_dir = root / "derivatives" / "labels"
    if not labels_dir.is_dir():
        return 0
    return sum(1 for p in labels_dir.glob("sub-spider*") if p.is_dir())


def build_spec(subject_count: int) -> DatasetSpec:
    """Build the `DatasetSpec` this adapter reports, parameterized by subject count.

    Args:
        subject_count: Number of subjects actually materialized under
            `_TARGET_ROOT` (or expected to be, before `prepare()` runs).

    Returns:
        DatasetSpec describing the SPIDER canal dataset.
    """
    return DatasetSpec(
        name="spider_canal",
        root=_TARGET_ROOT,
        format="bids-derivatives",
        label_keys={"canal": "canal_seg"},
        spinal_region=SPINAL_REGION,
        subject_count=subject_count,
        sites=SITES_PER_PAPER,
        license=LICENSE,
        role="validation",
    )


def prepare(force: bool = False) -> DatasetSpec:
    """Download (if needed) and materialize the SPIDER T2-only canal dataset.

    Downloads `masks.zip` and `images.zip` from the Zenodo record in full
    (no Range-request support upstream), selects only true-T2 series by
    filename (see `select_t2_series`), converts each retained series' paired
    image + mask into `.nii.gz` (see `materialize_series`), and writes both
    under `_TARGET_ROOT` in spine-generic BIDS-derivatives shape. The
    downloaded zips are deleted afterward regardless of outcome — only the
    per-subject `.nii.gz` files remain on disk.

    Args:
        force: If True, re-download/re-materialize even if `is_prepared()`
            already reports `_TARGET_ROOT` as complete. If False (default),
            skip that work and just return the spec for what is already there.

    Returns:
        DatasetSpec describing the materialized dataset root. `subject_count`
        reflects how many subjects are actually present after this call.
    """
    if not force and is_prepared(_TARGET_ROOT, expected_subject_count=EXPECTED_SUBJECT_COUNT):
        existing = _count_materialized_subjects(_TARGET_ROOT)
        logger.info(
            "SPIDER already prepared at %s (%d subjects) — skipping", _TARGET_ROOT, existing
        )
        return build_spec(subject_count=existing)

    work_dir = Path(tempfile.mkdtemp(prefix="spider_download_"))
    try:
        masks_zip_path = _download_to_file(MASKS_URL, work_dir / "masks.zip")
        images_zip_path = _download_to_file(IMAGES_URL, work_dir / "images.zip")

        with (
            zipfile.ZipFile(masks_zip_path) as masks_zf,
            zipfile.ZipFile(images_zip_path) as images_zf,
        ):
            mask_names = masks_zf.namelist()
            t2_stems = select_t2_series(mask_names)
            logger.info(
                "Selected %d true-T2 series out of %d mask files", len(t2_stems), len(mask_names)
            )

            mask_dirname = next((n.split("/", 1)[0] for n in mask_names if "/" in n), None)
            image_names = images_zf.namelist()
            image_dirname = next((n.split("/", 1)[0] for n in image_names if "/" in n), None)

            _TARGET_ROOT.mkdir(parents=True, exist_ok=True)
            materialized = 0
            for i, stem in enumerate(t2_stems, start=1):
                mask_member = f"{mask_dirname}/{stem}.mha" if mask_dirname else f"{stem}.mha"
                image_member = f"{image_dirname}/{stem}.mha" if image_dirname else f"{stem}.mha"

                mask_bytes = masks_zf.read(mask_member)
                image_bytes = images_zf.read(image_member)

                subject_id = subject_dirname("spider", i)
                materialize_series(image_bytes, mask_bytes, _TARGET_ROOT, subject_id)
                materialized += 1
                if materialized % 25 == 0:
                    logger.info(
                        "Materialized %d/%d SPIDER T2 subjects", materialized, len(t2_stems)
                    )

        logger.info(
            "SPIDER materialization complete: %d subjects at %s", materialized, _TARGET_ROOT
        )
        return build_spec(subject_count=materialized)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

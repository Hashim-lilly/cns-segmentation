"""Adapter for the Al-Kafri/Mendeley lumbar-spine thecal-sac dataset.

Three linked Mendeley records are involved:

- Raw MRI images (DICOM/IMA), DOI 10.17632/k57fr854j2.2 ("Lumbar Spine MRI
  Dataset", 515 patients).
- Ground-truth labels, DOI 10.17632/zbf6b4pttk.2 ("Label Image Ground Truth
  Data for Lumbar Spine MRI Dataset") — this is the only one this adapter
  downloads. See below for why.
- Source code, DOI 10.17632/8cp2cp7km8.2 (GPL-3.0; read/run only, never
  redistributed) — `09_Source_Code/03_Extract_Best_Slices/{Slices
  Numbers.csv, T1_Subfolders.csv, T2_Subfolders.csv, Extract_T1_Slices.m,
  Extract_T2_Slices.m, extract_dicom.m}` document exactly how each of the
  515 patients' three labeled axial slices (last 3 IVD levels, i.e. disks
  "D3"/"D4"/"D5") were picked out of that patient's raw DICOM study:
  `Slices Numbers.csv` gives, per patient folder number, the 1-based index
  (into a `natsortfiles`-sorted listing of that series' `.ima` files) of
  each of the 3 slices; `T1_Subfolders.csv`/`T2_Subfolders.csv` give the
  DICOM series subfolder name to search for (e.g. "T1_TSE_TRA",
  "T2_TSE_TRA_384_0005") within each patient's folder tree.

THE HARD PART turned out to already be solved for us, verified rather than
assumed: the ground-truth zip's `04_Intermediary_Ground_Truth_Data/`
subtree ships `T1_Output/T1_<patient>_D<disk>.png` and
`T2_Output/T2_<patient>_D<disk>.png` — the exact per-slice PNGs that
`Extract_T1_Slices.m`/`Extract_T2_Slices.m` themselves wrote by running the
correspondence logic above end-to-end (index into the sorted `.ima` listing
of the matched series subfolder, `dicomread`, contrast-stretch via
`imadjust(im2, stretchlim(im2, 0), [])`, `imwrite` to
`<prefix>_<patient>_D<disk>.png`). Their filenames use the identical
`<patient>_D<disk>` key as `05_Final_Ground_Truth_Data/Label_Images/
L1_<patient>_D<disk>.png`, so the image<->label correspondence for every
labeled slice is verifiable by filename cross-reference alone, with no need
to re-derive it from `Slices Numbers.csv` or touch a single raw DICOM file.

This was checked exhaustively (not assumed) against the real downloaded
zip before writing this module: all 1545 `Label_Images/L1_*.png` files
(515 patients x 3 disks) have a matching `T2_Output/T2_*.png` (and
`T1_Output/T1_*.png`) file, and none of the paired PNGs are truncated/
zero-byte. 515/515 patients, 1545/1545 disk-level pairs verified-mapped;
0 patients skipped. See `find_verified_pairs()` below, which still applies
this check at runtime (not just an assumption baked in from the one-time
audit) and will skip — never guess — any label lacking a matching T2
raw-slice filename, e.g. if a future dataset version drops one.

Practical consequence: the ~6.27GB raw-DICOM record (k57fr854j2) is never
downloaded by this adapter. `prepare()` only fetches
`Ground_Truth_Label.zip` (~1.04GB) from zbf6b4pttk. `Source_Code.zip` is
never downloaded in full either — the handful of CSV/`.m` files above were
read directly out of the remote zip via HTTP Range requests (see the
one-time investigation notes in this module's task history) purely to
confirm the correspondence logic being relied on; nothing from that GPL-3.0
zip is reproduced or shipped here beyond this description of its logic.

Each of the 1545 verified (patient, disk) pairs becomes its own pseudo-
subject: a 320x320 T2-weighted slice (`T2_Output`) as the "image" and a
binary thecal-sac mask thresholded from the corresponding label PNG
(pixel value 150; the other confirmed label values are 50=IVD, 100=
Posterior Element, 200=AAP, 250=Other/background, all discarded here) as
the "label", each promoted from a 2D array to a 1-slice-thick 3D NIfTI
volume — see `_slice_png_bytes_to_nifti_bytes()`. This is a deliberate
scope choice per the task: labels here are 2D (only the last 3 IVD-level
axial slices per patient), not full 3D lumbar volumes, so one pseudo-
subject per slice is used rather than assembling a sparse full-volume
NIfTI. `spinal_region="lumbar"` on the returned `DatasetSpec` should be
read with that caveat: coverage is the last three intervertebral disk
levels only, not the full lumbar spine.

Site caveat (same shape as `spider.py`'s): the source metadata gives no
per-patient acquisition-site identifier, so every subject here gets the
single tag prefix "alkafri" and this adapter reports `sites=1`.
"""

import logging
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import SimpleITK as sitk

from cns_segmentation.data.adapters.base import (
    is_prepared,
    subject_dirname,
    write_bids_subject,
)
from cns_segmentation.data.dataset_registry import DatasetSpec

logger = logging.getLogger(__name__)

# .../adapters/alkafri_mendeley.py -> parents[0]=adapters, [1]=data,
# [2]=cns_segmentation, [3]=src, [4]=cns-segmentation (repo root)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_TARGET_ROOT = _REPO_ROOT / "data" / "alkafri_mendeley"

_MENDELEY_API_BASE = "https://data.mendeley.com/public-api/datasets"
_LABELS_DATASET_SLUG = "zbf6b4pttk"
_LABELS_DATASET_VERSION = "2"
_LABELS_ZIP_FILENAME = "Ground_Truth_Label.zip"

_LABEL_MEMBER_RE = re.compile(
    r"^05_Final_Ground_Truth_Data/Label_Images/L1_(?P<patient>\d{4})_D(?P<disk>\d)\.png$"
)

# Confirmed by direct inspection of the live Ground_Truth_Label.zip listing:
# 1545 label PNGs (515 patients x 3 disks), every one with a matching
# T2_Output raw-slice filename (see module docstring). Used only as the
# `is_prepared()` short-circuit target for `prepare(force=False)` — the
# `subject_count` this module actually reports always comes from what is
# counted on disk (or re-verified against the live zip), not this constant.
EXPECTED_SUBJECT_COUNT = 1545

# Confirmed label pixel encoding (task-provided, cross-checked against real
# label PNGs: unique values found were exactly {50, 100, 150, 200, 250}).
THECAL_SAC_LABEL_VALUE = 150

# Per the dataset's own description: uniform 0.6875mm in-plane pixel
# spacing across all axial slices, 4.4mm centre-to-centre distance between
# adjacent slices. Applied to every pseudo-subject's 1-slice-thick NIfTI
# purely to preserve real-world scale in the header; since each volume has
# only one slice, the z-spacing value has no effect on anything but
# metadata.
PIXEL_SPACING_MM = 0.6875
SLICE_SPACING_MM = 4.4

LICENSE = "CC BY 4.0"
SPINAL_REGION = "lumbar"
SITES = 1

# `is_prepared()`'s default `min_file_size=1000` bytes is tuned for full 3D
# spine-generic volumes (vs. ~100-byte git-annex pointer stubs — see
# CLAUDE.md rule 2). This dataset's labels are single 320x320 binary masks
# in a 1-slice-thick NIfTI, which gzip-compress far smaller: confirmed
# 249-334 bytes across all 1545 real materialized label files. 100 bytes
# stays a generous margin above that gap while still rejecting an
# empty/truncated file.
MIN_LABEL_FILE_SIZE = 100


def _label_member(patient: str, disk: str) -> str:
    """Zip member path for a (patient, disk) pair's ground-truth label PNG."""
    return f"05_Final_Ground_Truth_Data/Label_Images/L1_{patient}_D{disk}.png"


def _t2_member(patient: str, disk: str) -> str:
    """Zip member path for a (patient, disk) pair's raw T2 slice PNG."""
    return f"04_Intermediary_Ground_Truth_Data/T2_Output/T2_{patient}_D{disk}.png"


def find_verified_pairs(member_names: list[str]) -> list[tuple[str, str]]:
    """Find (patient, disk) pairs with a verified label<->raw-slice correspondence.

    Pure/offline: operates on a zip listing's member names only, no file
    contents are read. A label PNG is only kept if its corresponding
    `T2_Output` raw-slice PNG is also present in `member_names` — anything
    else is logged and skipped rather than guessed at (per task
    instructions: a wrong image/label pairing is worse than a missing
    subject).

    Args:
        member_names: Zip member-name listing, e.g. from
            `zipfile.ZipFile.namelist()`.

    Returns:
        Sorted (by integer patient id, then integer disk number) list of
        unique (patient, disk) string pairs, e.g. ("0001", "3"), that have
        both a label PNG and a matching raw T2 PNG.
    """
    names = set(member_names)
    pairs = []
    for name in names:
        match = _LABEL_MEMBER_RE.match(name)
        if not match:
            continue
        patient, disk = match.group("patient"), match.group("disk")
        if _t2_member(patient, disk) in names:
            pairs.append((patient, disk))
        else:
            logger.warning(
                "Skipping patient %s disk %s: label PNG has no matching "
                "T2_Output raw-slice PNG in the zip listing",
                patient,
                disk,
            )
    return sorted(pairs, key=lambda pd: (int(pd[0]), int(pd[1])))


def _slice_array_from_png_bytes(png_bytes: bytes) -> np.ndarray:
    """Decode raw PNG bytes into a 2D uint8 array via SimpleITK.

    Args:
        png_bytes: Raw bytes of an 8-bit grayscale PNG file.

    Returns:
        2D (H, W) uint8 array.

    Raises:
        ValueError: If the decoded image is not 2D (e.g. an unexpected
            multi-channel PNG), since this adapter's inputs are all
            confirmed single-channel 8-bit grayscale.
    """
    with tempfile.TemporaryDirectory(prefix="alkafri_png_") as tmp:
        png_path = Path(tmp) / "in.png"
        png_path.write_bytes(png_bytes)
        image = sitk.ReadImage(str(png_path))
        array = sitk.GetArrayFromImage(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale PNG, got array shape {array.shape}")
    return array.astype("uint8")


def _slice_png_bytes_to_nifti_bytes(png_bytes: bytes, keep_value: Optional[int] = None) -> bytes:
    """Convert one 2D PNG slice's raw bytes into 1-slice-thick `.nii.gz` bytes.

    Pure format conversion — no network, no dataset-specific file naming.
    Promotes the decoded (H, W) array to a (1, H, W) SimpleITK volume (a
    single-slice 3D image, per the task's scope decision to keep each
    labeled slice as its own pseudo-subject rather than assembling a
    sparse full lumbar volume) and stamps it with the dataset's documented
    real-world pixel/slice spacing.

    Args:
        png_bytes: Raw bytes of an 8-bit grayscale PNG (320x320 in
            practice, but not assumed here).
        keep_value: If given, the output is a binary mask of
            `(array == keep_value)` (uint8, 0/1) instead of the raw
            grayscale values — used to pull the thecal-sac class out of the
            multi-label ground-truth PNG.

    Returns:
        Raw bytes of an equivalent `.nii.gz` file.
    """
    array = _slice_array_from_png_bytes(png_bytes)
    if keep_value is not None:
        array = (array == keep_value).astype("uint8")

    volume = sitk.GetImageFromArray(array[np.newaxis, :, :])
    volume.SetSpacing((PIXEL_SPACING_MM, PIXEL_SPACING_MM, SLICE_SPACING_MM))

    with tempfile.TemporaryDirectory(prefix="alkafri_nifti_") as tmp:
        nifti_path = Path(tmp) / "out.nii.gz"
        sitk.WriteImage(volume, str(nifti_path))
        return nifti_path.read_bytes()


def materialize_patient_disk(
    t2_png_bytes: bytes,
    label_png_bytes: bytes,
    target_root: Path,
    subject_id: str,
) -> None:
    """Convert one verified (patient, disk) pair's raw bytes into a BIDS subject.

    The pure, network-free half of this adapter: given a T2 raw-slice PNG
    and its corresponding multi-label ground-truth PNG (already in hand,
    however obtained), derives the binary thecal-sac label and writes both
    files out in spine-generic BIDS-derivatives shape via
    `write_bids_subject`. Feed this function small synthetic PNG bytes in
    tests instead of exercising `prepare()`'s network path.

    Args:
        t2_png_bytes: Raw bytes of the pair's `T2_Output` raw-slice PNG.
        label_png_bytes: Raw bytes of the pair's `Label_Images` ground-truth PNG.
        target_root: Dataset root directory to materialize into.
        subject_id: BIDS subject dirname, e.g. "sub-alkafri0001".
    """
    image_bytes = _slice_png_bytes_to_nifti_bytes(t2_png_bytes, keep_value=None)
    thecal_sac_bytes = _slice_png_bytes_to_nifti_bytes(
        label_png_bytes, keep_value=THECAL_SAC_LABEL_VALUE
    )
    write_bids_subject(target_root, subject_id, image_bytes, {"thecal_sac_seg": thecal_sac_bytes})


def _resolve_download_url(slug: str, version: str, filename: str) -> str:
    """Look up a dataset file's direct download URL via the Mendeley public API.

    Args:
        slug: Mendeley dataset id, e.g. "zbf6b4pttk".
        version: Dataset version string, e.g. "2".
        filename: Exact filename to match in the dataset's root file listing.

    Returns:
        Direct HTTPS download URL for the matching file.

    Raises:
        ValueError: If no file named `filename` is found in the listing.
    """
    url = f"{_MENDELEY_API_BASE}/{slug}/files?version={version}&folder_id=root"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    for entry in response.json():
        if entry.get("filename") == filename:
            return entry["content_details"]["download_url"]
    raise ValueError(f"No file named {filename!r} found in Mendeley dataset {slug} v{version}")


def _download_to_file(url: str, dest: Path) -> Path:
    """Stream-download `url` to `dest` in full.

    Args:
        url: Direct file-content URL to download.
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
    """Count `sub-alkafri*` subject dirs already materialized under `root`."""
    labels_dir = root / "derivatives" / "labels"
    if not labels_dir.is_dir():
        return 0
    return sum(1 for p in labels_dir.glob("sub-alkafri*") if p.is_dir())


def build_spec(subject_count: int) -> DatasetSpec:
    """Build the `DatasetSpec` this adapter reports, parameterized by subject count.

    Args:
        subject_count: Number of pseudo-subjects actually materialized under
            `_TARGET_ROOT` (or expected to be, before `prepare()` runs).

    Returns:
        DatasetSpec describing the Al-Kafri thecal-sac dataset.
    """
    return DatasetSpec(
        name="alkafri_mendeley_thecal_sac",
        root=_TARGET_ROOT,
        format="bids-derivatives",
        label_keys={"thecal_sac": "thecal_sac_seg"},
        spinal_region=SPINAL_REGION,
        subject_count=subject_count,
        sites=SITES,
        license=LICENSE,
        role="validation",
    )


def prepare(force: bool = False) -> DatasetSpec:
    """Download (if needed) and materialize the Al-Kafri thecal-sac dataset.

    Downloads only `Ground_Truth_Label.zip` (~1.04GB, from the zbf6b4pttk
    Mendeley record) — never the ~6.27GB raw-DICOM record — because that
    zip already ships verified-correct raw-slice PNGs alongside the label
    PNGs (see module docstring). Finds every (patient, disk) pair with a
    verified correspondence (`find_verified_pairs`), converts each into a
    2D-as-3D `.nii.gz` image + binary thecal-sac label pair
    (`materialize_patient_disk`), and writes them under `_TARGET_ROOT` in
    spine-generic BIDS-derivatives shape, one pseudo-subject per pair. The
    downloaded zip is deleted afterward regardless of outcome — only the
    per-subject `.nii.gz` files remain on disk.

    Args:
        force: If True, re-download/re-materialize even if `is_prepared()`
            already reports `_TARGET_ROOT` as complete. If False (default),
            skip that work and just return the spec for what is already there.

    Returns:
        DatasetSpec describing the materialized dataset root. `subject_count`
        reflects how many pseudo-subjects are actually present after this call.
    """
    if not force and is_prepared(
        _TARGET_ROOT,
        expected_subject_count=EXPECTED_SUBJECT_COUNT,
        min_file_size=MIN_LABEL_FILE_SIZE,
    ):
        existing = _count_materialized_subjects(_TARGET_ROOT)
        logger.info(
            "Al-Kafri already prepared at %s (%d subjects) — skipping", _TARGET_ROOT, existing
        )
        return build_spec(subject_count=existing)

    work_dir = Path(tempfile.mkdtemp(prefix="alkafri_download_"))
    try:
        zip_url = _resolve_download_url(
            _LABELS_DATASET_SLUG, _LABELS_DATASET_VERSION, _LABELS_ZIP_FILENAME
        )
        zip_path = _download_to_file(zip_url, work_dir / _LABELS_ZIP_FILENAME)

        with zipfile.ZipFile(zip_path) as zf:
            member_names = zf.namelist()
            pairs = find_verified_pairs(member_names)

            all_labeled_patients = {
                m.group("patient")
                for m in (_LABEL_MEMBER_RE.match(n) for n in member_names)
                if m
            }
            mapped_patients = {patient for patient, _ in pairs}
            skipped_patients = sorted(all_labeled_patients - mapped_patients, key=int)
            logger.info(
                "Al-Kafri correspondence: %d/%d patients verified-mapped "
                "(%d disk-level pairs); %d patients skipped: %s",
                len(mapped_patients),
                len(all_labeled_patients),
                len(pairs),
                len(skipped_patients),
                skipped_patients,
            )

            _TARGET_ROOT.mkdir(parents=True, exist_ok=True)
            materialized = 0
            for i, (patient, disk) in enumerate(pairs, start=1):
                label_bytes = zf.read(_label_member(patient, disk))
                t2_bytes = zf.read(_t2_member(patient, disk))

                subject_id = subject_dirname("alkafri", i)
                materialize_patient_disk(t2_bytes, label_bytes, _TARGET_ROOT, subject_id)
                materialized += 1
                if materialized % 200 == 0:
                    logger.info(
                        "Materialized %d/%d Al-Kafri subjects", materialized, len(pairs)
                    )

        logger.info(
            "Al-Kafri materialization complete: %d subjects at %s", materialized, _TARGET_ROOT
        )
        return build_spec(subject_count=materialized)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

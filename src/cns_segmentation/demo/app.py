"""Interactive Streamlit demo: upload/select → segment → visualize → export → score.

Pure wiring over existing modules — no new segmentation, mesh, or metrics
logic lives here:
  - `cns_segmentation.inference` for model loading + single-volume inference
    (`predict_volume`) and held-out subject listing (`list_heldout_subjects`).
  - `cns_segmentation.models.uncertainty.MCDropoutWrapper` (via
    `predict_volume(uncertainty=True)`) for entropy/variance/mutual-information.
  - `cns_segmentation.mesh.export` for the marching-cubes preview
    (`extract_surface`, no repair) and the full CFD-ready STL export
    (`export_cfd_mesh`).
  - `cns_segmentation.evaluation.metrics.evaluate_subject` and
    `cns_segmentation.evaluation.calibration` for per-case Dice/HD95/NSD and
    ECE, only when a ground-truth label is available for the selected case.

Every feature's core logic is a plain function taking/returning paths, arrays,
and dicts — `main()` at the bottom is the only place that touches `st.*`
widgets. This lets each feature be exercised directly in a script (real
subject, real checkpoint, real artifacts on disk) without a browser, which is
how this module was verified (no browser-automation tooling is available in
this environment).

Run: streamlit run src/cns_segmentation/demo/app.py
"""

import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import plotly.graph_objects as go

from cns_segmentation.data.dataset_registry import get_dataset, label_path
from cns_segmentation.evaluation.calibration import expected_calibration_error, plot_reliability_diagram
from cns_segmentation.evaluation.metrics import evaluate_subject
from cns_segmentation.inference import (
    build_inferer,
    checkpoint_class_id,
    list_heldout_subjects,
    load_model,
    load_yaml,
    predict_volume,
)
from cns_segmentation.mesh.export import MeshExportConfig, export_cfd_mesh, extract_surface
from cns_segmentation.models.segresnet import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Same checkpoint/config manifest as `scripts/evaluate.py`'s STRUCTURES,
# minus the eval-only reuse/external fields this demo doesn't need. Kept as
# a small local copy rather than importing `scripts/evaluate.py` — `scripts/`
# has no `__init__.py`, so it isn't an importable package.
STRUCTURES = {
    "cord": {
        "checkpoint": "experiments/spine_segresnet_phase1_20260811_065325/checkpoints/best_model.pth",
        "train_config": "configs/train_spine.yaml",
        "inference_config": "configs/inference.yaml",
        "dataset": "spine_generic_cord",
    },
    "canal": {
        "checkpoint": "experiments/spine_segresnet_canal_20260819_053543/checkpoints/best_model.pth",
        "train_config": "configs/train_spine_canal.yaml",
        "inference_config": "configs/inference_canal.yaml",
        "dataset": "spine_generic_canal",
    },
    "csf": {
        "checkpoint": "experiments/spine_segresnet_csf_20260819_052857/checkpoints/best_model.pth",
        "train_config": "configs/train_spine_csf.yaml",
        "inference_config": "configs/inference_csf.yaml",
        "dataset": "spine_generic_csf",
    },
    "rootlets": {
        "checkpoint": "experiments/spine_segresnet_rootlets_20260819_053526/checkpoints/best_model.pth",
        "train_config": "configs/train_spine_rootlets.yaml",
        "inference_config": "configs/inference_rootlets.yaml",
        "dataset": "spine_generic_rootlets",
    },
}


# --------------------------------------------------------------------------
# Feature 1: upload/select
# --------------------------------------------------------------------------


def list_available_subjects(structure: str, project_root: Path = _PROJECT_ROOT) -> list[dict]:
    """List known held-out subjects for `structure`, each with a ground-truth label on disk.

    Backs the "select a known case" half of feature 1 (the other half is a
    plain `st.file_uploader`, which needs no wrapper function).

    Returns:
        List of dicts with "subject", "site", "image", "label" (raw paths).
    """
    manifest = STRUCTURES[structure]
    train_config = load_yaml(project_root / manifest["train_config"])
    return list_heldout_subjects(train_config, structure, project_root)


# --------------------------------------------------------------------------
# Feature 2: segment-with-progress
# --------------------------------------------------------------------------


def load_structure_model(structure: str, project_root: Path = _PROJECT_ROOT) -> dict:
    """Load the model/inferer/configs for `structure`. Cache this at the call site (e.g. `st.cache_resource`).

    Returns:
        Dict with "model", "inferer", "device", "train_config", "inference_config", "structures".
    """
    manifest = STRUCTURES[structure]
    train_config = load_yaml(project_root / manifest["train_config"])
    inference_config = load_yaml(project_root / manifest["inference_config"])
    checkpoint_path = project_root / manifest["checkpoint"]
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found for '{structure}': {checkpoint_path}")

    device = get_device()
    model = load_model(inference_config["model"], checkpoint_path, device)
    inferer = build_inferer(inference_config["inference"]["sliding_window"])
    from cns_segmentation.inference import resolve_structures

    structures, _ = resolve_structures(train_config["data"].get("dataset", "spine_generic_cord"))

    return {
        "model": model,
        "inferer": inferer,
        "device": device,
        "train_config": train_config,
        "inference_config": inference_config,
        "structures": structures,
    }


def segment_case(
    image_path: Path,
    structure: str,
    project_root: Path = _PROJECT_ROOT,
    uncertainty: bool = True,
    n_mc_samples: int = 8,
    loaded: Optional[dict] = None,
) -> dict:
    """Run inference on one volume for `structure` and extract its binary mask.

    Args:
        image_path: Raw input NIfTI path.
        structure: Key into `STRUCTURES`.
        project_root: Repo root.
        uncertainty: Also run MC-Dropout (feature 4's data source).
        n_mc_samples: MC-Dropout forward passes.
        loaded: Pre-loaded `load_structure_model()` output, to avoid
            reloading the checkpoint across repeated calls (e.g. from a
            Streamlit session that already cached it).

    Returns:
        `predict_volume()`'s result dict, plus "target_class_id" (this
        structure's class id within the checkpoint's multi-class output,
        or 1 for a legacy binary checkpoint) and "structure".
    """
    ctx = loaded if loaded is not None else load_structure_model(structure, project_root)
    train_config = ctx["train_config"]
    spacing = ctx["inference_config"]["preprocessing"]["spacing"]

    result = predict_volume(
        image_path=image_path,
        model=ctx["model"],
        inferer=ctx["inferer"],
        spacing=spacing,
        device=ctx["device"],
        structures=ctx["structures"],
        uncertainty=uncertainty,
        n_mc_samples=n_mc_samples,
    )
    result["target_class_id"] = (
        checkpoint_class_id(train_config, structure) if ctx["structures"] is not None else 1
    )
    result["structure"] = structure
    result["spacing"] = spacing
    return result


# --------------------------------------------------------------------------
# Feature 3 + 4: 3D slice render + uncertainty overlay
# --------------------------------------------------------------------------

_AXIS_INDEX = {"sagittal": 0, "coronal": 1, "axial": 2}


def slice_2d(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    """Extract one 2D slice from a 3D volume along `axis` at `index`, transposed for display."""
    idx = _AXIS_INDEX[axis]
    if idx == 0:
        return volume[index, :, :].T
    if idx == 1:
        return volume[:, index, :].T
    return volume[:, :, index].T


def make_slice_figure(
    image: np.ndarray,
    axis: str,
    index: int,
    pred_binary: Optional[np.ndarray] = None,
    uncertainty_map: Optional[np.ndarray] = None,
    title: str = "",
) -> go.Figure:
    """Build a 2D slice figure: grayscale MRI + optional prediction contour + optional uncertainty heatmap.

    Backs feature 3 (slice render) alone when `uncertainty_map` is None, and
    feature 4 (uncertainty overlay) when it's supplied.
    """
    img_slice = slice_2d(image, axis, index)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=img_slice, colorscale="gray", showscale=False, name="MRI"))

    if pred_binary is not None:
        pred_slice = slice_2d(pred_binary.astype(np.float32), axis, index)
        fig.add_trace(
            go.Contour(
                z=pred_slice,
                showscale=False,
                contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                line=dict(color="red", width=2),
                name="Prediction",
            )
        )
    if uncertainty_map is not None:
        unc_slice = slice_2d(uncertainty_map, axis, index)
        fig.add_trace(
            go.Heatmap(
                z=unc_slice,
                colorscale="Hot",
                opacity=0.5,
                name="Uncertainty",
                colorbar=dict(title="uncertainty"),
            )
        )

    fig.update_layout(
        title=title or f"{axis} slice {index}",
        yaxis=dict(scaleanchor="x", autorange="reversed"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# --------------------------------------------------------------------------
# Feature 5: mesh preview
# --------------------------------------------------------------------------


def make_mesh_preview_figure(mask_path: Path, label_value: int = 1) -> Optional[go.Figure]:
    """Fast marching-cubes-only mesh preview (no repair) as a plotly Mesh3d figure.

    Explicitly a *preview*, not the CFD-ready pipeline — no hole-filling,
    normal-fixing, or smoothing is applied, and callers/UI must not imply
    this mesh passed `MeshQuality.passes_cfd_check` (only `export_case`'s
    full `export_cfd_mesh()` pipeline determines that).

    Returns:
        A plotly Figure, or None if the mask is empty (see `extract_surface`).
    """
    mesh = extract_surface(mask_path, label_value=label_value)
    if mesh is None:
        return None

    verts, faces = mesh.vertices, mesh.faces
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="lightpink", opacity=0.9,
            )
        ]
    )
    fig.update_layout(
        title="Mesh preview (marching cubes only — not CFD-validated)",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="data"),
    )
    return fig


# --------------------------------------------------------------------------
# Feature 6: export
# --------------------------------------------------------------------------


def export_case(seg_result: dict, subject_id: str, output_dir: Path) -> dict:
    """Write NIfTI (prediction + uncertainty maps, if present) and a CFD-pipeline STL to `output_dir`.

    Args:
        seg_result: Output of `segment_case()`.
        subject_id: Used to name output files.
        output_dir: Directory to write into (created if missing).

    Returns:
        Dict of output paths written, plus "mesh_quality" (a `MeshQuality`
        dataclass instance, or None if mesh export failed — e.g. empty mask).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    affine = seg_result["affine"]
    structure = seg_result["structure"]
    target_class_id = seg_result["target_class_id"]
    binary_mask = (seg_result["pred"] == target_class_id).astype(np.uint8)

    paths: dict = {}

    pred_path = output_dir / f"{subject_id}_{structure}_pred.nii.gz"
    nib.save(nib.Nifti1Image(binary_mask, affine), pred_path)
    paths["prediction_nifti"] = pred_path

    for key in ("entropy", "variance", "mutual_information"):
        if key in seg_result:
            unc_path = output_dir / f"{subject_id}_{structure}_{key}.nii.gz"
            nib.save(nib.Nifti1Image(seg_result[key].astype(np.float32), affine), unc_path)
            paths[f"{key}_nifti"] = unc_path

    stl_path = output_dir / f"{subject_id}_{structure}.stl"
    quality = export_cfd_mesh(pred_path, stl_path, config=MeshExportConfig(), label_value=1)
    paths["stl"] = stl_path if quality is not None else None
    paths["mesh_quality"] = quality

    return paths


# --------------------------------------------------------------------------
# Feature 7: per-case metrics
# --------------------------------------------------------------------------


def _resample_label_to_grid(label_path: Path, spacing: list) -> np.ndarray:
    """Load a raw ground-truth label and resample it onto the same RAS/spacing grid `predict_volume()` uses.

    `predict_volume()` reorients+resamples only the image; a raw label read
    straight off disk is on a different grid (native resolution/orientation)
    than the resulting prediction, so it must go through the same
    Orientationd+Spacingd steps (nearest-neighbor, since it's categorical)
    before it can be compared voxel-for-voxel against the prediction.
    """
    from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged, Orientationd, Spacingd

    transforms = Compose(
        [
            LoadImaged(keys=["label"]),
            EnsureChannelFirstd(keys=["label"]),
            Orientationd(keys=["label"], axcodes="RAS"),
            Spacingd(keys=["label"], pixdim=spacing, mode="nearest"),
        ]
    )
    data = transforms({"label": str(label_path)})
    return (data["label"][0].numpy() > 0).astype(np.uint8)


def compute_case_metrics(
    seg_result: dict,
    label_nifti_path: Path,
    output_dir: Path,
    subject_id: str,
) -> dict:
    """Score one case against ground truth: Dice/HD95/volume-error/NSD + ECE + reliability diagram.

    Requires `seg_result` to have been computed with `uncertainty=True` (for
    "mean_probs", the confidence source) — matches the confidence/correctness
    convention `scripts/uncertainty.py`'s ECE computation already uses:
    confidence = per-voxel max of MC-Dropout mean_probs, correctness =
    (argmax of mean_probs == ground truth), not the (possibly
    keep-largest-filtered) deterministic prediction.

    `label_nifti_path` is raw (native grid) — it's resampled onto
    `seg_result`'s post-`predict_volume()` grid (via `seg_result["spacing"]`)
    before scoring, since `evaluate_subject()` requires matching shapes and
    performs no resampling itself.

    Args:
        seg_result: Output of `segment_case(..., uncertainty=True)`.
        label_nifti_path: Ground-truth label NIfTI for this case/structure.
        output_dir: Where to write the reliability diagram PNG.
        subject_id: Used to name the output PNG.

    Returns:
        Dict with "metrics" (evaluate_subject() output), "ece" (calibration
        dict), "reliability_png" (path).

    Raises:
        ValueError: If `seg_result` has no "mean_probs" (uncertainty wasn't run).
    """
    if "mean_probs" not in seg_result:
        raise ValueError("compute_case_metrics requires segment_case(..., uncertainty=True)")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_class_id = seg_result["target_class_id"]
    affine = seg_result["affine"]
    binary_mask = (seg_result["pred"] == target_class_id).astype(np.uint8)

    label_np = _resample_label_to_grid(label_nifti_path, seg_result["spacing"])

    pred_path = output_dir / f"{subject_id}_metrics_pred.nii.gz"
    label_path = output_dir / f"{subject_id}_metrics_label.nii.gz"
    nib.save(nib.Nifti1Image(binary_mask, affine), pred_path)
    nib.save(nib.Nifti1Image(label_np, affine), label_path)

    metrics = evaluate_subject(pred_path, label_path)

    raw_pred_np = seg_result["mean_probs"].argmax(axis=0).astype(np.uint8)
    confidence = seg_result["mean_probs"].max(axis=0)
    correct = (raw_pred_np == target_class_id) == (label_np == 1)

    ece = expected_calibration_error(confidence.ravel(), correct.ravel(), n_bins=15)
    reliability_png = output_dir / f"{subject_id}_reliability.png"
    plot_reliability_diagram(ece, reliability_png, title=f"{subject_id} reliability")

    return {"metrics": metrics, "ece": ece, "reliability_png": reliability_png}


# --------------------------------------------------------------------------
# Streamlit UI — thin wiring over the functions above
# --------------------------------------------------------------------------


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="CNS Segmentation Demo", layout="wide")
    st.title("CNS Segmentation Demo")

    structure = st.sidebar.selectbox("Structure", list(STRUCTURES.keys()))
    source = st.sidebar.radio("Input", ["Known held-out subject", "Upload NIfTI"])

    image_path: Optional[Path] = None
    label_nifti: Optional[Path] = None
    subject_id = "uploaded"

    if source == "Known held-out subject":
        subjects = list_available_subjects(structure)
        if not subjects:
            st.sidebar.warning(f"No held-out subjects with a '{structure}' label found.")
            return
        choice = st.sidebar.selectbox(
            "Subject", subjects, format_func=lambda s: f"{s['subject']} ({s['site']})"
        )
        image_path = Path(choice["image"])
        label_nifti = Path(choice["label"])
        subject_id = choice["subject"]
    else:
        uploaded = st.sidebar.file_uploader("Upload a T2w NIfTI", type=["nii", "gz"])
        if uploaded is not None:
            tmp_path = Path("experiments/demo_uploads") / uploaded.name
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_bytes(uploaded.getvalue())
            image_path = tmp_path
            subject_id = tmp_path.stem.replace(".nii", "")

    if image_path is None:
        st.info("Select a known subject or upload a NIfTI to begin.")
        return

    run_uncertainty = st.sidebar.checkbox("Compute MC-Dropout uncertainty", value=True)

    if st.sidebar.button("Segment"):
        progress = st.progress(0, text="Loading model...")
        ctx = load_structure_model(structure)
        progress.progress(40, text="Running sliding-window inference...")
        result = segment_case(image_path, structure, uncertainty=run_uncertainty, loaded=ctx)
        progress.progress(100, text="Done.")
        st.session_state["seg_result"] = result
        st.session_state["subject_id"] = subject_id
        st.session_state["label_nifti"] = label_nifti

    if "seg_result" not in st.session_state:
        return

    result = st.session_state["seg_result"]
    binary_mask = (result["pred"] == result["target_class_id"]).astype(np.uint8)

    st.subheader("3D slice viewer")
    axis = st.selectbox("Axis", ["axial", "sagittal", "coronal"])
    max_index = result["image"].shape[_AXIS_INDEX[axis]] - 1
    index = st.slider("Slice", 0, max_index, max_index // 2)
    show_uncertainty = st.checkbox("Show uncertainty overlay", value=run_uncertainty and "entropy" in result)
    fig = make_slice_figure(
        result["image"], axis, index,
        pred_binary=binary_mask,
        uncertainty_map=result.get("entropy") if show_uncertainty else None,
        title=f"{subject_id} — {structure}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mesh preview")
    tmp_mask_path = Path("experiments/demo_uploads") / f"{subject_id}_{structure}_preview_mask.nii.gz"
    tmp_mask_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(binary_mask, result["affine"]), tmp_mask_path)
    mesh_fig = make_mesh_preview_figure(tmp_mask_path)
    if mesh_fig is not None:
        st.plotly_chart(mesh_fig, use_container_width=True)
    else:
        st.warning("Prediction mask is empty — no mesh to preview.")

    st.subheader("Export")
    if st.button("Export NIfTI + STL"):
        out_dir = Path("experiments/demo_exports") / subject_id
        paths = export_case(result, subject_id, out_dir)
        st.write({k: str(v) for k, v in paths.items() if k != "mesh_quality"})
        quality = paths["mesh_quality"]
        if quality is not None:
            st.write(
                f"CFD-ready: **{quality.passes_cfd_check}** "
                f"(watertight={quality.is_watertight}, manifold={quality.is_manifold}, "
                f"euler={quality.euler_number})"
            )

    label_nifti = st.session_state.get("label_nifti")
    if label_nifti is not None and label_nifti.exists():
        st.subheader("Per-case metrics")
        if "mean_probs" not in result:
            st.info("Re-run with 'Compute MC-Dropout uncertainty' checked to see per-case metrics.")
        else:
            out_dir = Path("experiments/demo_exports") / subject_id
            case = compute_case_metrics(result, label_nifti, out_dir, subject_id)
            st.write(case["metrics"])
            st.write(f"ECE: {case['ece']['ece']:.4f}")
            st.image(str(case["reliability_png"]))


if __name__ == "__main__":
    main()

"""Segmentation pipeline orchestrator for CFD-ready SAS extraction.

Runs pre-trained models in sequence to produce multi-label segmentation,
then extracts the CSF fluid domain via Boolean subtraction.

Usage:
    pipeline = SegmentationPipeline(output_dir=Path("outputs/"))
    result = pipeline.run(input_t2w=Path("sub-01_T2w.nii.gz"))
    # result.csf_domain -> Path to CSF domain mask
    # result.meshes -> dict of structure -> STL path
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

from .model_registry import MODELS, ModelSpec, SegTool, check_tool_available

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Results from the segmentation pipeline run."""

    input_path: Path
    cord_mask: Optional[Path] = None
    canal_mask: Optional[Path] = None
    rootlet_mask: Optional[Path] = None
    csf_domain: Optional[Path] = None
    combined_labels: Optional[Path] = None
    success: bool = False
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class SegmentationPipeline:
    """Orchestrates pre-trained models for multi-label SAS segmentation.

    The pipeline runs in sequence:
      1. TotalSpineSeg → cord + canal + vertebrae
      2. model-canal-seg → dural sac (outer boundary)
      3. RootletSeg → nerve rootlets
      4. Boolean extraction: CSF = canal − cord − rootlets

    Args:
        output_dir: Directory for all pipeline outputs.
        use_canal_seg: Use model-canal-seg for outer boundary (default True).
        use_rootlets: Include rootlet segmentation (default True).
        min_file_size: Minimum file size in bytes to consider valid (git-annex check).
    """

    def __init__(
        self,
        output_dir: Path,
        use_canal_seg: bool = True,
        use_rootlets: bool = True,
        min_file_size: int = 1000,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_canal_seg = use_canal_seg
        self.use_rootlets = use_rootlets
        self.min_file_size = min_file_size

    def _validate_input(self, input_path: Path) -> bool:
        """Validate input file exists and is not a git-annex stub."""
        if not input_path.exists():
            logger.error("Input file does not exist: %s", input_path)
            return False

        file_size = input_path.stat().st_size
        if file_size < self.min_file_size:
            logger.error(
                "Input file too small (%d bytes) — likely a git-annex pointer stub: %s",
                file_size,
                input_path,
            )
            return False

        return True

    def _run_sct_command(
        self, model: ModelSpec, input_path: Path, output_path: Path
    ) -> bool:
        """Run an SCT-based segmentation command.

        Args:
            model: ModelSpec with command_template.
            input_path: Path to input NIfTI.
            output_path: Path for output NIfTI.

        Returns:
            True if command succeeded.
        """
        cmd = model.command_template.format(
            input=str(input_path), output=str(output_path)
        )
        logger.info("Running: %s", cmd)

        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max per model
            )
            if result.returncode != 0:
                logger.error(
                    "Command failed (exit %d): %s\nstderr: %s",
                    result.returncode,
                    cmd,
                    result.stderr[:500],
                )
                return False
            return output_path.exists()
        except subprocess.TimeoutExpired:
            logger.error("Command timed out: %s", cmd)
            return False
        except FileNotFoundError:
            logger.error(
                "Command not found. Is %s installed?", model.tool.value
            )
            return False

    def _extract_csf_domain(
        self,
        canal_path: Path,
        cord_path: Path,
        rootlet_path: Optional[Path],
        output_path: Path,
    ) -> bool:
        """Compute CSF domain as Boolean subtraction: canal − cord − rootlets.

        Args:
            canal_path: Path to canal/dural sac binary mask.
            cord_path: Path to spinal cord binary mask.
            rootlet_path: Optional path to rootlet mask.
            output_path: Path for output CSF domain NIfTI.

        Returns:
            True if extraction succeeded.
        """
        logger.info("Extracting CSF domain via Boolean subtraction...")

        canal_nii = nib.load(canal_path)
        cord_nii = nib.load(cord_path)

        canal_data = np.asarray(canal_nii.dataobj).astype(bool)
        cord_data = np.asarray(cord_nii.dataobj).astype(bool)

        # CSF = canal − cord
        csf_data = np.logical_and(canal_data, np.logical_not(cord_data))

        # Subtract rootlets if available
        if rootlet_path is not None and rootlet_path.exists():
            rootlet_nii = nib.load(rootlet_path)
            rootlet_data = np.asarray(rootlet_nii.dataobj).astype(bool)
            csf_data = np.logical_and(csf_data, np.logical_not(rootlet_data))
            logger.info("Subtracted rootlets from CSF domain.")

        # Remove small disconnected components (keep largest)
        labeled, n_components = ndimage.label(csf_data)
        if n_components > 1:
            sizes = ndimage.sum(csf_data, labeled, range(1, n_components + 1))
            largest_label = np.argmax(sizes) + 1
            csf_data = labeled == largest_label
            logger.info(
                "Removed %d small components; kept largest (%d voxels).",
                n_components - 1,
                int(sizes[largest_label - 1]),
            )

        # Compute volume
        spacing = canal_nii.header.get_zooms()[:3]
        voxel_vol_mm3 = float(np.prod(spacing))
        csf_vol_mm3 = csf_data.sum() * voxel_vol_mm3
        csf_vol_cm3 = csf_vol_mm3 / 1000.0
        logger.info("CSF domain volume: %.1f cm³ (Sass ref: 97.3 cm³)", csf_vol_cm3)

        # Save output
        csf_nii = nib.Nifti1Image(
            csf_data.astype(np.uint8), canal_nii.affine, canal_nii.header
        )
        nib.save(csf_nii, output_path)
        logger.info("CSF domain saved to: %s", output_path)

        return True

    def _create_combined_labels(
        self,
        cord_path: Path,
        canal_path: Path,
        rootlet_path: Optional[Path],
        csf_path: Path,
        output_path: Path,
    ) -> bool:
        """Create a combined multi-label NIfTI for visualization and QC.

        Labels:
          0 = background
          1 = spinal cord
          2 = CSF domain
          3 = nerve rootlets
          4 = dura/canal boundary (outer shell)
        """
        cord_nii = nib.load(cord_path)
        cord_data = np.asarray(cord_nii.dataobj).astype(bool)
        canal_data = np.asarray(nib.load(canal_path).dataobj).astype(bool)
        csf_data = np.asarray(nib.load(csf_path).dataobj).astype(bool)

        combined = np.zeros(cord_data.shape, dtype=np.uint8)
        combined[csf_data] = 2  # CSF domain
        combined[cord_data] = 1  # Cord overwrites CSF

        if rootlet_path is not None and rootlet_path.exists():
            rootlet_data = np.asarray(nib.load(rootlet_path).dataobj).astype(bool)
            combined[rootlet_data] = 3

        # Outer boundary (dura shell = canal edge)
        canal_eroded = ndimage.binary_erosion(canal_data)
        dura_shell = np.logical_and(canal_data, np.logical_not(canal_eroded))
        combined[dura_shell] = 4

        combined_nii = nib.Nifti1Image(combined, cord_nii.affine, cord_nii.header)
        nib.save(combined_nii, output_path)
        logger.info("Combined labels saved to: %s", output_path)
        return True

    def run(self, input_t2w: Path) -> PipelineResult:
        """Run the full segmentation pipeline on a T2w volume.

        Args:
            input_t2w: Path to the input T2-weighted NIfTI file.

        Returns:
            PipelineResult with paths to all outputs.
        """
        result = PipelineResult(input_path=input_t2w)

        # Validate input
        if not self._validate_input(input_t2w):
            result.errors.append(f"Invalid input: {input_t2w}")
            return result

        # Derive subject ID for output naming
        subject_id = input_t2w.stem.replace(".nii", "").replace("_T2w", "")
        subj_dir = self.output_dir / subject_id
        subj_dir.mkdir(parents=True, exist_ok=True)

        # ─── Step 1: Spinal cord segmentation ───
        logger.info("Step 1: Spinal cord segmentation...")
        cord_output = subj_dir / f"{subject_id}_label-cord_seg.nii.gz"
        model = MODELS["cord_contrast_agnostic"]

        if not self._run_sct_command(model, input_t2w, cord_output):
            # Fallback to TotalSpineSeg cord
            logger.warning("Cord seg failed; trying TotalSpineSeg...")
            model = MODELS["totalspineseg"]
            tss_output = subj_dir / f"{subject_id}_totalspineseg.nii.gz"
            if self._run_sct_command(model, input_t2w, tss_output):
                # Extract cord label from multi-label output
                tss_nii = nib.load(tss_output)
                tss_data = np.asarray(tss_nii.dataobj)
                cord_data = (tss_data == 1).astype(np.uint8)
                cord_nii = nib.Nifti1Image(cord_data, tss_nii.affine, tss_nii.header)
                nib.save(cord_nii, cord_output)
            else:
                result.errors.append("Failed to segment spinal cord")
                return result

        result.cord_mask = cord_output

        # ─── Step 2: Spinal canal / dural sac segmentation ───
        logger.info("Step 2: Spinal canal segmentation...")
        canal_output = subj_dir / f"{subject_id}_label-canal_seg.nii.gz"

        if self.use_canal_seg:
            model = MODELS["canal_seg"]
            if not self._run_sct_command(model, input_t2w, canal_output):
                # Fallback to TotalSpineSeg canal
                logger.warning("Canal-seg failed; extracting from TotalSpineSeg...")
                tss_output = subj_dir / f"{subject_id}_totalspineseg.nii.gz"
                model = MODELS["totalspineseg"]
                if tss_output.exists() or self._run_sct_command(
                    model, input_t2w, tss_output
                ):
                    tss_nii = nib.load(tss_output)
                    tss_data = np.asarray(tss_nii.dataobj)
                    canal_data = (tss_data == 2).astype(np.uint8)
                    canal_nii = nib.Nifti1Image(
                        canal_data, tss_nii.affine, tss_nii.header
                    )
                    nib.save(canal_nii, canal_output)
                else:
                    result.errors.append("Failed to segment spinal canal")
                    return result
        else:
            # Use TotalSpineSeg directly
            model = MODELS["totalspineseg"]
            tss_output = subj_dir / f"{subject_id}_totalspineseg.nii.gz"
            if tss_output.exists() or self._run_sct_command(
                model, input_t2w, tss_output
            ):
                tss_nii = nib.load(tss_output)
                tss_data = np.asarray(tss_nii.dataobj)
                canal_data = (tss_data == 2).astype(np.uint8)
                canal_nii = nib.Nifti1Image(
                    canal_data, tss_nii.affine, tss_nii.header
                )
                nib.save(canal_nii, canal_output)
            else:
                result.errors.append("Failed to segment spinal canal")
                return result

        result.canal_mask = canal_output

        # ─── Step 3: Nerve rootlet segmentation (optional) ───
        rootlet_output = None
        if self.use_rootlets:
            logger.info("Step 3: Nerve rootlet segmentation...")
            rootlet_output = subj_dir / f"{subject_id}_label-rootlets_seg.nii.gz"
            model = MODELS["rootlet_seg"]
            if not self._run_sct_command(model, input_t2w, rootlet_output):
                logger.warning(
                    "Rootlet segmentation failed; proceeding without rootlets."
                )
                rootlet_output = None

        result.rootlet_mask = rootlet_output

        # ─── Step 4: Extract CSF domain ───
        logger.info("Step 4: Extracting CSF domain (canal - cord - rootlets)...")
        csf_output = subj_dir / f"{subject_id}_label-CSF_domain.nii.gz"

        if not self._extract_csf_domain(
            canal_output, cord_output, rootlet_output, csf_output
        ):
            result.errors.append("CSF domain extraction failed")
            return result

        result.csf_domain = csf_output

        # ─── Step 5: Create combined multi-label map ───
        logger.info("Step 5: Creating combined label map...")
        combined_output = subj_dir / f"{subject_id}_label-combined_dseg.nii.gz"
        self._create_combined_labels(
            cord_output, canal_output, rootlet_output, csf_output, combined_output
        )
        result.combined_labels = combined_output

        result.success = True
        logger.info("Pipeline completed successfully for %s", subject_id)
        return result

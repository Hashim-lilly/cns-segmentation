"""CSF domain extraction and assembly for CFD simulation.

Implements the Boolean subtraction: CSF = canal − cord − rootlets
and produces a unified fluid domain mesh with properly tagged boundaries
(pia wall, dura wall, inlet/outlet caps).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


@dataclass
class DomainMetrics:
    """Geometric metrics for the CSF domain, comparable to Sass 2017 reference.

    Reference values (Sass 2017, healthy female, foramen magnum to S5):
      - Total CSF volume: 97.3 cm³
      - Dura volume: 123.1 cm³
      - Cord volume: 19.9 cm³
      - Rootlet volume: 5.8 cm³
      - Dura surface area: 318.5 cm²
      - Cord surface area: 112.2 cm²
      - Rootlet surface area: 232.1 cm²
      - Max Reynolds: 174.9 (at C3-C4)
      - Avg Womersley: 9.6
    """

    csf_volume_cm3: float = 0.0
    cord_volume_cm3: float = 0.0
    canal_volume_cm3: float = 0.0
    rootlet_volume_cm3: float = 0.0
    csf_voxel_count: int = 0
    cord_voxel_count: int = 0
    n_connected_components: int = 0
    voxel_spacing_mm: tuple[float, ...] = (1.0, 1.0, 1.0)

    # Sass 2017 reference values for comparison
    SASS_CSF_VOLUME_CM3: float = 97.3
    SASS_CORD_VOLUME_CM3: float = 19.9
    SASS_ROOTLET_VOLUME_CM3: float = 5.8

    @property
    def csf_volume_ratio_vs_sass(self) -> float:
        """Ratio of extracted CSF volume to Sass 2017 reference."""
        if self.SASS_CSF_VOLUME_CM3 == 0:
            return 0.0
        return self.csf_volume_cm3 / self.SASS_CSF_VOLUME_CM3


def extract_csf_domain(
    canal_path: Path,
    cord_path: Path,
    rootlet_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    keep_largest_component: bool = True,
    morphological_closing: int = 0,
) -> tuple[np.ndarray, DomainMetrics]:
    """Extract the CSF fluid domain via Boolean subtraction.

    CSF domain = spinal canal (dura sac) − spinal cord − nerve rootlets

    Args:
        canal_path: Path to canal/dural sac segmentation NIfTI.
        cord_path: Path to spinal cord segmentation NIfTI.
        rootlet_path: Optional path to nerve rootlet segmentation.
        output_path: Optional path to save the CSF domain NIfTI.
        keep_largest_component: Remove all but the largest connected component.
        morphological_closing: Iterations of morphological closing to smooth
            the domain boundary (0 = disabled).

    Returns:
        Tuple of (CSF domain numpy array, DomainMetrics).
    """
    # Load masks
    canal_nii = nib.load(canal_path)
    cord_nii = nib.load(cord_path)

    canal_data = np.asarray(canal_nii.dataobj).astype(bool)
    cord_data = np.asarray(cord_nii.dataobj).astype(bool)

    spacing = tuple(float(s) for s in canal_nii.header.get_zooms()[:3])
    voxel_vol_mm3 = float(np.prod(spacing))

    # Boolean subtraction
    csf_data = np.logical_and(canal_data, np.logical_not(cord_data))

    # Subtract rootlets
    rootlet_data = None
    if rootlet_path is not None and Path(rootlet_path).exists():
        rootlet_nii = nib.load(rootlet_path)
        rootlet_data = np.asarray(rootlet_nii.dataobj).astype(bool)
        csf_data = np.logical_and(csf_data, np.logical_not(rootlet_data))

    # Optional morphological closing to smooth boundaries
    if morphological_closing > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        csf_data = ndimage.binary_closing(
            csf_data, structure=struct, iterations=morphological_closing
        )

    # Connected component analysis
    labeled, n_components = ndimage.label(csf_data)

    # Keep largest component
    if keep_largest_component and n_components > 1:
        sizes = ndimage.sum(csf_data, labeled, range(1, n_components + 1))
        largest_label = int(np.argmax(sizes)) + 1
        csf_data = (labeled == largest_label).astype(bool)
        logger.info(
            "Kept largest component (%d of %d); removed %d small fragments.",
            largest_label,
            n_components,
            n_components - 1,
        )

    # Compute metrics
    metrics = DomainMetrics(
        csf_volume_cm3=csf_data.sum() * voxel_vol_mm3 / 1000.0,
        cord_volume_cm3=cord_data.sum() * voxel_vol_mm3 / 1000.0,
        canal_volume_cm3=canal_data.sum() * voxel_vol_mm3 / 1000.0,
        rootlet_volume_cm3=(
            rootlet_data.sum() * voxel_vol_mm3 / 1000.0
            if rootlet_data is not None
            else 0.0
        ),
        csf_voxel_count=int(csf_data.sum()),
        cord_voxel_count=int(cord_data.sum()),
        n_connected_components=n_components,
        voxel_spacing_mm=spacing,
    )

    logger.info(
        "CSF domain: %.1f cm³ (%.0f%% of Sass ref), "
        "cord: %.1f cm³, rootlets: %.1f cm³",
        metrics.csf_volume_cm3,
        metrics.csf_volume_ratio_vs_sass * 100,
        metrics.cord_volume_cm3,
        metrics.rootlet_volume_cm3,
    )

    # Save if output path specified
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        csf_nii = nib.Nifti1Image(
            csf_data.astype(np.uint8), canal_nii.affine, canal_nii.header
        )
        nib.save(csf_nii, output_path)
        logger.info("CSF domain saved: %s", output_path)

    return csf_data.astype(np.uint8), metrics


def compute_cross_sectional_metrics(
    csf_domain: np.ndarray,
    spacing: tuple[float, ...],
    axis: int = 0,
) -> dict[str, np.ndarray]:
    """Compute per-slice cross-sectional area and hydraulic diameter.

    These are the key geometric parameters for validating against
    Sass 2017 Fig 7 hydrodynamic profiles.

    Args:
        csf_domain: 3D binary CSF domain array.
        spacing: Voxel spacing (mm) in each dimension.
        axis: Axis along which to compute slices (0=SI for RAS orientation).

    Returns:
        Dictionary with:
          - slice_positions_mm: axial position of each slice in mm
          - cross_sectional_area_mm2: CSF area per slice
          - hydraulic_diameter_mm: D_H = 4*A/P per slice
          - perimeter_mm: wetted perimeter per slice
    """
    n_slices = csf_domain.shape[axis]
    slice_spacing = spacing[axis]

    # Determine in-plane pixel area
    in_plane_axes = [i for i in range(3) if i != axis]
    pixel_area = spacing[in_plane_axes[0]] * spacing[in_plane_axes[1]]

    areas = np.zeros(n_slices)
    perimeters = np.zeros(n_slices)
    hydraulic_diameters = np.zeros(n_slices)
    positions = np.arange(n_slices) * slice_spacing

    for i in range(n_slices):
        # Extract slice
        slc = [slice(None)] * 3
        slc[axis] = i
        slice_data = csf_domain[tuple(slc)].astype(bool)

        # Cross-sectional area
        area = slice_data.sum() * pixel_area
        areas[i] = area

        if area > 0:
            # Perimeter estimation via edge detection
            struct = ndimage.generate_binary_structure(2, 1)
            eroded = ndimage.binary_erosion(slice_data, structure=struct)
            boundary = np.logical_and(slice_data, np.logical_not(eroded))
            # Approximate perimeter as boundary pixel count × pixel edge length
            avg_pixel_edge = np.sqrt(pixel_area)
            perimeters[i] = boundary.sum() * avg_pixel_edge

            # Hydraulic diameter: D_H = 4A/P
            if perimeters[i] > 0:
                hydraulic_diameters[i] = 4.0 * area / perimeters[i]

    return {
        "slice_positions_mm": positions,
        "cross_sectional_area_mm2": areas,
        "hydraulic_diameter_mm": hydraulic_diameters,
        "perimeter_mm": perimeters,
    }

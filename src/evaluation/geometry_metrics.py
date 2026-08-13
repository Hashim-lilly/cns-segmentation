"""Geometry validation metrics for CFD domain against Sass 2017 reference.

Compares extracted CSF geometry against published reference values:
  - Sass et al. (2017) Fluids Barriers CNS 14:36
  - Total CSF volume: 97.3 cm³
  - Avg Re: 68.5 (max 174.9 at C3-C4)
  - Avg Womersley: 9.6
  - Hydraulic diameter profile along spine (Fig 7c)

Also validates mesh quality against CFD requirements.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import trimesh
except ImportError:
    trimesh = None


# ──────────────────────────────────────────────────────────────────
# Sass 2017 Reference Values
# ──────────────────────────────────────────────────────────────────

@dataclass
class SassReference:
    """Reference geometric and hydrodynamic values from Sass et al. 2017.

    Source: Fluids Barriers CNS 14:36, Table 3 and Fig 7.
    Subject: healthy 23-year-old female, foramen magnum to S5.
    """

    # Volumes (cm³)
    csf_volume_cm3: float = 97.3
    dura_volume_cm3: float = 123.1
    cord_volume_cm3: float = 19.9
    rootlet_volume_cm3: float = 5.8

    # Surface areas (cm²)
    dura_surface_area_cm2: float = 318.5
    cord_surface_area_cm2: float = 112.2
    rootlet_surface_area_cm2: float = 232.1

    # Average cross-sectional areas (cm²)
    avg_dura_csa_cm2: float = 2.03
    avg_cord_csa_cm2: float = 0.33
    avg_rootlet_csa_cm2: float = 0.10

    # Lengths (cm)
    cord_length_cm: float = 44.8
    dura_length_cm: float = 60.4

    # Hydrodynamic parameters
    max_reynolds: float = 174.9
    avg_reynolds: float = 68.5
    avg_womersley: float = 9.6
    max_womersley: float = 22.96
    min_womersley: float = 1.6

    # CSF flow (cm³/s)
    peak_flow_c2c3: float = 4.75
    peak_flow_c7t1: float = 3.05
    peak_flow_t10t11: float = 1.26

    # CSF pulse wave velocity (cm/s)
    pulse_wave_velocity: float = 19.4


SASS_REF = SassReference()


@dataclass
class GeometryValidationResult:
    """Results of geometry validation against Sass reference."""

    # Volume comparisons
    csf_volume_cm3: float = 0.0
    volume_error_percent: float = 0.0
    volume_within_tolerance: bool = False

    # Surface comparisons (if mesh available)
    surface_area_cm2: Optional[float] = None

    # Cross-sectional metrics
    avg_csa_mm2: float = 0.0
    max_csa_mm2: float = 0.0
    min_csa_mm2: float = 0.0
    avg_hydraulic_diameter_mm: float = 0.0

    # Mesh quality
    is_watertight: bool = False
    is_manifold: bool = False
    euler_number: int = 0

    # Overall pass
    passes_geometry_check: bool = False
    notes: list[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def validate_against_sass(
    csf_volume_cm3: float,
    cord_volume_cm3: float = 0.0,
    rootlet_volume_cm3: float = 0.0,
    csa_profile: Optional[np.ndarray] = None,
    mesh_path: Optional[Path] = None,
    volume_tolerance_percent: float = 10.0,
) -> GeometryValidationResult:
    """Validate extracted geometry against Sass 2017 reference.

    Args:
        csf_volume_cm3: Extracted CSF domain volume in cm³.
        cord_volume_cm3: Extracted cord volume in cm³.
        rootlet_volume_cm3: Extracted rootlet volume in cm³.
        csa_profile: Array of per-slice cross-sectional areas (mm²).
        mesh_path: Optional path to STL mesh for surface validation.
        volume_tolerance_percent: Acceptable deviation from Sass ref (default 10%).

    Returns:
        GeometryValidationResult with all comparisons.
    """
    result = GeometryValidationResult()
    result.csf_volume_cm3 = csf_volume_cm3

    # Volume validation
    ref_vol = SASS_REF.csf_volume_cm3
    result.volume_error_percent = abs(csf_volume_cm3 - ref_vol) / ref_vol * 100.0
    result.volume_within_tolerance = result.volume_error_percent <= volume_tolerance_percent

    if result.volume_within_tolerance:
        result.notes.append(
            f"✓ CSF volume {csf_volume_cm3:.1f} cm³ within {volume_tolerance_percent}% "
            f"of Sass ref ({ref_vol:.1f} cm³)"
        )
    else:
        result.notes.append(
            f"✗ CSF volume {csf_volume_cm3:.1f} cm³ deviates {result.volume_error_percent:.1f}% "
            f"from Sass ref ({ref_vol:.1f} cm³)"
        )

    # Cord volume sanity check
    if cord_volume_cm3 > 0:
        cord_error = abs(cord_volume_cm3 - SASS_REF.cord_volume_cm3) / SASS_REF.cord_volume_cm3 * 100
        if cord_error > 30:
            result.notes.append(
                f"⚠ Cord volume {cord_volume_cm3:.1f} cm³ deviates {cord_error:.0f}% "
                f"from Sass ref ({SASS_REF.cord_volume_cm3:.1f} cm³) — "
                f"may indicate partial coverage (cervical only?)"
            )

    # Cross-sectional profile
    if csa_profile is not None:
        valid_slices = csa_profile[csa_profile > 0]
        if len(valid_slices) > 0:
            result.avg_csa_mm2 = float(valid_slices.mean())
            result.max_csa_mm2 = float(valid_slices.max())
            result.min_csa_mm2 = float(valid_slices.min())
            # Sass avg CSA for SSS = dura - cord - rootlets ≈ 2.03 - 0.33 - 0.10 = 1.60 cm²
            expected_avg_csa_mm2 = 160.0  # 1.60 cm² in mm²
            csa_ratio = result.avg_csa_mm2 / expected_avg_csa_mm2
            result.notes.append(
                f"  CSA profile: avg={result.avg_csa_mm2:.1f} mm², "
                f"ratio to Sass ref: {csa_ratio:.2f}"
            )

    # Mesh quality (if STL provided)
    if mesh_path is not None and trimesh is not None:
        mesh = trimesh.load(str(mesh_path))
        result.is_watertight = mesh.is_watertight
        result.is_manifold = mesh.is_winding_consistent
        result.euler_number = mesh.euler_number
        if mesh.is_watertight:
            result.surface_area_cm2 = mesh.area / 100.0  # mm² → cm²
            result.notes.append(
                f"  Mesh surface area: {result.surface_area_cm2:.1f} cm² "
                f"(Sass dura: {SASS_REF.dura_surface_area_cm2:.1f} cm²)"
            )

    # Overall pass
    result.passes_geometry_check = result.volume_within_tolerance
    if mesh_path is not None:
        result.passes_geometry_check = (
            result.passes_geometry_check
            and result.is_watertight
            and result.is_manifold
        )

    return result


def compute_hydraulic_diameter_profile(
    csa_profile: np.ndarray,
    perimeter_profile: np.ndarray,
) -> np.ndarray:
    """Compute hydraulic diameter along the spine: D_H = 4A/P.

    Args:
        csa_profile: Cross-sectional area per slice (mm²).
        perimeter_profile: Wetted perimeter per slice (mm).

    Returns:
        Hydraulic diameter per slice (mm). Sass ref avg: ~10 mm.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        dh = np.where(perimeter_profile > 0, 4 * csa_profile / perimeter_profile, 0.0)
    return dh


def estimate_reynolds_number(
    csa_mm2: float,
    hydraulic_diameter_mm: float,
    peak_flow_cm3_per_s: float,
    kinematic_viscosity: float = 0.7e-6,  # water at 37°C in m²/s
) -> float:
    """Estimate Reynolds number at a spinal level.

    Re = Q_peak * D_H / (ν * A_cs)

    Args:
        csa_mm2: Cross-sectional area in mm².
        hydraulic_diameter_mm: Hydraulic diameter in mm.
        peak_flow_cm3_per_s: Peak volumetric flow rate in cm³/s.
        kinematic_viscosity: CSF kinematic viscosity (m²/s), default water at 37°C.

    Returns:
        Reynolds number (dimensionless). Sass ref: max 174.9 at C3-C4.
    """
    # Convert units to SI
    csa_m2 = csa_mm2 * 1e-6
    dh_m = hydraulic_diameter_mm * 1e-3
    q_m3_per_s = peak_flow_cm3_per_s * 1e-6

    if csa_m2 == 0:
        return 0.0

    # Mean velocity = Q / A
    v_mean = q_m3_per_s / csa_m2

    # Re = v * D_H / ν
    re = v_mean * dh_m / kinematic_viscosity
    return float(re)

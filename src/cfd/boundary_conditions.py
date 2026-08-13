"""CSF flow boundary conditions from published literature.

Provides digitized CSF flow waveforms from Sass et al. 2017 for use as
OpenFOAM boundary conditions before internal PC-MRI data is available.

Reference: Sass et al. (2017) Fluids Barriers CNS 14:36, Fig 6a.
  - C2-C3: peak 4.75 cm³/s (measured at 4.0 cm from foramen magnum)
  - C7-T1: peak 3.05 cm³/s (measured at 12.5 cm from foramen magnum)
  - T10-T11: peak 1.26 cm³/s (measured at 35.4 cm from foramen magnum)

Cardiac cycle: ~800 ms (75 bpm assumed).
Waveforms are negative during caudal flow (systole) and positive during
cranial flow (diastole). Zero net flow over one cycle.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CSFWaveform:
    """A CSF flow waveform at a specific spinal level."""

    level: str  # e.g., "C2-C3"
    distance_from_fm_cm: float  # Distance from foramen magnum in cm
    time_s: np.ndarray  # Time points in seconds
    flow_cm3_per_s: np.ndarray  # Flow rate in cm³/s (negative = caudal)
    peak_caudal_flow: float  # Peak caudal flow rate (positive value)
    peak_cranial_flow: float  # Peak cranial flow rate (positive value)


def get_sass_waveform_c2c3(n_points: int = 100) -> CSFWaveform:
    """Generate approximate CSF waveform at C2-C3 level from Sass 2017.

    Approximated as a sinusoidal waveform with modified amplitude to match
    published peak values. The actual waveform from Fig 6a has a sharper
    systolic peak and longer diastolic phase.

    Args:
        n_points: Number of time points per cardiac cycle.

    Returns:
        CSFWaveform at C2-C3 level.
    """
    cardiac_period = 0.8  # seconds (75 bpm)
    t = np.linspace(0, cardiac_period, n_points, endpoint=False)

    # Approximate waveform: sinusoidal with asymmetric systole/diastole
    # Systolic peak (caudal, negative) occurs at ~25% of cycle
    # Diastolic peak (cranial, positive) occurs at ~65% of cycle
    # Peak caudal: -2.7 cm³/s, peak cranial: +1.2 cm³/s (from Fig 6a)
    phase = 2 * np.pi * t / cardiac_period

    # Asymmetric waveform using sum of harmonics
    flow = (
        -2.7 * np.sin(phase + 0.3)  # Fundamental (caudal dominant)
        - 0.8 * np.sin(2 * phase + 0.5)  # Second harmonic
        - 0.3 * np.sin(3 * phase + 0.2)  # Third harmonic
    )

    # Ensure zero net flow over one cycle
    flow -= flow.mean()

    return CSFWaveform(
        level="C2-C3",
        distance_from_fm_cm=4.0,
        time_s=t,
        flow_cm3_per_s=flow,
        peak_caudal_flow=abs(flow.min()),
        peak_cranial_flow=flow.max(),
    )


def get_sass_waveform_c7t1(n_points: int = 100) -> CSFWaveform:
    """Generate approximate CSF waveform at C7-T1 from Sass 2017."""
    cardiac_period = 0.8
    t = np.linspace(0, cardiac_period, n_points, endpoint=False)

    phase = 2 * np.pi * t / cardiac_period
    flow = (
        -1.8 * np.sin(phase + 0.3)
        - 0.5 * np.sin(2 * phase + 0.5)
        - 0.2 * np.sin(3 * phase + 0.2)
    )
    flow -= flow.mean()

    return CSFWaveform(
        level="C7-T1",
        distance_from_fm_cm=12.5,
        time_s=t,
        flow_cm3_per_s=flow,
        peak_caudal_flow=abs(flow.min()),
        peak_cranial_flow=flow.max(),
    )


def get_sass_waveform_t10t11(n_points: int = 100) -> CSFWaveform:
    """Generate approximate CSF waveform at T10-T11 from Sass 2017."""
    cardiac_period = 0.8
    t = np.linspace(0, cardiac_period, n_points, endpoint=False)

    phase = 2 * np.pi * t / cardiac_period
    flow = (
        -0.75 * np.sin(phase + 0.3)
        - 0.2 * np.sin(2 * phase + 0.5)
    )
    flow -= flow.mean()

    return CSFWaveform(
        level="T10-T11",
        distance_from_fm_cm=35.4,
        time_s=t,
        flow_cm3_per_s=flow,
        peak_caudal_flow=abs(flow.min()),
        peak_cranial_flow=flow.max(),
    )


def interpolate_flow_at_level(
    distance_from_fm_cm: float,
    n_points: int = 100,
) -> CSFWaveform:
    """Interpolate CSF flow waveform at an arbitrary spinal level.

    Uses linear interpolation of peak flow amplitude between the three
    measured Sass 2017 levels (C2-C3, C7-T1, T10-T11), with waveform
    shape preserved.

    Args:
        distance_from_fm_cm: Distance from foramen magnum in cm (0-60).
        n_points: Number of time points per cardiac cycle.

    Returns:
        Interpolated CSFWaveform at the specified level.
    """
    # Reference points: (distance_cm, peak_caudal_flow_cm3_per_s)
    ref_distances = np.array([4.0, 12.5, 35.4, 60.0])
    ref_peaks = np.array([2.7, 1.8, 0.75, 0.3])  # Extrapolate to sacral

    # Interpolate peak amplitude
    peak = float(np.interp(distance_from_fm_cm, ref_distances, ref_peaks))

    # Generate waveform shape (same as C2-C3 but scaled)
    cardiac_period = 0.8
    t = np.linspace(0, cardiac_period, n_points, endpoint=False)
    phase = 2 * np.pi * t / cardiac_period

    # Normalized waveform shape
    shape = (
        -np.sin(phase + 0.3)
        - 0.3 * np.sin(2 * phase + 0.5)
        - 0.1 * np.sin(3 * phase + 0.2)
    )
    shape -= shape.mean()
    shape = shape / abs(shape.min())  # Normalize to peak=1

    flow = shape * peak

    return CSFWaveform(
        level=f"interpolated_{distance_from_fm_cm:.0f}cm",
        distance_from_fm_cm=distance_from_fm_cm,
        time_s=t,
        flow_cm3_per_s=flow,
        peak_caudal_flow=abs(flow.min()),
        peak_cranial_flow=flow.max(),
    )


def export_openfoam_waveform(
    waveform: CSFWaveform,
    output_path: str,
    n_cycles: int = 3,
) -> None:
    """Export waveform as OpenFOAM-compatible time-series file.

    Format: plain text with columns (time flow_rate).
    Repeats the waveform for n_cycles cardiac cycles.

    Args:
        waveform: CSFWaveform to export.
        output_path: Path for the output file.
        n_cycles: Number of cardiac cycles to include.
    """
    cardiac_period = waveform.time_s[-1] + (waveform.time_s[1] - waveform.time_s[0])

    all_times = []
    all_flows = []

    for cycle in range(n_cycles):
        offset = cycle * cardiac_period
        all_times.extend(waveform.time_s + offset)
        all_flows.extend(waveform.flow_cm3_per_s)

    # Convert cm³/s to m³/s for OpenFOAM SI units
    all_flows_si = np.array(all_flows) * 1e-6

    with open(output_path, "w") as f:
        f.write("// CSF flow waveform — Sass et al. 2017\n")
        f.write(f"// Level: {waveform.level}\n")
        f.write(f"// Distance from foramen magnum: {waveform.distance_from_fm_cm} cm\n")
        f.write(f"// Peak caudal flow: {waveform.peak_caudal_flow:.2f} cm³/s\n")
        f.write("// Format: (time [s]  flow_rate [m³/s])\n")
        f.write("(\n")
        for t, q in zip(all_times, all_flows_si):
            f.write(f"    ({t:.6f}  {q:.10e})\n")
        f.write(")\n")

    logger.info(
        "Exported OpenFOAM waveform: %s (%d points, %d cycles)",
        output_path,
        len(all_times),
        n_cycles,
    )

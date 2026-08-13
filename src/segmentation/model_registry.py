"""Registry of pre-trained segmentation models for the CFD pipeline.

Each model entry tracks: name, version, source tool, expected inputs/outputs,
and the structures it segments. This enables version-locked reproducible runs
and automated QC reporting.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class SegTool(Enum):
    """Supported segmentation toolkits."""

    SCT = "spinalcordtoolbox"
    TOTALSPINESEG = "totalspineseg"
    CANAL_SEG = "model-canal-seg"
    ROOTLET_SEG = "model-spinal-rootlets"
    SPINEPS = "spineps"
    CUSTOM_NNUNET = "nnunet_custom"


@dataclass
class ModelSpec:
    """Specification for a pre-trained segmentation model."""

    name: str
    tool: SegTool
    version: str
    structures: list[str]
    input_contrast: str = "T2w"
    input_orientation: str = "RPI"
    description: str = ""
    url: str = ""
    command_template: str = ""
    output_labels: dict[str, int] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────────────────────────

MODELS: dict[str, ModelSpec] = {
    "cord_contrast_agnostic": ModelSpec(
        name="cord_contrast_agnostic",
        tool=SegTool.SCT,
        version="2.7",
        structures=["spinal_cord"],
        input_contrast="any",
        description="SCT contrast-agnostic spinal cord segmentation (nnU-Net based)",
        url="https://github.com/sct-pipeline/contrast-agnostic-softseg-spinalcord",
        command_template="sct_deepseg -task seg_sc_contrast_agnostic -i {input} -o {output}",
        output_labels={"spinal_cord": 1},
    ),
    "totalspineseg": ModelSpec(
        name="totalspineseg",
        tool=SegTool.TOTALSPINESEG,
        version="2.0",
        structures=[
            "spinal_cord",
            "spinal_canal",
            "vertebrae",
            "intervertebral_discs",
        ],
        input_contrast="T2w",
        description=(
            "Whole-spine multi-label segmentation via nnU-Net. "
            "Contrast/resolution/orientation-robust. Run via sct_deepseg."
        ),
        url="https://github.com/neuropoly/totalspineseg",
        command_template="sct_deepseg -task totalspineseg -i {input} -o {output}",
        output_labels={
            "spinal_cord": 1,
            "spinal_canal": 2,
            # Vertebrae and IVDs use higher label indices
        },
    ),
    "canal_seg": ModelSpec(
        name="canal_seg",
        tool=SegTool.CANAL_SEG,
        version="1.0",
        structures=["dural_sac"],
        input_contrast="T2w",
        description=(
            "Dural sac segmentation (cord + CSF + rootlets envelope) on T2w. "
            "Directly yields the fluid-domain outer boundary."
        ),
        url="https://github.com/ivadomed/model-canal-seg",
        command_template="sct_deepseg -task seg_canal_t2w -i {input} -o {output}",
        output_labels={"dural_sac": 1},
    ),
    "rootlet_seg": ModelSpec(
        name="rootlet_seg",
        tool=SegTool.ROOTLET_SEG,
        version="1.0",
        structures=["dorsal_rootlets", "ventral_rootlets"],
        input_contrast="T2w",
        description=(
            "Dorsal and ventral nerve rootlet segmentation C2-T1. "
            "Valošek 2024: Dice 0.67±0.16."
        ),
        url="https://github.com/ivadomed/model-spinal-rootlets",
        command_template="sct_deepseg -task seg_spinal_rootlets_t2w -i {input} -o {output}",
        output_labels={"dorsal_rootlets": 1, "ventral_rootlets": 2},
    ),
}


def get_model(name: str) -> ModelSpec:
    """Retrieve a model specification by name.

    Args:
        name: Registry key for the model.

    Returns:
        ModelSpec for the requested model.

    Raises:
        KeyError: If the model name is not in the registry.
    """
    if name not in MODELS:
        available = ", ".join(MODELS.keys())
        raise KeyError(
            f"Model '{name}' not found. Available: {available}"
        )
    return MODELS[name]


def list_models() -> list[str]:
    """List all registered model names."""
    return list(MODELS.keys())


def check_tool_available(tool: SegTool) -> bool:
    """Check whether the required segmentation tool is installed.

    Args:
        tool: The SegTool enum value to check.

    Returns:
        True if the tool's CLI is accessible.
    """
    import shutil

    tool_commands = {
        SegTool.SCT: "sct_deepseg",
        SegTool.TOTALSPINESEG: "sct_deepseg",
        SegTool.CANAL_SEG: "sct_deepseg",
        SegTool.ROOTLET_SEG: "sct_deepseg",
        SegTool.SPINEPS: "spineps",
        SegTool.CUSTOM_NNUNET: "nnUNetv2_predict",
    }

    cmd = tool_commands.get(tool)
    if cmd is None:
        logger.warning("Unknown tool: %s", tool)
        return False

    available = shutil.which(cmd) is not None
    if not available:
        logger.warning(
            "Tool '%s' (command: '%s') not found in PATH.", tool.value, cmd
        )
    return available

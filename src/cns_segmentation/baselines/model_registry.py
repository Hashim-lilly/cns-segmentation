"""Minimal registry of pretrained segmentation CLIs used as comparison baselines.

Deliberately not a cross-repo import of `cns_cfd.segmentation_bridge.model_registry`
(that module serves the CFD pipeline's own fallback chain and has a different
purpose/shape) — this is a small, local registry scoped to exactly the two
baselines this repo's held-out sites can be scored against: SCT's
contrast-agnostic cord model and SCT's rootlet model. Both are invoked via
`sct_deepseg`, installed at /home/l091835/sct_7.3.
"""

from dataclasses import dataclass


@dataclass
class ModelSpec:
    """Specification for a pretrained baseline segmentation CLI."""

    name: str
    structure: str
    command_template: str
    description: str = ""


MODELS: dict[str, ModelSpec] = {
    "sct_cord": ModelSpec(
        name="sct_cord",
        structure="cord",
        command_template="sct_deepseg spinalcord -i {input} -o {output}",
        description="SCT 7.3 contrast-agnostic spinal cord segmentation (nnU-Net based).",
    ),
    "sct_rootlets": ModelSpec(
        name="sct_rootlets",
        structure="rootlets",
        command_template="sct_deepseg rootlets -i {input} -o {output}",
        description=(
            "SCT 7.3 dorsal/ventral nerve rootlet segmentation. Outputs "
            "per-level class ids (2-9), not binary — evaluate_subject()'s "
            "class_map=None path binarizes both pred and label via >0, which "
            "is the correct comparison for this baseline (verified: SCT's "
            "native output grid matches the raw ground-truth label grid "
            "exactly, no resampling needed)."
        ),
    ),
}


def get_model(name: str) -> ModelSpec:
    """Retrieve a baseline model specification by name.

    Raises:
        KeyError: If `name` is not in the registry.
    """
    if name not in MODELS:
        raise KeyError(f"Baseline model '{name}' not found. Available: {list(MODELS)}")
    return MODELS[name]

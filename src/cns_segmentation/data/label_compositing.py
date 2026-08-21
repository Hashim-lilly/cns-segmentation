"""Merge per-structure binary label masks into one integer multi-class volume.

Spine-generic's ground truth is one binary mask file per anatomical
structure (cord, canal, csf, rootlets). A multi-class SegResNet needs a
single integer label volume instead, so `CompositeLabeld` combines the
per-structure masks that `spine_generic.create_datalist()` /
`SegmentationTrainer.setup_data()` load under separate `label_<structure>`
keys.
"""

import logging
from typing import Optional

import numpy as np
import torch
from monai.transforms import MapTransform, ResizeWithPadOrCrop

logger = logging.getLogger(__name__)

DEFAULT_LABEL_PRIORITY = ["canal", "thecal_sac", "csf", "cord", "rootlets"]
"""Overlap-resolution order, lowest to highest priority (later wins).

Verified empirically on 2026-08-19 against all reachable spine-generic
on-disk masks, each pair resampled to the same RAS / (1.0, 0.5, 0.5)mm grid
used by `cns_segmentation.data.transforms` before comparison (per-structure
masks are stored at different native resolutions per subject and are not
directly comparable before resampling). Method and full per-pair table
(% of A inside B, % of B inside A, Jaccard) are reproducible via
`dev_scratch/check_label_overlap.py` if that script is still present.

- canal is the anatomical outer container: 81-93% of cord voxels (mean 89%),
  93-100% of csf voxels (mean 97%), and 79-93% of rootlets voxels (mean 87%)
  fall inside it across subjects (n=254/10/20) — never complete containment,
  but always the large majority, so canal must never win an overlap.
- csf and cord are nearly mutually exclusive: 0-2.4% of csf voxels fall
  inside cord (mean 0.7%, n=10) — boundary/partial-volume only.
- rootlets barely touch cord: 0-3.0% of rootlets voxels fall inside cord
  (mean 1.1%, n=20, 1 subject skipped for a mismatched grid) — rootlets are
  also the most data-starved, thinnest structure, so they take highest
  priority to avoid being silently erased at their boundary with cord.
- thecal_sac (Phase 2, Al-Kafri/Mendeley) is placed directly above canal,
  provisionally: it's a lumbar-region outer-container structure analogous to
  canal, and never co-occurs with canal/csf/cord/rootlets on the same subject
  (spine-generic is cervical-thoracic, Al-Kafri is lumbar), so no real
  overlap data exists yet to verify this ordering against. Revisit once a
  subject with both a thecal_sac mask and another structure's mask exists.
"""


class CompositeLabeld(MapTransform):
    """Merge N single-structure binary label keys into one integer label map.

    Structure names not present in `structure_keys` are simply absent from
    the output volume — this transform does not require `structure_keys` to
    cover every entry in `DEFAULT_LABEL_PRIORITY`.

    Args:
        structure_keys: Maps structure name (e.g. "cord") to the dict key
            holding its loaded binary mask tensor (e.g. "label_cord").
        priority: Overlap-resolution order, lowest to highest priority.
            Also determines output class ids: background=0, then 1..N in
            this filtered priority order. Defaults to
            `DEFAULT_LABEL_PRIORITY`.
        output_key: Dict key to write the merged integer label to.
            Default "label".

    Raises:
        ValueError: If a structure in `structure_keys` is not found anywhere
            in `priority`.
    """

    def __init__(
        self,
        structure_keys: dict[str, str],
        priority: Optional[list[str]] = None,
        output_key: str = "label",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(
            keys=list(structure_keys.values()), allow_missing_keys=allow_missing_keys
        )
        self.structure_keys = structure_keys
        self.output_key = output_key

        priority = priority or DEFAULT_LABEL_PRIORITY
        ordered = [s for s in priority if s in structure_keys]
        missing = set(structure_keys) - set(ordered)
        if missing:
            raise ValueError(
                f"Structures {missing} not found in priority order {priority}"
            )
        self.class_map: dict[str, int] = {s: i + 1 for i, s in enumerate(ordered)}
        logger.info("CompositeLabeld class map: %s", self.class_map)

    def __call__(self, data: dict) -> dict:
        """Merge the configured structure masks into one integer label.

        Args:
            data: Dict containing each key in `structure_keys.values()`,
                each a channel-first binary(-ish, >0 thresholded) mask
                tensor/array of identical shape.

        Returns:
            Copy of `data` with `output_key` set to the merged integer
            label volume.
        """
        d = dict(data)
        ref_key = next(iter(self.structure_keys.values()))
        ref = d[ref_key]
        is_tensor = isinstance(ref, torch.Tensor)
        merged = (
            torch.zeros_like(ref, dtype=torch.uint8)
            if is_tensor
            else np.zeros_like(ref, dtype=np.uint8)
        )

        for structure, class_id in self.class_map.items():
            key = self.structure_keys[structure]
            mask_src = d[key]
            if mask_src.shape[1:] != ref.shape[1:]:
                # Same-subject structure masks can leave `Spacingd` with
                # different array shapes if their raw derivative files were
                # produced at a different native FOV than the reference
                # image (seen in practice on spine-generic derivatives —
                # resampling to a common voxel spacing doesn't guarantee a
                # common array shape unless the physical extent already
                # matched). Center pad/crop to the reference shape rather
                # than crash the whole run on one subject's file quirk.
                logger.warning(
                    "CompositeLabeld: '%s' mask shape %s != reference shape %s "
                    "for subject %s — center pad/cropping to match.",
                    structure,
                    tuple(mask_src.shape),
                    tuple(ref.shape),
                    d.get("subject", "<unknown>"),
                )
                mask_src = ResizeWithPadOrCrop(spatial_size=ref.shape[1:])(mask_src)
            mask = mask_src > 0
            merged[mask] = class_id

        d[self.output_key] = merged
        return d

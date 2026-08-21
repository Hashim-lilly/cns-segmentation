"""Tests for cns_segmentation.data.label_compositing.CompositeLabeld."""

import pytest
import torch

from cns_segmentation.data.label_compositing import CompositeLabeld


def _mask(shape, region):
    m = torch.zeros(shape, dtype=torch.uint8)
    m[region] = 1
    return m


class TestCompositeLabeldDefaultPriority:
    def test_overlap_resolves_to_higher_priority_structure(self):
        shape = (1, 4, 4, 4)
        # canal and cord fully overlap; DEFAULT_LABEL_PRIORITY ranks cord above canal.
        data = {
            "label_canal": _mask(shape, (slice(None), slice(0, 2), slice(0, 2), slice(0, 2))),
            "label_cord": _mask(shape, (slice(None), slice(0, 2), slice(0, 2), slice(0, 2))),
        }
        transform = CompositeLabeld(structure_keys={"canal": "label_canal", "cord": "label_cord"})
        merged = transform(data)["label"]
        assert merged[0, 0, 0, 0] == transform.class_map["cord"]

    def test_non_overlapping_regions_keep_their_own_class(self):
        shape = (1, 4, 4, 4)
        data = {
            "label_canal": _mask(shape, (slice(None), slice(0, 2), slice(0, 4), slice(0, 4))),
            "label_cord": _mask(shape, (slice(None), slice(2, 4), slice(0, 4), slice(0, 4))),
        }
        transform = CompositeLabeld(structure_keys={"canal": "label_canal", "cord": "label_cord"})
        merged = transform(data)["label"]
        assert merged[0, 0, 0, 0] == transform.class_map["canal"]
        assert merged[0, 3, 0, 0] == transform.class_map["cord"]

    def test_background_stays_zero(self):
        shape = (1, 4, 4, 4)
        data = {"label_cord": torch.zeros(shape, dtype=torch.uint8)}
        transform = CompositeLabeld(structure_keys={"cord": "label_cord"})
        merged = transform(data)["label"]
        assert torch.all(merged == 0)


class TestCompositeLabeldCustomPriority:
    def test_custom_priority_flips_overlap_winner(self):
        shape = (1, 4, 4, 4)
        data = {
            "label_canal": _mask(shape, (slice(None), slice(0, 2), slice(0, 2), slice(0, 2))),
            "label_cord": _mask(shape, (slice(None), slice(0, 2), slice(0, 2), slice(0, 2))),
        }
        transform = CompositeLabeld(
            structure_keys={"canal": "label_canal", "cord": "label_cord"},
            priority=["cord", "canal"],
        )
        merged = transform(data)["label"]
        assert merged[0, 0, 0, 0] == transform.class_map["canal"]


class TestCompositeLabeldClassMap:
    def test_class_map_follows_filtered_priority_order(self):
        transform = CompositeLabeld(
            structure_keys={
                "cord": "label_cord",
                "rootlets": "label_rootlets",
                "canal": "label_canal",
            },
        )
        # DEFAULT_LABEL_PRIORITY = ["canal", "thecal_sac", "csf", "cord",
        # "rootlets"]; thecal_sac/csf absent here.
        assert transform.class_map == {"canal": 1, "cord": 2, "rootlets": 3}

    def test_missing_structure_in_priority_raises(self):
        with pytest.raises(ValueError, match="not found in priority"):
            CompositeLabeld(structure_keys={"cord": "label_cord", "not_a_structure": "label_x"})


class TestCompositeLabeldThecalSac:
    def test_thecal_sac_ranks_between_canal_and_csf(self):
        transform = CompositeLabeld(
            structure_keys={
                "canal": "label_canal",
                "thecal_sac": "label_thecal_sac",
                "csf": "label_csf",
            },
        )
        # DEFAULT_LABEL_PRIORITY = ["canal", "thecal_sac", "csf", "cord", "rootlets"].
        assert transform.class_map == {"canal": 1, "thecal_sac": 2, "csf": 3}


class TestCompositeLabeldShapeMismatch:
    def test_mismatched_mask_shape_is_aligned_not_crashed(self, caplog):
        # Mirrors a real spine-generic case: one structure's derivative
        # resamples to a different array shape than the reference image.
        data = {
            "label_cord": _mask((1, 4, 8, 8), (slice(None), slice(None), slice(0, 4), slice(0, 4))),
            "label_canal": _mask((1, 4, 6, 6), (slice(None), slice(None), slice(0, 6), slice(0, 6))),
        }
        transform = CompositeLabeld(structure_keys={"cord": "label_cord", "canal": "label_canal"})
        merged = transform(data)["label"]
        assert merged.shape == (1, 4, 8, 8)
        assert "shape" in caplog.text

    def test_aligned_mask_still_contributes_its_class(self):
        shape_ref = (1, 4, 8, 8)
        shape_other = (1, 4, 4, 4)
        data = {
            "label_cord": torch.zeros(shape_ref, dtype=torch.uint8),
            "label_canal": torch.ones(shape_other, dtype=torch.uint8),
        }
        transform = CompositeLabeld(structure_keys={"cord": "label_cord", "canal": "label_canal"})
        merged = transform(data)["label"]
        # Center-padded 4x4x4 block of 1s should land in the middle of the 8x8 plane.
        assert merged[0, 0, 2, 2] == transform.class_map["canal"]
        assert merged[0, 0, 0, 0] == 0


class TestCompositeLabeldOutputKey:
    def test_custom_output_key(self):
        shape = (1, 4, 4, 4)
        data = {"label_cord": _mask(shape, (slice(None), slice(0, 2), slice(0, 2), slice(0, 2)))}
        transform = CompositeLabeld(structure_keys={"cord": "label_cord"}, output_key="label_merged")
        result = transform(data)
        assert "label_merged" in result
        assert "label" not in result

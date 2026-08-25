"""Canonical manifold check, shared by cns-segmentation and cns-cfd-simulation.

trimesh's `is_watertight` already means "every edge is shared by exactly two
faces" (closed + edge-manifold) — computed via `trimesh.graph.is_watertight`.
`is_winding_consistent` only checks orientation *among edges that already
passed that exactly-2 filter*, so it says nothing about boundary edges: an
open mesh (e.g. one face removed from a closed sphere) is `is_watertight=False`
but `is_winding_consistent=True`. Neither field alone is a correct manifold
check; both together are.
"""


def is_manifold(mesh: "trimesh.Trimesh") -> bool:  # noqa: F821 - trimesh optional import
    """True if `mesh` is a closed, edge-manifold, consistently-oriented surface.

    Args:
        mesh: trimesh.Trimesh to check.

    Returns:
        `mesh.is_watertight and mesh.is_winding_consistent`.
    """
    return bool(mesh.is_watertight) and bool(mesh.is_winding_consistent)

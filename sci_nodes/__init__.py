"""sci-nodes: scipy-dependent nonoha nodes.

Butterworth / notch / SMA filters and polyphase resampling, kept in lockstep
with nonoha-engine's ``node_flow/nodes/filter_nodes.py`` (same classes, same
registered type names) so graphs authored in nonoha-engine load unchanged in
`nonoha-embedded`.

Register with the shared contract ``register_nodes(registry)``.
"""

from nonoha_core.node_registry import NodeRegistry

from .filter_nodes import (
    BandPassFilterNode,
    HighPassFilterNode,
    LowPassFilterNode,
    NotchFilterNode,
    ResamplePolyNode,
    SmaFilterNode,
)


def register_nodes(registry: NodeRegistry) -> None:
    """Register scipy-based filter/resample nodes (same names as nonoha-engine)."""
    registry.register_node("BandPassFilter", BandPassFilterNode)
    registry.register_node("HighPassFilter", HighPassFilterNode)
    registry.register_node("LowPassFilter", LowPassFilterNode)
    registry.register_node("NotchFilter", NotchFilterNode)
    registry.register_node("SmaFilter", SmaFilterNode)
    registry.register_node("ResamplePoly", ResamplePolyNode)


__all__ = [
    "register_nodes",
    "BandPassFilterNode",
    "HighPassFilterNode",
    "LowPassFilterNode",
    "NotchFilterNode",
    "SmaFilterNode",
    "ResamplePolyNode",
]

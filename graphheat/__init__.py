"""HeatAdapter molecular graph models; the graphheat import name is kept for compatibility."""

__version__ = "0.1.0"

from .operators import GraphHeatNet, GraphHeatNet_NoGate, GraphHeatOperator
from .models_pyg import (
    APPNPNet,
    AttentiveFPHeatNet,
    AttentiveFPNet,
    GATNet,
    GATv2MLPAdapterNet,
    GATv2Net,
    GCNNet,
    GCNMLPAdapterNet,
    GINEMLPAdapterNet,
    GINEHeatNet,
    GINENet,
    GINMLPAdapterNet,
    GINNet,
    GPSNet,
    GraphSAGENet,
    GraphormerNet,
    MixHopNet,
    MPNNMLPAdapterNet,
    MPNNNet,
    PNANet,
)

__all__ = [
    "GraphHeatNet",
    "GraphHeatNet_NoGate",
    "GraphHeatOperator",
    "APPNPNet",
    "AttentiveFPHeatNet",
    "AttentiveFPNet",
    "GATNet",
    "GATv2MLPAdapterNet",
    "GATv2Net",
    "GCNNet",
    "GCNMLPAdapterNet",
    "GINEMLPAdapterNet",
    "GINEHeatNet",
    "GINENet",
    "GINMLPAdapterNet",
    "GINNet",
    "GPSNet",
    "GraphSAGENet",
    "GraphormerNet",
    "MixHopNet",
    "MPNNMLPAdapterNet",
    "MPNNNet",
    "PNANet",
]

"""Flows module for batch processing configurations."""

import importlib
from typing import Any

KNOWN_FLOWS = ["lg_concrete_v1", "lg_concrete_v1_test"]


def _discover_flows() -> dict[str, Any]:
    """Discover flows from the registry."""
    flows = {}

    # Registry of known flow modules - update this when adding new flows
    # Note: Test flows (ending with '_test') are excluded from this list

    for module_name in KNOWN_FLOWS:
        try:
            module = importlib.import_module(f"bubbola.flows.{module_name}")
            if hasattr(module, "flow"):
                module.flow.update({"name": module_name})
                flows[module_name] = module.flow
        except (ImportError, AttributeError):
            continue

    return flows


# Auto-discover all flows
_flows_dict = _discover_flows()

# Make flows available as module attributes
for name, flow in _flows_dict.items():
    globals()[name] = flow

# Export the dictionary and all flow names
__all__ = list(_flows_dict.keys()) + ["_flows_dict", "get_flows"]


def get_flows() -> dict[str, Any]:
    """Get all discovered flows as a dictionary.

    Returns:
        Dictionary mapping module names to their flow configurations
    """
    return _flows_dict.copy()

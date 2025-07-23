"""Flows module for batch processing configurations."""

from typing import Any

from bubbola.flows import small_parsing

# Explicit imports for PyInstaller compatibility
try:
    from bubbola.flows import (
        fattura_check_v1,
        lg_concrete_v1,
        lg_concrete_v1_test,
    )
except ImportError:
    # Fallback for development
    pass

# Registry of known flow modules
KNOWN_FLOWS = [
    "lg_concrete_v1",
    "lg_concrete_v1_test",
    "small_test",
    "fattura_check_v1",
]


def _discover_flows() -> dict[str, Any]:
    """Discover flows from the registry."""
    flows = {}

    # Try explicit imports first
    flow_modules = {
        "lg_concrete_v1": lg_concrete_v1,
        "lg_concrete_v1_test": lg_concrete_v1_test,
        "small_test": small_parsing,
        "fattura_check_v1": fattura_check_v1,
    }

    for module_name, module in flow_modules.items():
        try:
            if hasattr(module, "flow"):
                module.flow.update({"name": module_name})
                flows[module_name] = module.flow
        except (AttributeError, Exception):
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

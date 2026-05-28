# Mock for utils.pyx
"""Utility data structures for MPM connectivity management."""


class BoundingBox:
    """Fallback bounding-box container used when the compiled utilities are unavailable."""

    def __init__(self, *args, **kwargs):
        """Initialize the fallback bounding-box placeholder."""
        raise NotImplementedError("utils not available in this environment")


class BoundedCell:
    """Fallback bounded-cell container used when the compiled utilities are unavailable."""

    def __init__(self, *args, **kwargs):
        """Initialize the fallback bounded-cell placeholder."""
        raise NotImplementedError("utils not available in this environment")


class _KDTreeImpl:
    """Fallback KD-tree implementation placeholder used when compiled utilities are unavailable."""

    def __init__(self, *args, **kwargs):
        """Initialize the fallback KD-tree implementation placeholder."""
        raise NotImplementedError("utils not available in this environment")


class KDTree:
    """Fallback KD-tree wrapper used when the compiled utilities are unavailable."""

    def __init__(self, *args, **kwargs):
        """Initialize the fallback KD-tree placeholder."""
        raise NotImplementedError("utils not available in this environment")

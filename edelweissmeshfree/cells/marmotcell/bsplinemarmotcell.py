"""Mock stub for the Cython-compiled BSplineMarmotCellWrapper extension module.

The real implementation is compiled from ``bsplinemarmotcell.pyx``. This stub
is provided so that the package can be imported in environments where the
compiled extension is not available.
"""


# Mock for bsplinemarmotcell.pyx
class BSplineMarmotCellWrapper(MarmotCellWrapper):
    """Wrapper around a B-spline Marmot cell Cython extension.

    In production, this class extends :class:`MarmotCellWrapper` to provide
    B-spline-based shape functions within a Marmot MPM grid cell.
    This stub raises :exc:`NotImplementedError` when instantiated in
    environments without the compiled Marmot extension.
    """

    def __init__(self, *args, **kwargs):
        """This is a stub — raises :exc:`NotImplementedError` in this environment."""
        raise NotImplementedError("bsplinemarmotcell not available in this environment")

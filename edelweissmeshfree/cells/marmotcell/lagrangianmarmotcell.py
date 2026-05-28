"""Mock stub for the Cython-compiled LagrangianMarmotCellWrapper extension module.

The real implementation is compiled from ``lagrangianmarmotcell.pyx``. This stub
is provided so that the package can be imported in environments where the
compiled extension is not available.
"""


# Mock for lagrangianmarmotcell.pyx
class LagrangianMarmotCellWrapper(MarmotCellWrapper):
    """Wrapper around a Lagrangian Marmot cell Cython extension.

    In production, this class extends :class:`MarmotCellWrapper` to provide
    Lagrangian (standard FE) shape functions within a Marmot MPM grid cell.
    This stub raises :exc:`NotImplementedError` when instantiated in
    environments without the compiled Marmot extension.
    """

    def __init__(self, *args, **kwargs):
        """This is a stub — raises :exc:`NotImplementedError` in this environment."""
        raise NotImplementedError("lagrangianmarmotcell not available in this environment")

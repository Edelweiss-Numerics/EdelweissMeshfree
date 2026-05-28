"""Mock stub for the Cython-compiled LagrangianMarmotCellElementWrapper extension module.

The real implementation is compiled from ``lagrangianmarmotcellelement.pyx``. This stub
is provided so that the package can be imported in environments where the
compiled extension is not available.
"""


# Mock for lagrangianmarmotcellelement.pyx
class LagrangianMarmotCellElementWrapper(MarmotCellElementWrapper):
    """Wrapper around a Lagrangian Marmot cell element Cython extension.

    In production, this class extends :class:`MarmotCellElementWrapper` to
    provide Lagrangian (standard FE) shape functions for enriched MPM grid
    cells. This stub raises :exc:`NotImplementedError` when instantiated in
    environments without the compiled Marmot extension.
    """

    def __init__(self, *args, **kwargs):
        """This is a stub — raises :exc:`NotImplementedError` in this environment."""
        raise NotImplementedError("lagrangianmarmotcellelement not available in this environment")

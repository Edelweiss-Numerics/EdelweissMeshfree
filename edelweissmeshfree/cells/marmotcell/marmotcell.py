"""Mock stub for the Cython-compiled MarmotCellWrapper extension module.

The real implementation is compiled from ``marmotcell.pyx``. This stub is
provided so that the package can be imported in environments where the
compiled extension is not available.
"""


# Mock for marmotcell.pyx
class MarmotCellWrapper:
    """Wrapper around a Marmot cell Cython extension.

    In production, this class wraps a Cython-backed Marmot cell that
    implements the :class:`~edelweissmeshfree.cells.base.cell.CellBase`
    interface. This stub raises :exc:`NotImplementedError` when instantiated
    in environments without the compiled Marmot extension.
    """

    def __init__(self, *args, **kwargs):
        """This is a stub — raises :exc:`NotImplementedError` in this environment."""
        raise NotImplementedError("marmotcell not available in this environment")

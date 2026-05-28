"""Mock stub for the Cython-compiled MarmotCellElementWrapper extension module.

The real implementation is compiled from ``marmotcellelement.pyx``. This stub
is provided so that the package can be imported in environments where the
compiled extension is not available.
"""


# Mock for marmotcellelement.pyx
class MarmotCellElementWrapper(MarmotCellWrapper):
    """Wrapper around a Marmot cell element Cython extension.

    In production, this class wraps a Cython-backed Marmot cell element that
    provides enriched cell formulations for the MPM grid. This stub raises
    :exc:`NotImplementedError` when instantiated in environments without the
    compiled Marmot extension.
    """

    def __init__(self, *args, **kwargs):
        """This is a stub — raises :exc:`NotImplementedError` in this environment."""
        raise NotImplementedError('marmotcellelement not available in this environment')

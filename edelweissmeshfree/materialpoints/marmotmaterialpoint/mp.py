"""Mock stub for the Cython-compiled MarmotMaterialPointWrapper extension module.

The real implementation is compiled from ``mp.pyx``. This stub is provided so
that the package can be imported in environments where the compiled Marmot
extension is not available.
"""


# Mock for mp.pyx
class MarmotMaterialPointWrapper:
    """Wrapper around a Marmot material point Cython extension.

    In production, this class wraps a Cython-backed Marmot material point that
    implements the :class:`~edelweissmeshfree.materialpoints.base.mp.MaterialPointBase`
    interface and delegates constitutive computations to the Marmot library.
    This stub raises :exc:`NotImplementedError` when instantiated in
    environments without the compiled Marmot extension.
    """

    def __init__(self, *args, **kwargs):
        """This is a stub — raises :exc:`NotImplementedError` in this environment."""
        raise NotImplementedError('mp not available in this environment')

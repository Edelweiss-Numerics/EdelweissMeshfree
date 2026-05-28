"""Mock stub for the Cython-compiled MarmotMeshfreeApproximationWrapper extension module.

The real implementation is compiled from ``marmotmeshfreeapproximation.pyx``.
This stub is provided so that the package can be imported in environments where
the compiled Marmot extension is not available.
"""


# Mock for marmotmeshfreeapproximation.pyx
class MarmotMeshfreeApproximationWrapper:
    """Wrapper around a Marmot meshfree approximation Cython extension.

    In production, this class wraps a Cython-backed Marmot meshfree
    approximation scheme (e.g., RKPM) that implements the
    :class:`~edelweissmeshfree.meshfree.approximations.base.basemeshfreeapproximation.BaseMeshfreeApproximation`
    interface and delegates shape function evaluations to the Marmot library.
    This stub raises :exc:`NotImplementedError` when instantiated in
    environments without the compiled Marmot extension.
    """

    def __init__(self, *args, **kwargs):
        """This is a stub — raises :exc:`NotImplementedError` in this environment."""
        raise NotImplementedError("marmotmeshfreeapproximation not available in this environment")

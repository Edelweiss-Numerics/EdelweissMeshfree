"""Mock stub for the Cython-compiled MarmotMeshfreeKernelFunctionWrapper extension module.

The real implementation is compiled from ``marmotmeshfreekernelfunction.pyx``.
This stub is provided so that the package can be imported in environments where
the compiled Marmot extension is not available.
"""


# Mock for marmotmeshfreekernelfunction.pyx
class MarmotMeshfreeKernelFunctionWrapper:
    """Wrapper around a Marmot meshfree kernel function Cython extension.

    In production, this class wraps a Cython-backed Marmot kernel function
    that implements the
    :class:`~edelweissmeshfree.meshfree.kernelfunctions.base.basemeshfreekernelfunction.BaseMeshfreeKernelFunction`
    interface and delegates kernel evaluations to the Marmot library.
    This stub raises :exc:`NotImplementedError` when instantiated in
    environments without the compiled Marmot extension.
    """

    def __init__(self, *args, **kwargs):
        """This is a stub — raises :exc:`NotImplementedError` in this environment."""
        raise NotImplementedError("marmotmeshfreekernelfunction not available in this environment")

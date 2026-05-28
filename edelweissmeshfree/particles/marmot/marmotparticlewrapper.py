"""Mock stub for the Cython-compiled MarmotParticleWrapper extension module.

The real implementation wraps a Marmot particle that provides kernel function
evaluations and quadrature for meshfree methods (RKPM, etc.). This stub is
provided so that the package can be imported in environments where the
compiled Marmot extension is not available.
"""


class MarmotParticleWrapper:
    """Wrapper around a Marmot particle Cython extension.

    In production, this class wraps a Cython-backed Marmot particle that
    implements the :class:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle`
    interface and delegates kernel function computations to the Marmot library.
    This stub raises :exc:`NotImplementedError` when instantiated in
    environments without the compiled Marmot extension.
    """

    def __init__(self, *args, **kwargs):
        """This is a stub — raises :exc:`NotImplementedError` in this environment."""
        raise NotImplementedError("Marmot not available in this environment")

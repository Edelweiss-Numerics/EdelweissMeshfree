"""The implicit solvers EdelweissMeshfree had before EdelweissFE's implicit solvers solved its models.

.. deprecated::
    An :class:`~edelweissmeshfree.models.mpmmodel.MPMModel` is solved by EdelweissFE's implicit static solvers
    :class:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST`,
    :class:`~edelweissfe.solvers.nonlinearimplicitstaticparallel.NISTParallel` and
    :class:`~edelweissfe.solvers.nonlinearimplicitstaticparallelarclength.NISTPArcLength`; see the documentation page
    *Choosing a solver*.

This package keeps the old solvers, so that the examples which still use them keep working:

- :class:`~edelweissmeshfree.solvers.deprecated.nqs.NonlinearQuasistaticSolver` (replaced by ``NIST`` and
  ``NISTParallel``),
- :class:`~edelweissmeshfree.solvers.deprecated.nqsmparclength.NonlinearQuasistaticMarmotArcLengthSolver` (replaced
  by ``NISTPArcLength``),
- their common base :mod:`~edelweissmeshfree.solvers.deprecated.implicitbase`, and the predictors they take
  (:mod:`~edelweissmeshfree.solvers.deprecated.predictors`; EdelweissFE's solvers have the option ``extrapolation``).

They are frozen: they get no new features, only the adaptations that later changes of the interfaces they call force.
Constructing one emits a :class:`DeprecationWarning` that names its replacement. The package is deleted once no
example needs it any more.
"""

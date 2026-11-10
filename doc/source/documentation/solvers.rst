Solvers
=======

EdelweissFE's implicit solvers
------------------------------

An ``MPMModel`` answers what EdelweissFE's implicit solvers ask a model (see the models), so ``NIST``, ``NISTParallel``
and ``NISTPArcLength`` solve it as they solve a finite element model, built from Python as any EdelweissFE step; see
*Choosing a solver*. Their convergence criterion with the settings of EdelweissMeshfree's former solvers is a parameter
set of EdelweissFE's legacy criterion:

.. automodule:: edelweissmeshfree.solvers.meshfreeconvergencecriterion
   :members:

The explicit solver
-------------------

.. automodule:: edelweissmeshfree.solvers.meshfreeexplicitdynamic
   :members:

Deprecated implicit solvers
---------------------------

.. automodule:: edelweissmeshfree.solvers.deprecated

``NonlinearQuasistaticSolver`` class
************************************

.. automodule:: edelweissmeshfree.solvers.deprecated.nqs
   :private-members:
   :members:

``NonlinearQuasistaticMarmotArcLengthSolver`` class
***************************************************

.. automodule:: edelweissmeshfree.solvers.deprecated.nqsmparclength
   :private-members:
   :members:

Predictors
**********

The predictors make the initial guess of the Newton-Raphson scheme of the deprecated solvers.

.. automodule:: edelweissmeshfree.solvers.deprecated.predictors.basepredictor
   :members:
   :private-members:

.. automodule:: edelweissmeshfree.solvers.deprecated.predictors.linearpredictor
   :members:
   :private-members:

.. automodule:: edelweissmeshfree.solvers.deprecated.predictors.quadraticpredictor
   :members:
   :private-members:

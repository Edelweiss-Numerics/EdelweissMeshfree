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

EdelweissFE's explicit solver
-----------------------------

An RKPM model, alone or with finite elements, is integrated in explicit dynamics by EdelweissFE's ``NED`` and
``NEDParallel``, which advance its particles, reconnect them and evaluate them on the new kernel functions in every
increment (see EdelweissFE's documentation of ``NED``, "Entities that carry their own inertia", and *Choosing a solver*).

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

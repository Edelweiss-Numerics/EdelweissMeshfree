Solvers
=======

EdelweissFE's implicit solvers
------------------------------

An ``MPMModel`` answers what EdelweissFE's implicit solvers ask a model (see the models), so ``NIST``, ``NISTParallel``
and ``NISTPArcLength`` solve it as they solve a finite element model, built from Python as any EdelweissFE step. The
convergence criterion of the solvers below is a parameter set of EdelweissFE's legacy criterion:

.. automodule:: edelweissmeshfree.solvers.meshfreeconvergencecriterion
   :members:


``NonlinearQuasistaticSolver`` class
------------------------------------

.. automodule:: edelweissmeshfree.solvers.nqs
   :private-members:
   :members:

``NQSParallelForMarmot`` class
------------------------------

.. automodule:: edelweissmeshfree.solvers.nqsmarmotparallel
   :private-members:
   :members:


``NonlinearQuasistaticMarmotParallelArcLengthSolver`` class
-----------------------------------------------------------

.. automodule:: edelweissmeshfree.solvers.nqsmparclength
   :private-members:
   :members:

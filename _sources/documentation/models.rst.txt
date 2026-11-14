Models
======

Model are the core of the MPM simulation. They hold information about the
domain (cells, material points), the materials, the sets, etc.

``MPMModel`` class
------------------

.. automodule:: edelweissmeshfree.models.mpmmodel
   :members:
   :private-members:
   :inherited-members:

``MPMDiscreteSystem`` class
---------------------------------

What an ``MPMModel`` answers EdelweissFE's implicit solvers (``NIST``, ``NISTParallel``, ``NISTPArcLength``) and the
explicit solver ``MeshfreeExplicitDynamic`` when they (re)build their discrete system: the active domain, its degrees of freedom and the entities computed like elements, how
the solution passes between the model and the solver's vectors, and the update of the material points before the
cells are computed. See EdelweissFE's documentation, "What the solver asks the model".

.. automodule:: edelweissmeshfree.models.mpmdiscretesystem
   :members:

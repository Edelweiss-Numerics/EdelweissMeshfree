Numerical tools
===============

The numerical tools are a set of tools that can be used to solve a variety of
problems.

``MPMDofLocations`` class
-------------------------

The locations of the entities of a material point or RKPM model: the global indices of the local degrees of freedom of
its finite elements and constraints (:class:`~edelweissfe.numerics.doflocations.DofLocations`, the *LM* array), and of
its background cells, cell elements and particles. They are built on the numbering of the degrees of freedom of the
active domain (:class:`~edelweissfe.numerics.dofnumbering.DofNumbering`, the *ID* array). Neither is ever changed: moved
particles, or changed constraints, make new locations on the same numbering
(:meth:`~edelweissmeshfree.numerics.doflocations.DofLocations.withNewLocationsOf`).

.. automodule:: edelweissmeshfree.numerics.doflocations
   :members:
   :private-members:
   :inherited-members:

Meshfree utilities
==================

Approximation base
------------------

.. automodule:: edelweissmeshfree.meshfree.approximations.base.basemeshfreeapproximation
   :members:
   :private-members:

Kernel function base
--------------------

.. automodule:: edelweissmeshfree.meshfree.kernelfunctions.base.basemeshfreekernelfunction
   :members:
   :private-members:

Marmot wrappers
---------------

Limit on the condition of the moment matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A particle that is stretched far beyond its neighbours -- torn, with its damage close to one -- can reach with a face
into a void, a borehole or a crack, where only the kernels of one wall of particles cover it. These kernels lie nearly
in one plane, on one side of the point, and reproducing a linear field there takes shape functions of the size of the
distance to the wall over the scatter of the kernel centres about it. The particle's smoothed gradients and its
stiffness grow with them, and so does its bound of the stable time increment of an explicit analysis: one such
particle can set the time increment of the whole model.

``MarmotMeshfreeApproximationWrapper`` takes the optional argument ``momentMatrixConditionLimit``: a point whose
moment matrix (scaled to a unit diagonal) has a larger condition number is evaluated with the completeness order 0,
the Shepard functions, which stay between 0 and 1. The default ``inf`` sets no limit. With a limit, every point below
it is evaluated bit for bit as without one, so an intact model is unchanged; a limit between ``1e4`` and ``1e6``
separates the points of intact particles from those of a torn one (see Marmot's documentation of the meshfree
approximation for the measure and the values it takes). The limit acts on the shape functions, not on the material:
a torn particle keeps its mass and its stiffness, it is not eroded.

.. code-block:: python

    approximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernel", 3, completenessOrder=1, momentMatrixConditionLimit=1e5
    )

.. automodule:: edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation
   :members:
   :private-members:

.. automodule:: edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction
   :members:
   :private-members:

Integration helpers
-------------------

.. automodule:: edelweissmeshfree.meshfree.particlekerneldomain
   :members:
   :private-members:

.. automodule:: edelweissmeshfree.meshfree.vci
   :members:
   :private-members:

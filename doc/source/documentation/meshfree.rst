Meshfree
========

The meshfree module contains the core ingredients for meshfree methods such as
RKPM (Reproducing Kernel Particle Method) and MLS (Moving Least Squares):
kernel functions and approximation schemes.

Kernel functions
----------------

Kernel functions define the support and shape of individual basis functions
in meshfree approximations.

``BaseMeshfreeKernelFunction`` class
*************************************

.. automodule:: edelweissmeshfree.meshfree.kernelfunctions.base.basemeshfreekernelfunction
   :members:
   :private-members:

``MarmotMeshfreeKernelFunction`` class
***************************************

.. automodule:: edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction
   :members:
   :private-members:

Approximations
--------------

Approximation schemes compute shape function values and derivatives at arbitrary
points in the domain from a set of kernel functions.

``BaseMeshfreeApproximation`` class
************************************

.. automodule:: edelweissmeshfree.meshfree.approximations.base.basemeshfreeapproximation
   :members:
   :private-members:

``MarmotMeshfreeApproximation`` class
***************************************

.. automodule:: edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation
   :members:
   :private-members:

Particle kernel domain
----------------------

.. automodule:: edelweissmeshfree.meshfree.particlekerneldomain
   :members:
   :private-members:

Variational consistency improvement
-------------------------------------

.. automodule:: edelweissmeshfree.meshfree.vci
   :members:
   :private-members:

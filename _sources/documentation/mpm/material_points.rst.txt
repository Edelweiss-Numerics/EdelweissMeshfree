Material Points
===============

.. automodule:: edelweissmeshfree.materialpoints.base.mp
   :members:
   :private-members:

``MarmotMaterialPoint`` class
-----------------------------

.. automodule:: edelweissmeshfree.materialpoints.marmotmaterialpoint.mp
   :members:
   :private-members:

``PythonMaterialPoint`` class
-----------------------------

.. automodule:: edelweissmeshfree.materialpoints.pythonmaterialpoint.mp
   :members:
   :private-members:

Theory (small-strain linear elasticity in plane strain)
-------------------------------------------------------

The Python material point stores kinematics and stress in Voigt form:

.. math::

   \boldsymbol{\varepsilon} =
   \begin{bmatrix}
   \varepsilon_{xx} & \varepsilon_{yy} & \varepsilon_{zz} & \gamma_{xy}
   \end{bmatrix}^\mathsf{T},
   \qquad
   \boldsymbol{\sigma} =
   \begin{bmatrix}
   \sigma_{xx} & \sigma_{yy} & \sigma_{zz} & \sigma_{xy}
   \end{bmatrix}^\mathsf{T}.

For each increment, the temporary state is updated as:

.. math::

   \boldsymbol{\varepsilon}^{n+1} = \boldsymbol{\varepsilon}^{n} + \Delta\boldsymbol{\varepsilon},
   \qquad
   \boldsymbol{\sigma}^{n+1} = \mathbf{C}\,\boldsymbol{\varepsilon}^{n+1}.

The plane-strain elastic tangent for isotropic material parameters
Young's modulus :math:`E` and Poisson ratio :math:`\nu` is:

.. math::

   \mathbf{C} = \frac{E}{(1+\nu)(1-2\nu)}
   \begin{bmatrix}
   1-\nu & \nu & \nu & 0 \\
   \nu & 1-\nu & \nu & 0 \\
   \nu & \nu & 1-\nu & 0 \\
   0 & 0 & 0 & \frac{1-2\nu}{2}
   \end{bmatrix}.

After convergence, the temporary state is accepted as persistent state for the next increment.

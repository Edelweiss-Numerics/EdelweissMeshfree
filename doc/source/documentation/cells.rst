Cells
=====

.. automodule:: edelweissmeshfree.cells.base.cell
   :members:
   :private-members:

``MarmotCell`` class
--------------------

.. automodule:: edelweissmeshfree.cells.marmotcell.marmotcell
   :members:
   :private-members:

``LagrangianMarmotCell`` class
------------------------------

.. automodule:: edelweissmeshfree.cells.marmotcell.lagrangianmarmotcell
   :members:
   :private-members:

``BSplineMarmotCell`` class
------------------------------

.. automodule:: edelweissmeshfree.cells.marmotcell.bsplinemarmotcell
   :members:
   :private-members:

``PythonCell`` class
--------------------

.. automodule:: edelweissmeshfree.cells.pythoncell.cell
   :members:
   :private-members:

Theory (Quad4 interpolation and cell kernels)
---------------------------------------------

The Python cell implementation uses a bilinear Quad4 interpolation in the parent space
:math:`(\xi,\eta)\in[-1,1]^2`:

.. math::

   N_1 = \frac{1}{4}(1-\xi)(1-\eta),\quad
   N_2 = \frac{1}{4}(1+\xi)(1-\eta),\quad
   N_3 = \frac{1}{4}(1+\xi)(1+\eta),\quad
   N_4 = \frac{1}{4}(1-\xi)(1+\eta).

With nodal displacement vector
:math:`\mathbf{u}_n=[u_{1x},u_{1y},u_{2x},u_{2y},u_{3x},u_{3y},u_{4x},u_{4y}]^\mathsf{T}`,
the material point displacement increment is interpolated as:

.. math::

   \Delta\mathbf{u}_{mp} = \sum_{a=1}^{4} N_a\,\Delta\mathbf{u}_a.

The strain increment follows from the standard strain-displacement matrix:

.. math::

   \Delta\boldsymbol{\varepsilon}_{mp} = \mathbf{B}\,\Delta\mathbf{u}_n,

where :math:`\mathbf{B}` is assembled from the spatial gradients
:math:`\nabla N_a` (obtained using the Jacobian mapping from parent to physical space).

Cell residual and tangent contributions are assembled from material point stress
:math:`\boldsymbol{\sigma}` and consistent tangent :math:`\mathbf{C}`:

.. math::

   \mathbf{P}_{int} \mathrel{+}= \mathbf{B}^\mathsf{T}\boldsymbol{\sigma}\,V_{mp},
   \qquad
   \mathbf{K} \mathrel{+}= \mathbf{B}^\mathsf{T}\mathbf{C}\mathbf{B}\,V_{mp}.

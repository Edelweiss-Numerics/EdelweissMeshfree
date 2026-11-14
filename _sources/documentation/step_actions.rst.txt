Step actions
============

Step actions represent the actions that can be performed during a simulation step,
such as the enforcement of boundary conditions, the application of external forces.

A load is an EdelweissFE load (:class:`~edelweissfe.stepactions.base.loadbase.LoadBase`): it assembles its own
residual contribution, the *negative* of the external load it applies, and the tangent of that contribution in
``assembleLoad(P, K, U_np, dU, timeStep, locations)``, wherever it acts -- on the cells of the background grid, on
the cells of the material points that carry it, or on the faces of particles. A solver therefore only loops over the
loads. The loads here derive from one of three bases, which implement ``assembleLoad`` once for all loads of their
kind: :class:`~edelweissmeshfree.stepactions.base.mpmbodyloadbase.MPMBodyLoadBase`,
:class:`~edelweissmeshfree.stepactions.base.mpmdistributedloadbase.MPMDistributedLoadBase` and
:class:`~edelweissmeshfree.stepactions.base.particledistributedloadbase.ParticleDistributedLoadBase`.

Body loads
----------

``MPMBodyLoadBase`` class
*************************

.. automodule:: edelweissmeshfree.stepactions.base.mpmbodyloadbase
   :members:
   :private-members:

``BodyLoad`` class
******************

.. automodule:: edelweissmeshfree.stepactions.bodyload
   :members:
   :private-members:

Neumann boundary conditions
---------------------------

``MPMDistributedLoadBase`` class
********************************

.. automodule:: edelweissmeshfree.stepactions.base.mpmdistributedloadbase
   :members:
   :private-members:

``DistributedLoad`` class
*************************

.. automodule:: edelweissmeshfree.stepactions.distributedload
   :members:
   :private-members:

``ParticleDistributedLoadBase`` class
*************************************

.. automodule:: edelweissmeshfree.stepactions.base.particledistributedloadbase
   :members:
   :private-members:

``ParticleDistributedLoad`` class
*********************************

.. automodule:: edelweissmeshfree.stepactions.particledistributedload
   :members:
   :private-members:

Dirichlet boundary conditions
-----------------------------

A Dirichlet boundary condition on the nodes of a node set is EdelweissFE's
:class:`~edelweissfe.stepactions.dirichlet.StepAction`, constructed from Python with the node set, the field and the
prescribed components. A solver prescribes the nodes of the set that lie in the active domain (see
:meth:`~edelweissfe.numerics.dofnumbering.DofNumbering.locateFieldOnNodeSet`).

``ParticleNitscheDirichlet`` class
**********************************

A weak Dirichlet boundary condition by Nitsche's method, which the particles integrate over their faces like a
distributed load.

.. automodule:: edelweissmeshfree.stepactions.particlenitschedirichlet
   :members:
   :private-members:

Arclength control
-----------------

The arc length controllers are EdelweissFE's
(:class:`~edelweissfe.stepactions.base.arclengthcontrollerbase.ArcLengthControllerBase`), so that EdelweissFE's arc
length solver ``NISTPArcLength`` takes them by their name.

``Indirect control`` class
**************************

.. automodule:: edelweissmeshfree.stepactions.indirectcontrol
   :members:
   :private-members:

``Particle indirect control`` class
***********************************

.. automodule:: edelweissmeshfree.stepactions.particleindirectcontrol
   :members:
   :private-members:

Constraints
===========

.. _residual-and-tangent-convention:

The residual and the tangent
----------------------------

A constraint of EdelweissMeshfree is an EdelweissFE constraint, and enters the equation system by EdelweissFE's rule
(see the EdelweissFE solver documentation, *The residual and the tangent*): its ``computeKernels(U_np, dU, P, K,
timeStep)`` adds the residual contribution :math:`\mathbf{P}` -- the *negative* of the force the constraint exerts on
its nodes, what an element of the same mechanics would add as its internal force -- and its tangent
:math:`\mathbf{K} = \partial \mathbf{P} / \partial \mathbf{U}`. A penalty constraint that pulls the displacement
:math:`u = \mathbf{N} \cdot \Delta\mathbf{U}` of a particle towards :math:`\bar{u}` adds

.. math::

    \mathbf{P} \mathrel{+}= k\, \mathbf{N}\, (\mathbf{N} \cdot \Delta\mathbf{U} - \bar{u}), \qquad
    \mathbf{K} \mathrel{+}= k\, \mathbf{N} \otimes \mathbf{N} .

The nodal values of the particles and of the cells are increments on the current configuration, so these constraints
read the increment ``dU``. Their nodes are those the particles or material points currently interpolate from: a
constraint finds them anew at the start of every increment, in ``updateConnectivity``, which may also find the
constraint out of reach (``withinReach``), e.g. a contact constraint whose particle is far from the boundary. A
constraint takes part in the equation system (``active``) while it is within reach, unless it is switched off from
outside: setting ``active`` to False, e.g. by a ``modelupdate`` step action, switches a constraint off whatever it
finds in reach, until it is switched on again. These are the rules of EdelweissFE's constraint base class, see the
EdelweissFE constraint documentation.

Because :math:`\mathbf{K}` is the derivative of exactly what is added to :math:`\mathbf{P}`, each constraint is
checked by central differences of its own ``computeKernels`` in
``edelweissmeshfree/constraints/test/residual_tangent_test.py``, with EdelweissFE's check
(:mod:`edelweissfe.utils.residualtangentcheck`); add a case there for a new one.

A constraint written for explicit analyses alone, with forces and without a tangent, derives from
:class:`~edelweissmeshfree.constraints.base.mpmexplicitconstraintbase.MPMExplicitConstraintBase` and implements
``computeKernelsExplicit(U_np, dU, P, timeStep)``, with the same sign: an implicit solver refuses it.

The base classes
----------------

.. automodule:: edelweissmeshfree.constraints.base.mpmconstraintbase
   :members:
   :private-members:

.. automodule:: edelweissmeshfree.constraints.base.mpmexplicitconstraintbase
   :members:
   :private-members:

``PenaltyEqualValue`` constraint class
--------------------------------------

.. automodule:: edelweissmeshfree.constraints.penaltyequalvalue
   :members:
   :private-members:

``PenaltyConstrainMP2Node`` constraint class
--------------------------------------------

.. automodule:: edelweissmeshfree.constraints.penaltyconstrainmp2node
   :members:
   :private-members:

``PenaltyWeakDirichlet`` constraint class
-----------------------------------------

.. automodule:: edelweissmeshfree.constraints.penaltyweakdirichlet
   :members:
   :private-members:


``ParticlePenaltyWeakDirichlet`` constraint class
-------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlepenaltyweakdirichtlet
   :members:
   :private-members:

``ParticlePenaltyEqualValue`` constraint class
----------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlepenaltyequalvalue
   :members:
   :private-members:

``ParticlePenaltyCartesianBoundary`` constraint class
------------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlepenaltycartesianboundary
   :members:
   :private-members:

``ParticlePenaltyRigidBodyContact`` constraint class
----------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlepenaltyrigidbodycontact
   :members:
   :private-members:

``ParticleLagrangianEqualValue`` constraint class
--------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlelagrangianequalvalue
   :members:
   :private-members:

``ParticleLagrangianWeakDirichlet`` constraint class
-----------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlelagrangianweakdirichlet
   :members:
   :private-members:

``ParticleLagrangianCartesianBoundary`` constraint class
---------------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlelagrangiancartesianboundary
   :members:
   :private-members:

``ParticleMortarWeakDirichlet`` constraint class
-------------------------------------------------

.. automodule:: edelweissmeshfree.constraints.particlemortarweakdirichlet
   :members:
   :private-members:

Constraints for explicit analyses
---------------------------------

``ParticlePenaltyCartesianBoundaryConstraintExplicit`` constraint class
***********************************************************************

.. automodule:: edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit
   :members:
   :private-members:

``ParticlePenaltyContactImplicitSurfaceConstraintExplicit`` constraint class
****************************************************************************

.. automodule:: edelweissmeshfree.constraints.explicit.particlepenaltyrigidbodycontactexplicit
   :members:
   :private-members:

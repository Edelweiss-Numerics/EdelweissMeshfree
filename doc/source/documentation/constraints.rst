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
finds in reach, until it is switched on again.

A constraint with Lagrange multipliers adds them as scalar variables of the model. The condition :math:`g` it enforces
is one on increments, like the nodal values; the multiplier :math:`\lambda` is the force or traction that balances the
total stresses of the particles or elements, so it is a **total**. A multiplier that prescribes the increment
:math:`\Delta\bar{u}` of a particle's displacement :math:`u = \mathbf{N} \cdot \Delta\mathbf{U}` adds

.. math::

    \mathbf{P} \mathrel{+}= \lambda\, \mathbf{N}, \qquad
    P_{\lambda} \mathrel{+}= \mathbf{N} \cdot \Delta\mathbf{U} - \Delta\bar{u}, \qquad
    \mathbf{K}_{U\lambda} \mathrel{+}= \mathbf{N}, \qquad
    \mathbf{K}_{\lambda U} \mathrel{+}= \mathbf{N}^T,

with :math:`\lambda` read from ``U_np``: the solver continues it from the last converged increment, also across a
rebuilt equation system, as the model keeps the converged multipliers in its scalar variables, and its checkpoint keeps
them for a restart. The residual is linear in :math:`\lambda`, so where Newton's method starts :math:`\lambda` changes
only its first residual, not the iterates after the first correction. The deprecated solvers
(:mod:`edelweissmeshfree.solvers.deprecated`) continue the multipliers only while the equation system is kept, restart
them at zero after a rebuild, and do not check the residual of the multipliers for convergence.

Because :math:`\mathbf{K}` is the derivative of exactly what is added to :math:`\mathbf{P}`, each constraint is
checked by central differences of its own ``computeKernels`` in
``edelweissmeshfree/constraints/test/residual_tangent_test.py``, with EdelweissFE's check
(:mod:`edelweissfe.utils.residualtangentcheck`); add a case there for a new one.

A constraint written for explicit analyses alone, with forces and without a tangent, derives from
:class:`~edelweissmeshfree.constraints.base.mpmexplicitconstraintbase.MPMExplicitConstraintBase` and implements
``computeKernelsExplicit(U_np, dU, P, timeStep)``, with the same sign: an implicit solver refuses it. The explicit
solver, in turn, accepts only a constraint that acts in an explicit analysis (``actsInExplicitAnalysis``): one that
measures its condition at the particles' current positions, as the penalty contacts do, explicit or not. A constraint on
the increment, such as a weak Dirichlet condition, would act as if nothing had moved there.

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

``ParticleFiniteElementMortarTie`` constraint class
---------------------------------------------------

Ties an RKPM domain to a finite element domain, see :doc:`coupled_rkpm_fe`. Its tangent is checked by central
differences in ``edelweissmeshfree/constraints/test/mortar_tie_test.py``, together with the integrals it stands for.

.. automodule:: edelweissmeshfree.constraints.particlefiniteelementmortartie
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

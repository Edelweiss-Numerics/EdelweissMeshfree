Constraints
===========

.. _residual-and-tangent-convention:

The residual and the tangent
----------------------------

A constraint of EdelweissMeshfree is an EdelweissFE constraint, and enters the discrete system by EdelweissFE's rule
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
constraint takes part in the discrete system (``active``) while it is within reach, unless it is switched off from
outside: setting ``active`` to False, e.g. by a ``modelupdate`` step action, switches a constraint off whatever it
finds in reach, until it is switched on again. Switched off from outside, it rests: the model does not locate it anew,
and it keeps what it found and computed last, its reaction force included. These are the rules of EdelweissFE's
constraint base class, see the EdelweissFE constraint documentation.

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
rebuilt discrete system, as the model keeps the converged multipliers in its scalar variables, and its checkpoint keeps
them for a restart. The residual is linear in :math:`\lambda`, so where Newton's method starts :math:`\lambda` changes
only its first residual, not the iterates after the first correction. The deprecated solvers
(:mod:`edelweissmeshfree.solvers.deprecated`) continue the multipliers only while the discrete system is kept, restart
them at zero after a rebuild, and do not check the residual of the multipliers for convergence.

Because :math:`\mathbf{K}` is the derivative of exactly what is added to :math:`\mathbf{P}`, each constraint is
checked by central differences of its own ``computeKernels`` in
``edelweissmeshfree/constraints/test/residual_tangent_test.py``, with EdelweissFE's check
(:mod:`edelweissfe.utils.residualtangentcheck`); add a case there for a new one.

A constraint written for explicit analyses alone, with forces and without a tangent, derives from
:class:`~edelweissmeshfree.constraints.base.mpmexplicitconstraintbase.MPMExplicitConstraintBase` and implements
``computeKernelsExplicit(U_np, dU, P, timeStep)``, with the same sign, and
``addBoundOfAbsoluteStiffnessTimes(U_np, dU, weights, bound, timeStep, nextIncrement=None)``, the bound of its
stiffness for the critical time step of ``NED`` (from its springs, or nothing, saying so), with ``nextIncrement`` over
the states the next increment can reach (``follow-critical-time-step``; the springs at the smallest gap of each point
within it): an implicit solver refuses it. The explicit
solver, in turn, accepts only a constraint that acts in an explicit analysis (``actsInExplicitAnalysis``): one that
measures its condition at the particles' current positions, as the penalty contacts do, explicit or not, and as the
penalty weak Dirichlet condition does where it holds the displacement at zero (a support). A constraint on the
increment, such as a weak Dirichlet condition that prescribes a change, would act as if nothing had moved there.

**Points of a particle.** A particle knows every field only by its value and its gradient at its center, so its own
interpolation at a point of its own -- a vertex, a face center, its center -- is its shape functions linearized about
the center, :math:`\mathbf{G}`
(:meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.getInterpolationVectorLinearizedAboutCenter`): for
the displacement, the motion of the point. The shape functions at the point, :math:`\mathbf{N}`
(:meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.getInterpolationVector`), are exact at an arbitrary
location and differ from :math:`\mathbf{G}` away from the center. The constraints that act on points of particles (the
penalty weak Dirichlet condition, the cartesian boundaries, the rigid body contacts, implicit and explicit) take one
option, ``interpolationLinearizedAboutCenter``
(:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`):

- ``False``, the default: they measure the field at the point and distribute their force there with :math:`\mathbf{N}`.
  In an explicit analysis, where the point is where the particle has moved it with :math:`\mathbf{G}`, the penalty
  stiffness :math:`k\,\mathbf{N}\mathbf{G}^\mathsf{T}` is unsymmetric: it does work over a closed cycle of motion and
  can feed energy into softened material next to a support.
- ``True``: they use :math:`\mathbf{G}` for both, for every field: the stiffness
  :math:`k\,\mathbf{G}\mathbf{G}^\mathsf{T}` is symmetric and the constraint conservative;
  ``edelweissmeshfree/constraints/test/vertexpenaltywork_test.py`` checks that a support on the vertices then does no
  work over a closed cycle, nor does a contact on a face of a real particle, whose contact points always use
  :math:`\mathbf{G}`.
  The FE-particle impact (:doc:`fe_projectile_on_rkpm_slab`) opts in for its supports.
  The two operations such a constraint needs besides its own physics -- locating the nodes of its particles in its
  block, and distributing a force on a point -- are in :mod:`edelweissmeshfree.constraints.base.pointsofparticles`,
  which the explicit constraints and the support use, with the penalty springs on points of a particle that give a
  constraint's force and the bound of its stiffness for the critical time step of ``NED`` from the same code. A penalty
  contact of points of one particle with an obstacle in an explicit analysis derives from
  :class:`~edelweissmeshfree.constraints.base.penaltyspringsexplicit.PenaltySpringsOnPointsOfOneParticleExplicit`, as
  the Cartesian boundary and the contact with an analytical surface do: the base keeps the points, the nodes and the
  reach current, so a new one is the gap of a point, its gradient and whether the obstacle is within reach, and a
  constructor that calls the base's.

The contact points on the faces of particles (:ref:`the contact, the tie and the rigid contact of particle faces
<particle-face-contact>`) have no such option: they always use :math:`\mathbf{G}`, since a point on a face stays on
it only if it moves as the particle moves the face.

The base classes
----------------

.. automodule:: edelweissmeshfree.constraints.base.mpmconstraintbase
   :members:
   :private-members:

.. automodule:: edelweissmeshfree.constraints.base.mpmexplicitconstraintbase
   :members:
   :private-members:

.. automodule:: edelweissmeshfree.constraints.base.pointsofparticles
   :members:

.. automodule:: edelweissmeshfree.constraints.base.penaltyspringsexplicit
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

.. _particle-face-contact:

Contact of particles with finite elements
-----------------------------------------

Particles meet a body of finite elements through EdelweissFE's contact constraint,
:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`, the same one that couples two bodies of finite
elements: it is one contact for every kind of contact point (see EdelweissFE's contact theory, *Contact points: the
slave side as points* and *One contact for every kind of contact point*). The contact points of particles are a third
kind, next to the nodes and the quadrature points of finite element faces: quadrature points on faces of particles --
a surface of particles, as the generators and the side sets of an Exodus mesh make them, or (particle, face) pairs
(:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`). Each moves
with the particle's interpolation linearized about its center, :math:`\mathbf{G}` (see *Points of a particle* above),
its force reaches the nodes of the particle's kernel functions with the same :math:`\mathbf{G}`, and it stands for its
share of the face, so the penalty is a stiffness per unit area, as for finite elements. The particles are
updated-Lagrangian: the nodes of their kernel functions carry the increment, so the displacement a point moves with is
the increment, from where the point was at the last update of the connectivity, its origin.

The master is the faceted boundary of the finite elements, which EdelweissFE's surface element generator creates; it
may have corners, convex and re-entrant, such as the crests and roots of a thread (EdelweissFE's contact theory,
*Corners of the master*). An overlap at the start is either real, such as a groove cut smaller than the thread, and
pushed apart, at once or over a time (``initialOverlap=pushApart``, the default), or an error of the geometry, and
adjusted: kept as the stress-free state (``initialOverlap=adjust``; *An initial overlap*). The flat faces of particles
in the curved hole of a screw are the second kind: they are chords of the screw's arcs and lie inside the screw by
their sag, although the concrete fits the screw it was cut by. The contact acts in implicit and explicit analyses. In
``NED``, the stiffness of its springs adds to the frequencies of the nodes they couple, on the particles' side inside
their field matrix (:doc:`choosing_a_solver`), on the side of the finite elements as for any penalty constraint.

Its tests are ``edelweissmeshfree/constraints/test/particlefacecontact_test.py`` (the force of a pressed face, the
tangent by central differences, pairing, entering, tunnelling, an overlap at the start ramped out or adjusted,
restart) and examples 185 (the same result as an analytic wall), 187 (impact of particles and of finite elements on
finite elements, and a stiff contact that sets the time increment), 191 (:doc:`fe_projectile_on_rkpm_slab`), 192 (the
contact patch test, implicit: the pressure a block of particles transmits between two platens of finite elements,
against a block of finite elements; uniform away from the free ends of the particles, and why not at them, see the
module of the contact points) and 193 (a toy pull-out: the head of a stud of finite elements bearing on
particles, implicit).

The same constraint with the law ``type=tied`` **ties** particles to finite elements (EdelweissFE's contact theory, *A
tie: contact that never opens and never slides*; it is not EdelweissFE's ``tie`` constraint, which eliminates the
degrees of freedom of slave nodes, see *Two ties* there). At the first update, every point of a face of a particle
within ``searchDistance`` of the master is paired once with the closest point of its closest facet, mapped onto the
facet's parent face, and held there in every direction by penalty springs, in tension and in compression. A gap or an
overlap of the two surfaces at the pairing -- two meshes of one curved surface never coincide -- is part of the state
at rest. The particles are updated-Lagrangian, as for the contact: at every update the tied points are located again,
where the particles have moved them, and their displacement from rest is measured from there, so a point stays tied to
the same point of the master however far it moves. The tie joins a region of particles to a far field of finite
elements, in implicit and explicit analyses, where the mortar tie of :doc:`coupled_rkpm_fe` (implicit, 2D) does not
reach. A new mesh of the elements hands the tie on (EdelweissFE's contact theory, *A new mesh*): the same points stay
tied to the same points of the master, displaced from rest as at the last accepted state -- in ``NED`` the state of
the evaluation after the particles are reconnected, the first at the accepted configuration. ``NED`` refuses a model
modifier that acts during the analysis in a model with particles; that state is what it checkpoints, and what a new
mesh in a following step, or in a run resumed from the checkpoint, starts from. The points of particles
are never regenerated by a change of the mesh, so they need no interpolation from old points to new ones.

Its tests are in the same file (at rest with a gap and with an overlap, pulled back in every direction, the tangent, a
particle that moves on, a restart, a new mesh of the master) and examples 194 and 195:

- 194, the patch test of the tie, implicit: a column of particles between two blocks of elements, stretched and
  sheared, with a gap or an overlap at the start. It transmits the force of the column, and the error halves with the
  size of the particles; the traction at single points varies. The mechanism is that of example
  192: the particles balance a homogeneous stress with their shape functions at the centers of their faces, the tie
  loads them with :math:`\mathbf{G}` at 2 x 2 points per face, and the particles deviate slightly from the homogeneous
  state within two or three particles of a side of the column. The four springs of a face hold it against the tilts of
  that deviation, so the stiffer the tie, the more the traction varies across a face. The traction of a tie at a single
  point of particles is therefore not a measure of the stress there; its resultant over a few particles is. In shear,
  the springs that hold the sides of the particles add a smaller band and their compliance; the module says why.
- 195, explicit: a pulse through a bar of elements with a tied segment, against the bar of elements alone. A segment
  of elements tied in reflects what the two springs of its ties reflect, transmits the rest, and delays the transmitted
  pulse by their compliance, to first order in how finely the elements resolve the pulse. A segment of particles
  changes the pulse more: it reflects more at its ends, and delays the pulse along its length, by a phase speed of its
  own. Both shrink with the size of the particles and elements, the reflection faster than the ties' own. The state
  the ties accept in ``NED`` hands them on to a new mesh of the elements without a jump of their forces; the test makes
  that mesh by hand in the middle of the pulse, standing for a refinement in a following step or a resumed run.

The springs of the tie raise the frequencies of the points they hold, which ``NED``'s critical time step counts, like
the contact's normal spring, along every axis (:doc:`choosing_a_solver`).

**Moving a deck from** ``ParticlePenaltyFacetContactExplicit``. The contact of the vertices of particles with a faceted
surface is replaced by this contact, as in examples 185, 187 and 191:

- **Why.** It was a second contact next to EdelweissFE's, without its corners of the master, its overlap at the start,
  its implicit path and its rigid bodies; it acted on the vertices, with a penalty per point, in explicit analyses
  only, and its springs were not in the critical time step of ``NED``. The contact above is one constraint for every
  kind of contact point, the faces of particles included.
- **The points.** Contact points on the faces of the particles that can meet the surface, a surface of particles or
  (particle, face) pairs, instead of the particles whose vertices met it. The option
  ``interpolationLinearizedAboutCenter`` is gone: the points always move, and distribute their force, with
  :math:`\mathbf{G}`, which the option chose; its default, :math:`\mathbf{N}`, is not available.
- **The penalty**, a stiffness per unit area. The old contact put its penalty :math:`k_v` at every vertex of every
  given particle: :math:`n_v` vertices on a face of the area :math:`A_f` (2 in 2D, 4 in 3D), so
  :math:`p = k_v\, n_v / A_f` per unit area keeps the stiffness of the face, whatever the number of contact points
  per face. Example 191: :math:`2 \times 10^5` per vertex, faces of 20 mm x 20 mm, so :math:`2 \times 10^5 \cdot 4 /
  400 = 2 \times 10^3`.
- **The master** is a surface of finite elements of the model, which takes the reaction. A deck whose facets were moved
  by something else (``twoWayCoupling=False``, such as a fixed floor) holds the nodes of a layer of elements under
  the surface (example 185), meets a discrete rigid body (below), or, for a plane, uses the Cartesian boundary
  (:class:`~edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit.ParticlePenaltyCartesianBoundaryConstraintExplicit`).
- **The outputs** have EdelweissFE's names: ``contactForceOnParticles`` is ``contactForceOnSlave``, and
  ``numberOfPointsInContact`` is ``numberOfPointsWithASpring``.

Contact of particles with a rigid body
--------------------------------------

The same contact points meet a discrete rigid body -- a support, an indenter, a projectile that does not deform --
through EdelweissFE's :mod:`edelweissfe.constraints.surfacetodiscreterigidbodypenalty`, the contact of the faces and
nodes of finite elements with a rigid body (EdelweissFE's contact theory, *Integrated contact against a discrete rigid
body*): the constraint is built with
:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces` in place of the
faces of elements. Each point is paired with the plane of its closest triangle of the rigid surface, which moves and
rotates with the body's reference point; the force of a point reaches the nodes of the kernel functions of its
particle with :math:`\mathbf{G}`, and the reference
point with the force and its moment. Points farther than the search distance from the rigid surface take no part
until they approach. Its tests are ``edelweissmeshfree/constraints/test/particlefacerigidcontact_test.py`` (the force
and the moment of a pressed face, action = reaction, the tangent by central differences against a moved and rotated
body, a particle that approaches). The particle constraints for analytical surfaces above
(``ParticlePenaltyRigidBodyContact``, with an implicit function) remain the way to describe a boundary by a formula
rather than by a mesh.

.. automodule:: edelweissmeshfree.constraints.base.contactpointsonparticlefaces
   :members:

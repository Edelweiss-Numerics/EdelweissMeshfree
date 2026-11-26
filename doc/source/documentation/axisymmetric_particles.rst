Axisymmetric particles
======================

A body of revolution under a load of revolution -- a tube under internal pressure, an anchor pulled out of a bore, a
ring punch pressed into a block -- deforms the same in every half plane through its axis. Its model is one such half
plane, the **section r-z**: the first coordinate is the radius :math:`r \ge 0`, the second the axial coordinate
:math:`z`, and every point of the section stands for the ring it sweeps about the axis. Whatever the model integrates,
it integrates over these rings, with the circumference :math:`2 \pi r` as the weight: a particle's volume is the volume
of its ring, its mass the mass of the ring, its forces the forces on the whole ring.

The particles are Marmot's axisymmetric particles, the twins of the plane-strain particles:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Axisymmetric particle
     - Stabilization, solvers
   * - ``GradientEnhancedFiniteStrainSQCNIxNSNI/Axisymmetric/Quad``
     - SQCNI smoothed gradients with NSNI; implicit and explicit (``NED``)
   * - ``DisplacementSQCNI/Axisymmetric/Quad``
     - SQCNI smoothed gradients only; implicit only (it has no explicit path)

How the ring weight enters their smoothed gradients, the hoop stretch :math:`r / R` and the hoop stress, why they pass
the axisymmetric patch tests, and what they refuse (VCI, the correction ``CWFCORRECTION``) is derived in Marmot's
documentation: `Axisymmetric Particles
<https://materialmodelingtoolbox.github.io/Marmot/pages/features/axisymmetricparticles.html>`_. This page shows how
they are used in EdelweissMeshfree.

An axisymmetric model
---------------------

The model says that it is axisymmetric, and its particles must agree:

.. code-block:: python

    from edelweissmeshfree.models.mpmmodel import MPMModel

    model = MPMModel(2, axisymmetric=True)

``MPMModel(2, axisymmetric=True)`` is EdelweissFE's
:attr:`~edelweissfe.models.femodel.FEModel.axisymmetric` model, the one an input file makes with
``*job, domain=axisymmetric``. A particle knows whether it is axisymmetric
(:attr:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.axisymmetric`, Marmot's section of the particle),
and :meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.prepareYourself` refuses an axisymmetric particle in a plane or
a 3D model and a plane particle in an axisymmetric one: either would integrate its forces over another measure than
the elements and the contact points next to it, and nothing would balance. For the same reason an axisymmetric model
refuses material points, cells and cell elements: those of EdelweissMeshfree integrate over the section, not over the
rings.

The particles in the section r-z
--------------------------------

The particles are made by the generators of plane particles, with x as the radius and y as the axis; e.g. a ring of
inner radius ``a``, outer radius ``b`` and height ``h``:

.. code-block:: python

    generateRectangularQuadParticleGrid(
        model,
        journal,
        lambda number, vertices, volume: MarmotParticleWrapper(
            "DisplacementSQCNI/Axisymmetric/Quad", number, vertices, 0.0, approximation, material
        ),
        name="ring",
        x0=a, y0=0.0, l=b - a, h=h, nX=nAcross, nY=nUp,
    )

Its surfaces ``ring_left`` and ``ring_right`` are then the inner and the outer cylinder, ``ring_bottom`` and
``ring_top`` the annuli. The section lies in :math:`r \ge 0`: a particle with a vertex below the axis by more than the
round-off of a mesher (:math:`10^{-12}` of the diagonal of the particle's bounding box) is refused when it is made (a
``ValueError``); a
particle with an edge on the axis is a particle like any other, and the contact points of that edge stand for rings of
radius 0, no area: the axis is a line, not a surface of the body.

The nodes of the kernel functions belong at the **centers of the particles**
(:func:`~edelweissmeshfree.generators.kernelmatchingtoparticlegenerator.generateKernelMatchingToParticle`, or a grid of
kernel functions offset by half a particle). The size of an axisymmetric particle
(:func:`~edelweissmeshfree.particles.base.baseparticle.characteristicLength`), from which that generator takes the
support of its kernel function, is the edge of the square of its area in the section, as it is made, not the cube root
of the volume of its ring.

**Which particle with which nodes.** The displacement particle has no stabilization: its smoothed gradients alone keep
it stable, which they do with the nodes at the centers of the particles. With the nodes at the vertices of the
particles, it has a mode of nearly zero energy, a radial oscillation from node to node, which a thick-walled cylinder
under pressure shows as a displacement many times the true one. EdelweissMeshfree cannot tell where a user put the
nodes, so it does not refuse that combination: use the displacement particle with nodes at the particle centers only,
or the gradient-enhanced particle, whose NSNI stabilization controls the oscillating modes.

Loads, supports, contact and ties
---------------------------------

- **Pressure** on the faces of particles
  (:class:`~edelweissmeshfree.stepactions.particledistributedload.ParticleDistributedLoad`, ``"pressure"``): Marmot
  integrates it over the ring of each face at the face's radius after the increment: the ring grows with the hoop
  stretch of the face.
- **Supports at points**
  (:class:`~edelweissmeshfree.constraints.particlepenaltyweakdirichtlet.ParticlePenaltyWeakDirichlet`)
  hold a component at the vertices of particles with a spring each. The value held needs no ring weight, but the
  stiffness of a spring competes with the ring stiffness of the particles, which grows with the radius as
  :math:`2 \pi r`: a fixed penalty holds less well far from the axis. A
  component held at every node of the kernel functions, as :math:`u_z = 0` of a slice in plane strain, is held
  everywhere: the reproducing kernels interpolate zero nodal values as zero.
- **Contact and ties with finite elements** are EdelweissFE's one contact constraint
  (:mod:`edelweissfe.constraints.surfacetodeformablesurfacepenalty`, with the law ``tied`` for a tie), whose contact
  points are the quadrature points of the faces of the particles
  (:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`). The
  points take the model and measure their areas when they are made, as the model integrates: in an axisymmetric model,
  every point stands for the area :math:`2 \pi r_q L_q` of its ring, as the points of the faces of axisymmetric
  elements do (``CX8RUL``, ``GCAX8R``, see EdelweissFE's contact theory, :ref:`edelweissfe:contact-out-of-plane`), so
  that the contact forces balance the particles' forces and the elements' forces on the full
  rings; the constraint refuses points made for a model of another kind. Two points per edge integrate the nodal forces
  of a uniform pressure on a ring exactly. In a plane model, particles are a slice of unit thickness, and so are their
  contact points: the elements they meet must be of plane sections of thickness 1.

The critical time step
----------------------

``NED`` takes the critical time step of the particles from an estimate, a lower bound of it from the row sums of the
absolute value of their mass-scaled stiffness
(:meth:`~edelweissmeshfree.models.mpmdiscretesystem.MPMDiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia`). An
axisymmetric particle bounds its rows as its plane strain twin does: its stiffness depends on its nodes only through the
shape functions and their gradients at its center, with the hoop terms in the derivatives with respect to the
interpolated increment, and the ring weight scales its stiffness and its lumped mass alike. The estimate is a lower
bound for axisymmetric particles as for any. Away from the axis, where the hoop terms, of the order of :math:`E V /
r^2`, are small next to the gradient terms of the particle's size :math:`h`, of the order of :math:`E V / h^2`, it is as
tight as for the plane strain twins; near the axis the two are of one order, and this is not shown. Its margin is the
price of the bound, not of the ring: the Collatz-Wielandt bound of the absolute value of the stiffness with weights
constant per field, and the Cauchy-Schwarz bound of the particles' rows. With ``carry-critical-time-step-weights``,
every estimate takes one more step of the power iteration towards the Perron vector of that absolute value, which
tightens it for both alike (``models/test/criticaltimestepofparticles_test.py``).

The penalty springs of contact and ties add their row sums to the estimate where they act, with the areas of their
rings, as for the axisymmetric elements. Where a penalty constraint of the discrete system acts on the displacement of
the particles -- a contact or a tie on their faces --, the weights of that field stay constant per field, so that the
rows of the springs add as they are: the carried weights then tighten the estimate of the other fields only.

Examples
--------

- ``examples/196_axisymmetric_thick_walled_cylinder_rkpm_test``: a tube under internal pressure, of either particle,
  against Lamé's solution, converging as the particles get smaller.
- ``examples/197_axisymmetric_fe_rkpm_fe_tie_patch_test``: the axisymmetric twin of the tie patch test: a ring of
  particles tied between rings of ``CX8RUL`` elements transmits axial tension across annuli and a radial expansion
  across cylinders as a ring of finite elements does, on thread pools of 1, 4 and 8 threads alike.

Particles
=========

A particle enters the discrete system like an EdelweissFE element: its ``computeKernels(K, P, U, dU, time, dT)``, with
the argument order of :meth:`edelweissfe.elements.base.baseelement.BaseElement.computeKernels`, adds its internal
forces to the residual :math:`\mathbf{P}` and their tangent to :math:`\mathbf{K}`. A particle is incremental on the
current configuration: its kernel functions move with it, and their nodal values are increments, so it reads ``dU``,
not ``U``. A Marmot particle whose material cannot integrate the increment -- Marmot throws -- requests a cutback
(EdelweissFE's ``CutbackRequest``), as an EdelweissFE element does, and the solver retries the increment smaller.

In an explicit analysis (EdelweissFE's ``NED``), a particle is advanced, accepted twice, reconnected and evaluated in
every increment, in an order that its implementation relies on: a zero-length increment leaves the rates alone, an
accept is idempotent, and the material's tangent is required. The estimate of the critical time step is not part of
that order: the particle describes its stiffness to Marmot's bound and is left as it was found, wherever it is asked;
the stiffness is that of an accepted state after the accept, where ``NED`` asks for it. The calls, in their
order, and these contracts are written down once, in Marmot's documentation: `the explicit life cycle of a particle
<https://materialmodelingtoolbox.github.io/Marmot/pages/features/meshfreeparticles.html#explicit-life-cycle-of-a-particle>`_.

``BaseParticle`` class
----------------------

.. automodule:: edelweissmeshfree.particles.base.baseparticle
   :members:
   :private-members:

``MarmotParticleWrapper`` class
-------------------------------

.. automodule:: edelweissmeshfree.particles.marmot.marmotparticlewrapper
   :members:
   :private-members:

Particles
=========

A particle enters the equation system like an EdelweissFE element: its ``computeKernels(K, P, U, dU, time, dT)``, with
the argument order of :meth:`edelweissfe.elements.base.baseelement.BaseElement.computeKernels`, adds its internal
forces to the residual :math:`\mathbf{P}` and their tangent to :math:`\mathbf{K}`. A particle is incremental on the
current configuration: its kernel functions move with it, and their nodal values are increments, so it reads ``dU``,
not ``U``.

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

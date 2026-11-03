Particle managers
=================

.. automodule:: edelweissmeshfree.particlemanagers.base.baseparticlemanager
   :members:
   :private-members:

Reusing neighbour lists: the skin
---------------------------------

Searching for the kernel functions that cover each particle is the single most expensive part of a
dynamic meshfree simulation, and in a dynamic simulation it is also largely redundant. Kernel functions
are bonded to particle centres, so a neighbour set changes through *deformation*, not through the body
as a whole moving: on an explicit Taylor bar impact, a mean of 0.41% of particles change their set from
one increment to the next, and most increments change none at all.

:class:`~edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager.KDBinOrganizedParticleManager`
can therefore be given a **neighbour list skin**. The search then reports every kernel function that
covers a particle *or comes within the skin of covering it*, and that answer is reused until accumulated
motion could have carried something across the remaining margin. Increments in between still rebuild the
shape functions, which have to be rebuilt in any case because the particles have moved.

This is exact rather than approximate. The kernel functions the skin admits early evaluate to exactly
zero at the particle, and the reproducing-kernel reconstruction discards anything that does, so the
shape functions are the ones a full search would have produced. A kernel function that *leaves* the
support is handled by the same mechanism. ``examples/144_neighbour_list_skin_test`` asserts this, by
running the same impact with the skin off and on and requiring the two displacement fields to be equal.

Choosing the skin is a trade-off, not a monotone improvement. A larger skin searches less often but
admits more kernel functions, and each of those costs something in the reconstruction and in the wider
degree-of-freedom stencil that the physics then has to touch. Measured on an explicit Taylor bar impact
at 108,864 degrees of freedom, over 20 increments:

=========================  ==================  ==================
``neighbourListSkin``      searches out of 21  time for the step
       ``Fraction``
=========================  ==================  ==================
0.0 (search every step)    22                  75.4 s
0.02                       7                   59.7 s
**0.05**                   **3**               **52.8 s**
0.10                       2                   63.7 s
=========================  ==================  ==================

So 0.05 is best here and 0.10 is worse than 0.05 despite searching less often. Since the optimum depends
on the impact velocity, the increment size, the support radius and how fast the body deforms, the
default is ``0.0`` -- every increment searches, exactly as before -- and a value has to be chosen
deliberately for the problem at hand.

A skin requires every kernel function to have box support, because the reuse argument relies on the
search testing an inflated bounding box; the manager refuses a non-zero skin otherwise rather than
silently over-reporting neighbours.

``KDBinOrganizedParticleManager`` class
---------------------------------------

.. automodule:: edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager
   :members:
   :private-members:

Legacy implementation
---------------------

.. automodule:: edelweissmeshfree.particlemanagers.oldkdbinorganizedparticlemanager
   :members:
   :private-members:

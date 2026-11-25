Particle managers
=================

.. automodule:: edelweissmeshfree.particlemanagers.base.baseparticlemanager
   :members:
   :private-members:

Searching the kernel functions of the particles
-----------------------------------------------

In every increment, a particle needs the kernel functions that *cover* it: those whose support holds one of its
evaluation points (the centres of its faces, or its vertices, depending on the particle). The particles move, and with
kernel functions bonded to the particles the kernel functions move too, so
:class:`~edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager.KDBinOrganizedParticleManager` searches anew
in every increment.

**Why bins?** Without any order, every evaluation point would be tested against every kernel function: the cost grows
with the number of points *times* the number of kernel functions, and both grow with the model. Sorted into bins, a
point is tested only against the few kernel functions of its own bin, and the cost grows with the number of points
alone. This is the *broad phase* of the search; the test of a candidate is the *narrow phase*.

The search works on *groups of points*: the evaluation points of one particle form a group, consecutive in one array
of all points, and ``firstPointOfGroup[g]`` is where group ``g`` starts (``firstPointOfGroup[g + 1]``, where it ends).
The search returns its answer in the same layout: the kernel functions found for group ``g`` are
``kernels[firstKernelOfGroup[g] : firstKernelOfGroup[g + 1]]``. It goes in three steps.

1. **Bins.** The boxes of the kernel functions are sorted into a regular grid of bins, each bin half as wide as the
   mean box: a kernel function is in every bin its box overlaps
   (:class:`~edelweissmeshfree.particlemanagers.binsofkernelfunctions.BinsOfKernelFunctions`). The bins are filled by a
   *counting sort*: count the kernel functions of every bin, take the cumulative sum of the counts as the first place
   of every bin, then put every kernel function into its places. The result is the compressed layout of a sparse
   matrix, one row per bin: the kernel functions of bin ``b`` are
   ``kernelsInBins[firstKernelOfBin[b] : firstKernelOfBin[b + 1]]``. An evaluation point lies in exactly one bin, and
   every box that holds the point overlaps that bin, so the kernel functions of that bin are the only candidates for
   the point.
2. **Box test.** A candidate covers the point if the point lies strictly inside its box. For the kernel functions of
   EdelweissMeshfree (tensor-product B-splines) the support *is* the box, so this is the answer; a point on the boundary
   lies where the kernel function is zero and is not covered. A kernel function whose support is smaller than its box
   (``hasBoxSupport`` is false) is asked in addition whether its support holds one of the points.
3. **Assignment.** A particle gets its kernel functions in the order of their labels, each once. The kernel functions of
   a particle *changed* if the numbers of the kernel functions found differ from the numbers the last search recorded
   for it. A particle whose kernel functions changed is assigned the new ones (``assignKernelFunctions``); every other
   particle keeps its kernel functions and only updates its shape functions (``updateShapeFunctions``), because it has
   moved. The particles whose kernel functions changed are reported
   (:attr:`~edelweissmeshfree.particlemanagers.base.baseparticlemanager.BaseParticleManager.particlesWithChangedKernelFunctions`):
   only they are located anew in the degrees of freedom.

A grid of 4 x 3 bins with the boxes ``A``, ``B`` and ``C`` of three kernel functions. The point ``x`` lies in the
second bin of the middle row, which lists ``A`` and ``B``: they are its only candidates, and the box test keeps ``A``.
``C`` is never tested for ``x``.

.. code-block:: text

   +---------+---------+---------+---------+
   |  **************   |         |         |
   |  * A    |     *   |        ********** |
   |  *      |     *   |        *|       * |
   +--*------+-----*---+--------*+-------*-+
   |  *      |x **************  *|       * |
   |  **************   |     *  *|    C  * |
   |         |  *      |     *  *|       * |
   +---------+--*------+-----*--*+-------*-+
   |         |  *      |  B  *  ********** |
   |         |  **************   |         |
   |         |         |         |         |
   +---------+---------+---------+---------+

The particles are searched in chunks on the thread pool, and two things make that pay off:

- The loops of steps 1 and 2 run in compiled code (``binsofkernelfunctions.pyx``): they visit every evaluation point and
  its candidates in every increment, and a few interpreted operations per point would cost more than all the rest of
  an explicit increment.
- Python objects are touched only for the particles whose kernel functions changed. In free-threaded Python, every
  reference a worker thread takes to an object created by another thread is an atomic operation on that object, and
  building new lists of the kernel functions for all particles in every increment takes *longer* with more threads.
  So the search compares numbers, and a Marmot particle updates its shape functions with the kernel functions Marmot
  holds, without a Python object.

What remains of the update is mostly the evaluation of the shape functions, in Marmot, which no search can save.

Reusing neighbour lists: the skin
---------------------------------

The manager can be given a **neighbour list skin** (``neighbourListSkinFraction``, a fraction of the smallest support
half-width). Every box is then grown by the skin, so the search reports every kernel function that covers a particle
*or comes within the skin of covering it*, and that answer is reused until the motion since the search -- of the
particles and of the kernel functions relative to each other -- could have carried a point across the skin. Increments
in between still update the shape functions, which have to be updated in any case because the particles have moved.

This is exact rather than approximate. The kernel functions the skin admits early evaluate to exactly zero at the
particle, and the reproducing-kernel approximation skips every kernel function that does, so the shape functions are
the ones a search without the skin would have produced. A kernel function that *leaves* the support is handled by the
same mechanism. ``examples/144_neighbour_list_skin_test`` asserts this, by running the same impact with the skin off
and on and requiring the two displacement fields to be equal.

Since the search is compiled, the skin saves little: it saves the search and the bins, which are small next to the
shape functions, and it pays for the motion check in every increment and for the extra kernel functions, which every
particle evaluates and assembles. The default is therefore ``0.0``: every increment searches.

A skin requires every kernel function to have box support, because the reuse argument relies on the search testing a
grown box; the manager refuses a non-zero skin otherwise rather than silently over-reporting neighbours.

``KDBinOrganizedParticleManager`` class
---------------------------------------

.. automodule:: edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager
   :members:
   :private-members:

The bins of the kernel functions
--------------------------------

.. automodule:: edelweissmeshfree.particlemanagers.binsofkernelfunctions
   :members:

Legacy implementation
---------------------

.. automodule:: edelweissmeshfree.particlemanagers.oldkdbinorganizedparticlemanager
   :members:
   :private-members:

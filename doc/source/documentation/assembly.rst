System matrix assembly
======================

For a meshfree simulation the assembly is not a minor cost, and it often decides how large a model fits in memory.
EdelweissFE's implicit solvers assemble the system matrix of an
:class:`~edelweissmeshfree.models.mpmmodel.MPMModel` in one of two ways, chosen by their option
``systemMatrixAssembly``. The machinery is EdelweissFE's -- see its *System matrix assembly* page for the addressing,
the offset map and the C++ core; this page is about choosing between the two for a meshfree model.

Why assembly is a memory problem
--------------------------------

Every particle contributes a dense block over the kernel functions in its support, and the supports of neighbouring
particles overlap heavily, so many blocks add to the same entry of the matrix. The default path, ``vij``, writes each
block into its own slice of one **VIJ staging array**, with the global row and column of every value, and sums the
duplicates into CSR afterwards. The staging array, its row and column indices and its gather map therefore grow with
the number of contributions, not with the number of entries of the matrix: with a large support, they are many times
larger than the matrix they produce.

The other path, ``direct``, derives the CSR pattern from the DOF lists of the cells and particles themselves, gives
every contribution its destination in the CSR matrix in advance, and scatters it there. Neither a staging array nor
the row and column indices of the contributions exist at all.

A cell or a particle couples all its degrees of freedom: its block is the dense ``nDof x nDof`` matrix, column-major,
in its own DOF numbering (EdelweissFE's
:class:`~edelweissfe.numerics.systemmatrixcontributor.SystemMatrixContributorBase`). The Marmot cells and particles,
extension types outside that base class, declare it themselves.

Switching it on
---------------

.. code-block:: python

    solver = NISTParallel(
        jobInfo, journal, linsolver="pardiso", **meshfreeCriterionOptions | {"systemMatrixAssembly": "direct"}
    )

``NIST``, ``NISTParallel`` and ``NISTPArcLength`` take the option. Nothing else changes: the elements, cells and
particles ask for the block of their entity as before, and only what sits behind that request differs (EdelweissFE's
:class:`~edelweissfe.numerics.systemmatrixassembly.SystemMatrixAssembly`). The result is the same up to the order in
which the contributions are summed.

What it costs
-------------

The threads of ``NISTParallel`` compute the entities in parallel, and all of them scatter into the one CSR data array
of the matrix, every addition atomic; per thread there is only a scratch block. So the memory of the direct path does
not grow with the number of threads, while the staging path's memory grows with the number of contributions:

- When the entries of the matrix receive many more contributions than there are threads -- RKPM with a large support
  -- the direct path needs much less memory, and is also faster, since there are no duplicates to sum.
- When the entries receive few contributions -- the cells of the material point method, finite elements -- the
  staging array is small, and the direct path saves little.

The order in which the threads add to an entry depends on their timing, so with more than one thread the direct path
reproduces a previous run to round-off, not bit for bit.

The staging path has a hard limit as well: its gather map indexes the contributions with 32-bit integers, so it cannot
be built for more than ``INT32_MAX`` (about 2.15e9) contributions, and raises ``OverflowError``. Beyond that size the
direct path is the only one.

Where the memory ceiling is
---------------------------

Once the staging array is gone, the peak memory of a large model is dominated by transients outside the solve: the
construction of the CSR pattern, and its reconstruction whenever the connectivity changes -- which, as particles move,
happens on many increments. Size a machine for this peak, not for what the solve occupies, and if a run dies of
memory, look at the rebuild of the discrete system before suspecting the solve.

Reference
---------

The assembly classes are EdelweissFE's, in :mod:`edelweissfe.numerics.systemmatrixassembly`.

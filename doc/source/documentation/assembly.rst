System matrix assembly
======================

For a meshfree simulation the assembly is not a minor cost, and it is usually what decides how large a
model will fit in memory. This page explains the two available assembly paths, how to switch, what each
costs, and where the ceiling actually sits.

.. warning::

   The figures below were measured on one problem class -- a Taylor-bar impact, RS-SNNI x SDI RKPM
   discretisation, support scaling factor 2.2, between 15,120 and 120,960 degrees of freedom, on
   deformed post-impact states. They are evidence, not constants.

Why assembly is a memory problem
--------------------------------

Every particle contributes a dense block over its own neighbours, and those blocks overlap heavily. The
long-standing path writes each block into its own slice of one **VIJ staging array**, then sums the
duplicates into CSR afterwards. At 43,350 DOF that means **1.72e9 stored contributions for a 71.2 M
entry matrix** -- about 24 duplicates per entry -- so the staging array is 12.84 GiB and its gather map
a further 6.42 GiB, to produce a 0.53 GiB result.

The alternative is to give every contribution its destination in advance and *scatter* it straight into
CSR. Then no staging array exists at all. The machinery lives in EdelweissFE -- see its
*System matrix assembly* page for the addressing, the offset map and the C++ core; this page is about
using it from a meshfree simulation.

Switching it on
---------------

.. code-block:: python

    nonlinearSolver.useDirectCSRAssembly = True

That is the whole change. What it does:

- no VIJ staging array is allocated,
- the particle kernel scatters each block through the offset map from inside its own ``prange``,
- every other contributor -- cells, elements, cell elements, constraints, body loads, cell and particle
  distributed loads -- writes into a small reused scratch block that is then scattered.

None of the contributors themselves changed. They ask the same question they always did (*"give me the
block for this entity"*); only what sits behind that request differs, via
:class:`~edelweissmeshfree.numerics.systemmatrixcontribution.SystemMatrixContributionBase`.

What it is worth
----------------

At 43,350 DOF, one restarted increment, 16 threads:

.. list-table::
    :width: 100%
    :widths: 45 25 25
    :header-rows: 1

    * -
      - staging path
      - direct path
    * - ``compute system`` per Newton iteration
      - 5.221 s
      - **4.244 s**
    * - gather / reduce per iteration
      - 1.335 s
      - **0.466 s**
    * - ``solve step``
      - 127.28 s
      - **116.70 s**
    * - **peak resident memory**
      - 60.23 GiB
      - **34.59 GiB**
    * - memory during the solve
      - 41.61 GiB
      - **25.83 GiB**

**1.23x on the assembly, 1.09x on the whole solve step, 1.74x on peak memory.** And measured across a
2.8x range in problem size, assembly cost and memory both scale **linearly** in DOF
(``dof**0.97`` from 43,350 to 70,800, ``dof**1.06`` from 70,800 to 120,960).

The result is unchanged to round-off. Verified against a *control pair* rather than against zero: two
runs of the staging path differ from each other by 2.2e-16 in the exported reaction force, and the
direct path differs from the staging path by **the same 2.2e-16**. In other words it is as close to the
reference as the reference is to itself.

It is also required above about 50,000 DOF
------------------------------------------

The staging path stores a 32-bit index per COO pair in its gather map, so it cannot build that map
beyond ``INT32_MAX`` = 2.15e9 pairs. For this problem class that limit is reached at roughly **50,000
DOF -- with only a third of a 187 GB machine in use**. It is not a memory limit; it is a counter limit,
and it applies to the staging path only.

.. list-table:: Measured feasibility
    :width: 100%
    :widths: 25 20 20 35
    :header-rows: 1

    * - grid
      - DOF
      - COO pairs
      - staging path
    * - 17 x 17 x 50
      - 43,350
      - 1.5e9
      - works
    * - 18 x 18 x 51
      - 49,572
      - 1.8e9
      - works, near the limit
    * - 19 x 19 x 56
      - 60,648
      - 2.2e9
      - **raises** ``OverflowError``
    * - 24 x 24 x 70
      - 120,960
      - 4.6e9
      - impossible

So above ~50,000 DOF the direct path is not an optimisation, it is the only path. 120,960 DOF runs on the
direct path at 126.77 GiB peak.

.. note::

   The feasibility probe ``dof_envelope.py`` builds the *undeformed* connectivity and undercounts pairs
   by about 13.5% against a deformed state, because deformation grows the neighbour lists. Scale its
   numbers up before judging whether a grid will survive a whole simulation.

Private copies, and reproducibility
-----------------------------------

Threads need somewhere to accumulate. By default each gets its own copy of the CSR data, which is
synchronisation-free and keeps the summation order fixed, so results are bit-reproducible. That costs
``numThreads x nnz x 8`` bytes. ``directCSRNumBuffers`` reduces the count, down to ``1`` for a fully
atomic scatter.

Measured at 43,350 DOF on 16 threads: 16 copies cost 11.72 GiB and 3.43 s; one copy costs 3.76 GiB and
3.83 s, i.e. **11.8% slower for 3.1x less memory**. The intermediate setting of 4 is not worth having --
it pays 10.2 of those 11.8 percentage points for only part of the saving, because the cost is the atomic
instruction rather than contention.

**Keep the default unless memory is genuinely binding.** Atomics make the summation order depend on
thread interleaving, so re-running a simulation no longer reproduces the previous run bit-for-bit. Given
that peak memory is usually set by something other than the assembly (below), the saving often buys
nothing while the reproducibility loss is real.

Where the memory ceiling actually is
------------------------------------

Once the staging array is gone, peak memory is dominated by **transients that have nothing to do with
the solve**:

.. list-table:: at 120,960 DOF
    :width: 100%
    :widths: 50 25 25
    :header-rows: 1

    * -
      - peak
      - during the solve
    * - single increment
      - 100.34 GiB
      - 88.89 GiB
    * - multi-increment run
      - 126.77 GiB
      - ~97 GiB

The gap is the CSR pattern construction, and rebuilding the pattern and assembler whenever the
connectivity changes -- which happens on most increments (22 rebuilds in 38 increments at this size).
Both are one-off costs per pattern, not per iteration.

Practical consequence: **size a machine for the peak, not for what the solve occupies**, and expect the
peak to sit well above it. If a run dies of memory, look at the setup phase before suspecting the solve.

Limitations
-----------

- **Only dense contributions.** The scatter addresses a dense ``nDof x nDof`` column-major block. Every
  meshfree entity lays its contribution out that way, but a constraint is free to declare a sparse
  custom VIJ pattern; such an entity raises ``NotImplementedError`` at registration rather than being
  silently misassembled.
- **The arc-length and generalized-alpha solvers are not supported.** Both form more than one matrix
  (a reference-load stiffness, or a mass matrix) and combine them by array arithmetic, which a path that
  accumulates into a single CSR copy cannot express. They stay on the staging path and say so.
- **The non-particle contributors are assembled sequentially.** They are under 2% of the runtime, so
  this is deliberate; a threaded caller reaching
  :class:`~edelweissmeshfree.numerics.systemmatrixcontribution.DirectCSRContribution` raises rather than
  racing.
- **The assembly diagnostics need the staging path**, since they compare against it. Requesting them
  together with ``useDirectCSRAssembly`` raises.

Diagnostics
-----------

Three switches on the parallel Marmot solver, all off by default:

``verifyDirectCSRAssembly``
    Assemble the particles several ways on identical state and cross-check. Reports an addressing check
    on identical values, a re-evaluation noise floor, and the fused kernel on 16 and on 1 thread. The
    fused result is judged against the *measured* noise floor rather than against zero -- which is what
    allows it to stay meaningful when atomics make the floor non-zero. Costs about 4x the particle
    assembly; the solve itself continues on the staging path, so no result changes.

``timeDirectCSRAssembly``
    Time both paths component by component, in the same process, on the same increment, alternating
    which goes first. Also prints a term-by-term memory table for both paths with the C++ accounting as
    a cross-check on the arithmetic.

``directCSRBufferSweep``
    Time several private-copy counts in turn, e.g. ``(16, 4, 1)``, on the same state -- so the
    memory/speed trade is measured rather than argued.

.. note::

   Two traps when measuring this. Restart from a checkpoint and run **one** increment; and check that
   the restart took before quoting anything -- the exported reaction force should have one row, at the
   checkpoint's time. Also note that the increment counter is tested *after* an increment is yielded, so
   a maximum of ``N`` runs the checkpoint's increment *and* the next one.

Reference
---------

``SystemMatrixContribution`` classes
------------------------------------

.. automodule:: edelweissmeshfree.numerics.systemmatrixcontribution
   :members:
   :private-members:

Choosing a solver
=================

The nonlinear solver
--------------------

An :class:`~edelweissmeshfree.models.mpmmodel.MPMModel` is solved by EdelweissFE's implicit static solvers, as a finite
element model is: the solver asks the model what its equation system is made of (see EdelweissFE's documentation, *What
the solver asks the model*), and the model answers with its active domain.

.. list-table::
    :width: 100%
    :widths: 25 75
    :header-rows: 1

    * - Solver
      - Use it for
    * - ``NIST``
      - a quasi-static step; the elements, cells and particles are computed one after the other
    * - ``NISTParallel``
      - the same, with the elements, cells, particles and material points computed on all threads
        (``OMP_NUM_THREADS``)
    * - ``NISTPArcLength``
      - a quasi-static step under arc length control, e.g. past a limit point; the controller is a step action, named by
        the solver option ``arcLengthController``

A simulation is set up from Python as any EdelweissFE step. The model holds what defines its connectivity -- the
managers that assign material points to cells (``mpmManagers``) and particles to kernel functions
(``particleManagers``), the managers of the variationally consistent integration (``vciManagers``), and the
constraints. The step holds the loads, the Dirichlet boundary conditions and the arc length controller, by type:

.. code-block:: python

    from edelweissfe.solvers.nonlinearimplicitstaticparallel import NISTParallel
    from edelweissfe.steps.adaptivestep import AdaptiveStep
    from edelweissfe.steps.stepmanager import StepActionCollection

    from edelweissmeshfree.solvers.meshfreeconvergencecriterion import (
        jobInfoWithMeshfreeTolerances,
        meshfreeCriterionOptions,
    )

    model.particleManagers.append(particleManager)
    model.vciManagers.append(vciManager)
    for constraint in constraints:
        model.constraints[constraint.name] = constraint

    stepActions = StepActionCollection()
    stepActions["distributedload"][pressure.name] = pressure

    jobInfo = jobInfoWithMeshfreeTolerances()
    solver = NISTParallel(jobInfo, journal, linsolver="pardiso", **meshfreeCriterionOptions)
    step = AdaptiveStep(
        0, model, fieldOutputController, journal, jobInfo, solver, outputManagers, stepActions,
        startInc=0.1, maxInc=0.1, minInc=1e-3, maxNumInc=100, maxIter=15, criticalIter=5, maxGrowIter=10,
    )
    step.solve()

``jobInfoWithMeshfreeTolerances`` and ``meshfreeCriterionOptions`` set the convergence criterion of EdelweissMeshfree's
former solvers (see :mod:`~edelweissmeshfree.solvers.meshfreeconvergencecriterion`); tolerances that differ are passed
by name, e.g. ``jobInfoWithMeshfreeTolerances(fluxResidualTolerance=5e-3)``. The settings of the former solvers map
onto EdelweissFE's as follows:

.. list-table::
    :width: 100%
    :widths: 45 55
    :header-rows: 1

    * - Former solver
      - EdelweissFE
    * - ``max. iterations``, ``critical iterations``, ``allowed residual growths``
      - step options ``maxIter``, ``criticalIter``, ``maxGrowIter``
    * - ``failed increment cutback factor``
      - step option ``cutbackFactor``
    * - the time stepper
      - step options ``startInc``, ``maxInc``, ``minInc``, ``maxNumInc``, ``increaseFactor``; a time stepper without the
        zero increment: ``makeZeroIncrementFirst=False``
    * - relative and absolute tolerances
      - the tolerances of ``jobInfoWithMeshfreeTolerances``
    * - ``predictor=QuadraticPredictor()`` (or ``LinearPredictor()``)
      - solver options ``extrapolation="quadratic"`` (``"linear"``), ``retryWithoutExtrapolation=True``,
        ``extrapolateAcrossConnectivityChange=True``
    * - ``line search``, ``line search after n iterations``, ``line search every n iterations``, ``line search alphas``
      - solver options ``lineSearch``, ``lineSearchAfterIterations``, ``lineSearchEveryIterations``,
        ``lineSearchStepLengths``
    * - ``fall back to quasi Newton after allowed residual growths``
      - solver option ``fallBackToQuasiNewton``
    * - a linear solver instance
      - solver option ``linsolver``: a type with its defaults, or a definition in ``jobInfo["linearSolvers"]``
        (see the worked configuration below)
    * - ``useDirectCSRAssembly``
      - solver option ``systemMatrixAssembly="direct"`` (see :doc:`assembly`)
    * - ``restartWriteInterval``, ``readRestart``
      - EdelweissFE's restart checkpoints, see below

**Restart.** EdelweissFE's restart output manager writes a checkpoint of the converged state; an ``MPMModel`` adds the
state of its material points and particles, from which their positions follow (see
:meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.writeRestart`). The output manager writes the state of every output
manager, which it finds in ``model.outputManagers``. A resumed run builds the model as the checkpointed run did,
restores its state, and continues the step:

.. code-block:: python

    from edelweissfe.outputmanagers.restart import OutputManager as RestartOutputManager
    from edelweissfe.outputmanagers.restart import RestartOutputManagerSchema
    from edelweissfe.utils.checkpoint import ResumeCheckpoint

    restartOutput = RestartOutputManager(
        "restart", model, fieldOutputController, journal, None,
        configuration=RestartOutputManagerSchema(writeInterval=1, baseName="restart"),
    )
    outputManagers = [ensightOutput, restartOutput]
    model.outputManagers = {outputManager.name: outputManager for outputManager in outputManagers}

    # in the resumed run, right after model.prepareYourself(journal):
    resumeCheckpoint = ResumeCheckpoint("restart_1.h5")
    resumeCheckpoint.restoreModel(model, journal)
    # ... set up the step as before, then
    step.solve(resumeCheckpoint)

**Differences to the former solvers.** The same settings give the same Newton iterations; a few things are different
by design:

- The reaction forces ``P`` at the nodes are those of the elements, cells and particles; the former solvers added the
  forces of the constraints.
- A constraint that couples the same degree of freedom more than once adds every contribution to it; the former
  solvers kept only one of them.
- A cell, particle or material point whose material fails in Marmot asks for the increment to be cut back to half its
  size, as an EdelweissFE element does, and the step cuts back by the larger of this and ``cutbackFactor``; the former
  solvers always cut back by their own factor.
- A step that reaches ``maxNumInc`` ends there, and the simulation continues; the former solvers reported a failed
  incrementation. A step that cannot reduce its increment any further raises ``StepFailed``.
- The quadratic extrapolation is the parabola through the last three accepted states; the former quadratic predictor
  was not exact.
- The scalar variables, e.g. Lagrange multipliers, are checked for convergence with EdelweissFE's tolerances.

The former solvers are kept, deprecated, in :mod:`edelweissmeshfree.solvers.deprecated`.

Choosing a linear solver
------------------------

Every Newton iteration of an implicit meshfree simulation spends most of its time in one linear solve,
so the choice of linear solver and its settings usually matters more than anything else you can tune.
This section collects what has been **measured** on RKPM problems, what turned out not to matter, and how
to find out for your own model.

.. warning::

   Every number in this section was measured on one problem class: a **Taylor-bar impact, RS-SNNI x SDI
   RKPM discretisation, support scaling factor 2.2**, between 15,120 and 120,960 degrees of freedom,
   on deformed post-impact states with active penalty contact.

   They are reported as evidence, not as universal constants. The support radius in particular changes
   this solver's behaviour more than anything else (see :ref:`the operator <rkpm-operator-properties>`),
   so treat a figure from a different discretisation as a hypothesis to re-measure.

Start here
**********

.. list-table:: A first choice
    :width: 100%
    :widths: 30 30 40
    :header-rows: 1

    * - Situation
      - Use
      - Why
    * - below ~40,000 DOF
      - ``pardiso`` with ``reuseSymbolicFactorization=True``
      - a direct factorisation still wins, and symbolic reuse is worth 3x on its own
    * - above ~40,000 DOF
      - ``blockamg``
      - AMG's iteration count is mesh independent here; a direct factorisation is not
    * - above ~50,000 DOF
      - ``blockamg``, and ``systemMatrixAssembly="direct"`` on the solver
      - the staging assembly path cannot build its gather map beyond ~2.15e9 COO pairs
    * - memory bound rather than time bound
      - ``blockamg`` + ``systemMatrixAssembly="direct"``
      - a direct factorisation's fill-in is the one term that has never been bounded here

The crossover near 40,000 DOF is a measured property of this problem class, not a general rule. Below
it, PARDISO with symbolic reuse beat every iterative configuration tried; above it, the direct solver's
cost grows as roughly ``dof**1.4`` while AMG's grows about linearly.

The direct solver
*****************

``reuseSymbolicFactorization``
    **Always turn this on** for a Newton sequence. The sparsity pattern is identical across the
    iterations of one increment, so the reordering and symbolic factorisation (PARDISO phase 11) can be
    computed once instead of per solve. Measured: **3.0-3.2x faster** overall, with solutions deviating
    by 1e-13 -- phase 11 was 29 s of a 41 s linear solve before, and one call instead of twelve after.

Threading
    The direct solve saturates at about **8 MKL threads**: 2.9x at 16 threads, and nothing further at
    36. Do not size a machine for a direct solver on thread count alone.

.. _rkpm-operator-properties:

The operator is the thing
*************************

Before tuning a preconditioner it is worth knowing what the RKPM operator looks like, because it does
not resemble the sparse operators AMG was designed for. Measured at 43,350 DOF:

.. list-table::
    :width: 100%
    :widths: 45 55
    :header-rows: 0

    * - stored entries per row
      - ~1643 (max 2925)
    * - structural density
      - 3.79%
    * - symmetry
      - exact, ``||A - A^T||_max / ||A||_max`` = 9.1e-15
    * - diagonal spread
      - only 52.8x (min 1.15e4, max 6.09e5)
    * - **median off-diagonal, relative to** ``sqrt(|a_ii| a_jj|)``
      - **1.29e-06**
    * - off-diagonals that are exactly zero
      - 2.59%

Two consequences follow, and both are counter-intuitive.

**Penalty contact is not the problem.** A diagonal spread of 52.8x is mild, and only 44 rows sit above
the 99.9% quantile. Contact stiffness does not create the decades of scale separation that would confuse
a strength-of-connection criterion -- which is consistent with ``eps_strong`` having no measurable effect
(below).

**The operator is numerically far sparser than it is structurally.** The typical stored entry is a
millionth of the diagonal. Smoothed aggregation therefore weighs ~1643 candidates per row where a
handful matter, and the Galerkin product carries the rest down to every coarse level. This is the single
most useful thing to know about tuning AMG here, and it is what ``hierarchyDropTol`` exploits.

**The support radius dominates.** The same preconditioner configuration needed **17 outer iterations at
support 1.8 and 108.5 at support 2.2**. If iteration counts are unacceptable, the discretisation is a
bigger lever than the solver -- but note that the support radius is an accuracy decision, not a free
one: the solution changes with it.

Settings that matter
********************

``hierarchyDropTol`` -- the largest single win
""""""""""""""""""""""""""""""""""""""""""""""

Builds each AMG hierarchy from a copy of the matrix with off-diagonal ``a_ij`` dropped where
``|a_ij| < hierarchyDropTol * sqrt(|a_ii| |a_jj|)``. Only the *preconditioner* is sparsified: the
operator the Krylov method applies, and therefore the residual it drives to zero, is untouched. **A
preconditioner change cannot move the converged solution beyond the outer tolerance** -- it can only
change how many iterations reaching it takes.

Measured at 70,800 DOF over 3 increments and 35 solves:

.. list-table::
    :width: 100%
    :widths: 12 20 16 18 18 16
    :header-rows: 1

    * - ``tol``
      - nnz for the hierarchy
      - iters / solve
      - GMRES s / iter
      - ``linear solve``
      - ``solve step``
    * - off
      - 122.8 M
      - 131
      - 0.0626
      - 377.16 s
      - 889.76 s
    * - 1e-6
      - 48.7 M (39.7%)
      - 133
      - 0.0408
      - 295.37 s
      -
    * - 1e-5
      - 36.1 M (29.4%)
      - 128
      - 0.0356
      - 269.07 s
      -
    * - 1e-4
      - 24.5 M (19.9%)
      - 124
      - 0.0320
      - 239.81 s
      - 736.56 s
    * - 1e-3
      - 13.1 M (10.7%)
      - 132
      - 0.0276
      - 230.22 s
      -
    * - 1e-2
      - 5.25 M (4.3%)
      - 129
      - 0.0241
      - 208.50 s
      - 713.70 s

**1.81x on the linear solve and 1.25x on the whole solve step**, with every configuration converging the
same increments with the same single cutback.

The important column is *iterations per solve*: 131, 133, 128, 124, 132, 129. **Flat across four orders
of magnitude**, while retaining as little as 4.3% of the entries. The discarded entries were never
carrying preconditioning information, and the entire gain is cost per iteration (2.60x cheaper). Do not
expect faster convergence from this -- expect cheaper iterations.

The default is ``0.0`` (off) for two honest reasons: the sweep never found the point where quality
breaks, so the safe end of the range is unknown; and it has been measured on one operator at one support
radius. ``1e-4`` has the most margin on either side for this problem class.

Plain truncation is what to use. The standard refinement -- lumping each dropped entry onto the diagonal
so row sums are preserved exactly -- was implemented and measured, and is neutral at best and 8.1% slower
at ``1e-2``; see ``hierarchyDropLumping`` in the table below. Three independent results now point the same
way on this operator (the rigid-body nullspace is inert, ``eps_strong`` does nothing, and lumping does
nothing): **near-null-space and strength-of-connection fidelity are not what limits this preconditioner.**
What limited it was the cost of applying a hierarchy built from a numerically dense operator, which is
what the truncation fixes.

The criterion is scale invariant, and symmetric in ``(i, j)`` for a symmetric matrix, so the sparsified
block stays symmetric as smoothed aggregation and the Chebyshev smoother both require. The diagonal is
always kept, so no row can be emptied -- the sparsest configuration above still leaves 35 off-diagonals
in the emptiest row.

``symmetric``
"""""""""""""

Set ``symmetric=False`` for a single-field problem. With one field the reverse block sweep is applied to
identical input and returns an identical result, so it is a wasted AMG V-cycle per outer iteration.

``outerTol`` versus adaptive forcing
""""""""""""""""""""""""""""""""""""

This one has a trap in it. ``outerTol`` is ``None`` by default, which selects Eisenstat--Walker forcing
with ``etaMin``/``etaMax``. **Passing a number for** ``outerTol`` **bypasses forcing entirely** -- the
tolerance is then fixed at that value on every solve and ``etaMax`` has no effect whatsoever.

Forcing was tested against a fixed ``1e-4`` with ``etaMax=1e-2``, and it is **not** a reliable win:

.. list-table::
    :width: 100%
    :widths: 30 35 35
    :header-rows: 1

    * - problem size
      - outer iterations
      - ``linear solve``
    * - 43,350 DOF
      - -6%
      - +3%
    * - 70,800 DOF
      - **+65%**
      - **+61%**
    * - 120,960 DOF
      - -16%
      - -13%

At 70,800 DOF it also pushed the Newton loop from 5 iterations to 8: looser linear solves bought fewer
inner iterations and paid for them with more outer ones. Measure it on your problem before enabling it.

Settings measured to do nothing
*******************************

These were each tested at 43,350 and 70,800 DOF against the shipped configuration. They are recorded so
that time is not spent on them again.

.. list-table::
    :width: 100%
    :widths: 28 72
    :header-rows: 1

    * - Setting
      - Result
    * - ``useRigidBodyNullspace=False``
      - **Bit-identical** results -- reaction forces agreeing to all 16 digits. Inert on this operator.
    * - ``sweeps=2``
      - **Bit-identical** results, at **55-67% more time**. For one field the block sweep reapplies the
        same AMG to the same input, so extra sweeps are pure cost.
    * - ``eps_strong`` 0.04 / 0.25
      - Reaches AMGCL (the results do change) but gives no benefit: 629 / 628 / 607 outer iterations at
        43,350 DOF for 0.04 / 0.10 / 0.25, and 696 / 673 / 744 at 70,800.
    * - ``hierarchyStalenessFactor=1.0`` (eager refresh)
      - Iteration counts change by under 1.5% and cost 12% more time. Hierarchy staleness is not what
        makes these solves expensive: in the baseline, refreshed and reused solves have near-identical
        median iteration counts.
    * - ``hierarchyDropLumping=True``
      - Neutral at ``hierarchyDropTol=1e-4`` (-0.08%) and **8.1% slower** at ``1e-2``. This is the AMG
        literature's preferred filtering construction -- lumping each dropped entry onto the diagonal
        preserves row sums exactly, and it verifiably does so here (relative row-sum error 4.65e-02
        truncated against 3.11e-15 lumped at ``1e-2``). It simply buys nothing on this operator: outer
        GMRES time is unchanged, so convergence is identical, and the whole penalty is the extra pass
        itself, +32% on the hierarchy build. Consistent with ``useRigidBodyNullspace`` being inert --
        this preconditioner is insensitive to near-null-space fidelity.

Where the time actually goes
****************************

Only about a quarter to three quarters of the linear solve is the Krylov iteration, depending on the
support radius. At support 1.8, with iteration counts low, the split was:

.. list-table::
    :width: 100%
    :widths: 45 25 30
    :header-rows: 1

    * - Component
      - s / solve
      - share
    * - equilibration
      - 0.24356
      - 35%
    * - outer GMRES
      - 0.18928
      - 27%
    * - hierarchy build
      - 0.13960
      - 20%
    * - off-diagonal split
      - 0.08398
      - 12%

Two of those have since been addressed. The **off-diagonal split is now skipped automatically for a
single field** -- it used to copy the whole matrix into a dictionary that stayed empty, worth 5.8% of the
linear solve at 70,800 DOF. **Equilibration is now the largest non-Krylov cost** and is recomputed on
every solve.

If your iteration counts are low, do not assume the Krylov loop is where your time is. Read the
``blockamg:`` entries in the performance table before optimising anything.

Memory, and what limits problem size
************************************

For large models the binding constraint is usually memory, and usually a *transient* rather than the
steady state:

.. list-table::
    :width: 100%
    :widths: 40 30 30
    :header-rows: 1

    * - at 120,960 DOF
      - peak
      - during the solve
    * - single increment
      - 100.34 GiB
      - 88.89 GiB
    * - multi-increment run
      - 126.77 GiB
      - ~97 GiB

Practical consequences:

- **Use** ``systemMatrixAssembly="direct"`` **above ~50,000 DOF.** It is not only faster; the staging assembly
  path stores an int32 index per COO pair and therefore cannot build its gather map beyond 2.15e9 pairs,
  which this problem class reaches at about 50,000 DOF -- with only a third of a 187 GB machine in use.
- Assembly cost and memory then scale **linearly** in DOF, measured over a 2.8x range.
- Peak memory is set by setup transients (pattern construction, and rebuilding on a connectivity change),
  not by the assembly or the solve. Size a machine for the peak, and expect it to be well above what the
  solve itself occupies.

Diagnosing a slow solve
***********************

**Read the per-solve log lines.** ``blockamg`` prints one line per solve with its outer iteration
count, whether the hierarchy was refreshed or reused, the tolerance it was given, and the time taken. A
warning is emitted above ``warnOuterIterations`` outer iterations.

**Pair each solve with the residual it was asked to reduce.** Iteration counts vary strongly *within* an
increment -- from single digits early to several hundred late -- so a mean over an increment hides the
structure. This matters: an early attempt to characterise AMG at large DOF concluded it was degrading
with problem size, when in fact the expensive solves were simply the late-Newton ones, present in the same
proportion at every size. The iterations-against-residual curves at 43,350 and 120,960 DOF lie on top of
each other.

**Capture the hard systems.** ``dumpOnDegradationDir`` together with ``dumpOnDegradationThreshold``
writes the matrix and right-hand side of any solve exceeding a given iteration count, for offline
analysis. That is how the operator table above was produced.

Pitfalls when measuring
***********************

Four traps, each of which produced a wrong conclusion before being caught.

**Divide iteration totals by the number of solves.** Two configurations rarely take the same number of
Newton iterations, so summed outer-iteration counts are not comparable. A drop-tolerance sweep initially
appeared to reduce iterations by 10%; per solve it was flat.

**Establish a noise floor, and match it to what you are varying.** For this problem class at 70,800 DOF:
two runs of the *same binary* differ by **0.13%** in outer iterations and **9.8e-13** in the exported
reaction force. Two *different builds* of nominally equivalent code differ by about **2%** in iteration
count. So a floor measured by rerunning one binary does not license conclusions about a code change, and
neither floor is the right comparator for a *preconditioner* change, where the Newton path necessarily
differs.

**A preconditioner change cannot be validated by comparing solutions.** It changes the iterate path by
construction. What must hold is that every increment still meets the same convergence criterion, and that
the cutback count does not grow.

**Do not trust a percentage measured in another regime.** The off-diagonal split was 12% of the linear
solve at support 1.8 and 5.8% at support 2.2, because the Krylov loop's share of the total is completely
different in the two cases.

A worked configuration
**********************

For an RKPM impact problem above the crossover, on the deformed states where the work actually happens:

.. code-block:: python

    from edelweissfe.config import registry
    from edelweissfe.config.linsolve import LinearSolverDefinition
    from edelweissfe.utils.schema import buildSchemaFromOptions

    blockamg, blockamgSchema = registry.lookup("linsolver", "blockamg")
    jobInfo["linearSolvers"] = {
        "amg": LinearSolverDefinition(
            blockamg,
            buildSchemaFromOptions(
                blockamgSchema,
                {
                    "tolerance": 1e-4,  # fixed; note this bypasses Eisenstat--Walker forcing entirely
                    "symmetric": False,  # the reverse sweep is a wasted V-cycle for a single field
                    "dropTolerance": 1e-4,  # ~1.8x on the linear solve; verify on your own operator first
                },
            ),
        )
    }

    solver = NISTParallel(
        jobInfo,
        journal,
        linsolver="amg",
        **meshfreeCriterionOptions | {"systemMatrixAssembly": "direct"},  # required above ~50,000 DOF
    )

Then check three things in the performance table before tuning further: the share of the linear solve
that is actually ``blockamg: outer GMRES``, the outer iteration counts on the *late* Newton iterations of
an increment, and whether peak memory is being set by a setup transient rather than by the solve.

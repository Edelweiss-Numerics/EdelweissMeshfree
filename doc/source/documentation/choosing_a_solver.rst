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
    * - ``allowFallBackToRestart``, ``numberOfRestartsToStore``
      - step option ``maxCheckpointFallbacks``, which also bounds the number of fallbacks, with a restart output
        (its ``numberOfFilesToKeep``)
    * - ``skipZeroIncrements`` (it kept the zero increment and skipped only its solve)
      - EdelweissFE's implicit dynamic solver skips the solve of a zero-length increment by itself; for a quasi-static
        solver, a time stepper without the zero increment (``makeZeroIncrementFirst=False``). Neither applies to
        ``MeshfreeExplicitDynamic``: its step starts with the zero increment, which evaluates the forces at rest

**Restart.** EdelweissFE's restart output manager writes a checkpoint of the converged state; an ``MPMModel`` adds the
state of its material points and particles, from which their positions follow (see
:meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.writeRestart`), and the state of the output managers of the step.
A resumed run builds the model as the checkpointed run did, restores its state, and continues the step:

.. code-block:: python

    from edelweissfe.outputmanagers.restart import OutputManager as RestartOutputManager
    from edelweissfe.outputmanagers.restart import RestartOutputManagerSchema
    from edelweissfe.utils.checkpoint import ResumeCheckpoint

    restartOutput = RestartOutputManager(
        "restart", model, fieldOutputController, journal, None,
        configuration=RestartOutputManagerSchema(writeInterval=1, baseName="restart"),
    )
    outputManagers = [ensightOutput, restartOutput]

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

Explicit dynamics
-----------------

An RKPM model in explicit dynamics is solved by
:class:`~edelweissmeshfree.solvers.meshfreeexplicitdynamic.MeshfreeExplicitDynamic`, an EdelweissFE solver built into an
EdelweissFE step like the implicit ones. It integrates particles only; finite elements, material points, cell elements
and constraints with Lagrange multipliers are refused. So is a constraint that does not act in an explicit analysis
(``actsInExplicitAnalysis``, see :doc:`constraints`): the constraints are evaluated at the configuration just accepted,
where the nodes of the kernel functions carry neither a solution nor an increment, so a weak Dirichlet condition or an
equal value on the increment would act as if nothing had moved. The penalty contacts act, explicit or not.

.. code-block:: python

    from edelweissfe.steps.adaptivestep import AdaptiveStep
    from edelweissfe.steps.stepmanager import StepActionCollection

    from edelweissmeshfree.solvers.meshfreeexplicitdynamic import MeshfreeExplicitDynamic

    model.particleManagers.append(particleManager)
    stepActions = StepActionCollection()
    stepActions["distributedload"][pressure.name] = pressure

    step = AdaptiveStep(
        0, model, fieldOutputController, journal, {}, MeshfreeExplicitDynamic({}, journal), outputManagers, stepActions,
        stepLength=tMax, startInc=increment, maxInc=increment, minInc=increment, maxNumInc=1000,
    )
    step.solve()

Why it is not EdelweissFE's explicit solver ``NED``
***************************************************

Both integrate with the same leapfrog: the velocity half an increment behind the displacement, divided by a lumped mass,
the first increment of motion a half step. They differ in what they integrate. ``NED`` advances nodes that carry their
total solution on a fixed mesh: an element advances its state and returns its force in one call, the lumped mass is
assembled once per mesh, and a stable increment follows from the element wave speeds. The kernel functions of an RKPM
model move with its particles, and the state lives in the particles. Every increment therefore

1. advances the particles on the kernel functions they were assigned at its start,
2. lets the model assign them to the kernel functions they reach now, and
3. evaluates the internal forces, the lumped mass and the momentum on the new kernel functions.

The lumped mass is assembled anew in every increment, and the nodal values are coefficients of the current kernel
functions. ``NED``'s stable increment, damping, multi-point constraints, Dirichlet conditions by prescribed velocity and
energy balance need what particles do not provide, so the increment is chosen by the step.

What to know
************

- **The zero increment.** The first increment of a step must be the zero increment (the default of a step): it
  evaluates the forces at rest and projects the initial velocity from the particles' momentum.
- **The velocity.** It is carried from increment to increment while the equation system numbers the same nodes, which
  holds when the particle managers bond the kernel functions to the particles. Otherwise, and in every increment with
  the option ``projectVelocityEveryIncrement=True``, it is projected from the particles' momentum, which dissipates more.
- **The equation system.** When the active nodes cannot change (bonded kernel functions, no material point managers),
  a change of the connectivity only locates the changed particles and the constraints anew; otherwise the equation
  system is rebuilt.
- **Nodes without mass.** A kernel function that the particles reach only where it vanishes has no positive lumped
  mass; its node has no equation of motion and keeps a zero velocity.
- **A particle that cannot take an increment** ends the step with ``StepFailed``: the increment is set by stability,
  not by convergence, so it is not cut back.
- **Pressure.** A particle pressure is assembled as in implicit runs: as for the finite elements, a positive pressure
  pushes on the face, against its outward normal.
- **Restart** works as for the implicit solvers (see above); the checkpoint carries the half-step velocity, the net
  force and the lumped mass.

Choosing a linear solver
------------------------

Every Newton iteration of an implicit meshfree simulation spends most of its time in one linear solve, so the choice of
linear solver and its settings usually matters more than anything else you can tune. This section collects what was
learned on RKPM problems -- what helps, what turned out not to matter -- and how to find out for your own model.

.. note::

   The advice was gathered on one problem class: an impact problem discretized with RKPM particles of a large support,
   on deformed states with active penalty contact. The support radius changes the behaviour of the linear solvers more
   than anything else (see :ref:`the operator <rkpm-operator-properties>`), so treat the advice as a hypothesis to check
   on a different discretization.

Start here
**********

.. list-table:: A first choice
    :width: 100%
    :widths: 30 30 40
    :header-rows: 1

    * - Situation
      - Use
      - Why
    * - small and medium models
      - ``pardiso`` with ``reuseSymbolicFactorization=True``
      - a direct factorization is fastest while its fill-in stays moderate, and the symbolic reuse saves most of its
        setup
    * - large models
      - ``blockamg``
      - the iteration count of AMG stays nearly constant as the model grows; the cost of a direct factorization grows
        faster than the model
    * - models whose staging assembly exceeds its pair counter
      - ``blockamg``, and ``systemMatrixAssembly="direct"`` on the solver
      - the staging assembly cannot index more than ``INT32_MAX`` contributions (see :doc:`assembly`)
    * - memory bound rather than time bound
      - ``blockamg`` + ``systemMatrixAssembly="direct"``
      - the fill-in of a direct factorization is the one memory term that is not bounded by the model size

Where the crossover between a direct and an iterative solver lies depends on the discretization: measure both on a
representative state of your model.

The direct solver
*****************

``reuseSymbolicFactorization``
    **Turn this on** for a Newton sequence. The sparsity pattern is the same in all iterations of an increment, so the
    reordering and the symbolic factorization are computed once instead of in every solve. The solutions agree to
    round-off.

Threading
    The direct solve stops scaling well beyond a moderate number of threads. Do not size a machine for a direct solver
    on its thread count alone.

.. _rkpm-operator-properties:

The operator is the thing
*************************

Before tuning a preconditioner, know what the RKPM operator looks like, because it does not resemble the sparse
operators AMG was designed for:

- **Structurally dense:** every row has as many stored entries as there are kernel functions in reach of the particles
  around its node -- with a large support, a large multiple of what a finite element mesh gives.
- **Numerically sparse:** the typical stored off-diagonal entry is orders of magnitude smaller than the diagonal; only a
  few entries per row carry weight.
- **Symmetric** to round-off, with a **mild spread** of the diagonal.

Two consequences follow, and both are counter-intuitive.

**Penalty contact is not the problem.** It does not create the scale separation that would confuse a
strength-of-connection criterion; the diagonal spread stays mild.

**The operator is numerically far sparser than it is structurally.** Smoothed aggregation weighs every stored entry
where a handful matter, and the Galerkin product carries them all down to the coarse levels. This is the most useful
thing to know about tuning AMG here, and it is what ``dropTolerance`` exploits.

**The support radius dominates.** A larger support costs many more outer iterations with the same preconditioner. If
the iteration counts are unacceptable, the discretization is a bigger lever than the solver -- but the support radius is
an accuracy decision, not a free one: the solution changes with it.

Settings that matter
********************

``dropTolerance`` -- the largest single win
"""""""""""""""""""""""""""""""""""""""""""

Builds each AMG hierarchy from a copy of the matrix without the off-diagonal entries :math:`a_{ij}` with
:math:`|a_{ij}| < \tau \sqrt{|a_{ii}| |a_{jj}|}`, :math:`\tau` = ``dropTolerance``. Only the *preconditioner* is
sparsified: the operator the Krylov method applies, and therefore the residual it drives to zero, is untouched. **A
change of the preconditioner cannot move the converged solution beyond the outer tolerance** -- it can only change how
many iterations reaching it takes.

On the RKPM operator, the outer iterations per solve stay flat over orders of magnitude of ``dropTolerance``, while the
hierarchy keeps only a small fraction of the entries. The whole gain is a cheaper iteration, not a faster convergence.

The default is ``0.0`` (off): the point where the quality of the preconditioner breaks down was never reached, so the
safe end of the range is unknown, and it was explored on one operator only. Start in the middle of the range that was
tried, e.g. ``1e-4``, and check the iteration counts per solve on your model.

Use plain truncation. Adding the dropped entries to the diagonal (``lumpDroppedEntries``), so that the row sums are kept
exactly, buys nothing on this operator and costs an extra pass over the matrix. Together with the rigid body modes and
the strength-of-connection threshold having no effect (below), this says that **near-null-space and
strength-of-connection fidelity are not what limits this preconditioner**; the cost of applying a hierarchy built from a
numerically dense operator is, and the truncation removes it.

The criterion is scale invariant and symmetric in :math:`(i, j)`, so the sparsified block stays symmetric, as smoothed
aggregation and the Chebyshev smoother require. The diagonal is always kept, so no row can be emptied.

``symmetric``
"""""""""""""

Set ``symmetric=False`` for a single-field problem. With one field the reverse block sweep is applied to identical input
and returns an identical result, so it is a wasted AMG cycle per outer iteration.

``tolerance`` versus adaptive forcing
"""""""""""""""""""""""""""""""""""""

This one has a trap. ``tolerance`` is unset by default, which selects Eisenstat--Walker forcing. **Passing a number for**
``tolerance`` **switches forcing off entirely**: the tolerance is then the same in every solve.

Forcing is **not** a reliable win here. Looser linear solves early in the Newton loop buy fewer inner iterations, and
can pay for them with more Newton iterations; depending on the model size, forcing came out ahead or clearly behind a
fixed tolerance. Measure it on your problem before relying on it.

Settings that did nothing
*************************

These were each tested against the configuration above, and are recorded so that time is not spent on them again:

- **The rigid body modes as near-null-space** (``rigidBodyModesPerAggregate``): results identical to the last digit;
  inert on this operator.
- **More block sweeps** (``sweeps=2``): identical results at a large extra cost. For one field a further sweep applies
  the same AMG to the same input.
- **The strength-of-connection threshold:** changes the results, but not the iteration counts in any consistent
  direction.
- **Refreshing the hierarchy eagerly** instead of reusing it (``reusePreconditioner``): the iteration counts barely
  change, and the time grows. Reused and refreshed solves need about the same number of iterations.
- **Lumping the dropped entries** (``lumpDroppedEntries``): see above; it keeps the row sums exactly, and changes
  nothing else but the cost of the build.

Where the time goes
*******************

Depending on the support radius, the Krylov iteration is only a part of the linear solve. With low iteration counts the
equilibration of the matrix, the build of the hierarchy and the split into fields can together cost more than the outer
iterations. The split is skipped for a single field; the equilibration is computed in every solve.

If your iteration counts are low, do not assume the Krylov loop is where your time is. Read the ``blockamg`` entries of
the performance table before optimizing anything.

Memory, and what limits the model size
**************************************

For large models the binding constraint is usually memory, and usually a *transient* rather than the steady state:

- **Use** ``systemMatrixAssembly="direct"`` once the staging assembly approaches its pair counter (see :doc:`assembly`).
  Assembly cost and memory then grow linearly with the model.
- The peak memory is set by setup transients -- the construction of the CSR pattern, and its reconstruction when the
  connectivity changes -- not by the assembly or the solve. Size a machine for the peak, and expect it to lie well above
  what the solve itself occupies.

Diagnosing a slow solve
***********************

**Read the per-solve log lines.** ``blockamg`` prints one line per solve with its outer iteration count, whether the
hierarchy was refreshed or reused, the tolerance it was given, and the time taken.

**Pair each solve with the residual it was asked to reduce.** The iteration counts vary strongly *within* an increment,
from few early to many in the last Newton iterations, so a mean over an increment hides the structure. An early attempt
to characterize AMG on large models concluded that it degraded with the model size, when in fact the expensive solves
were the late Newton ones, present in the same proportion at every size: the curves of iterations against residual of
small and large models lie on top of each other.

Pitfalls when measuring
***********************

**Divide iteration totals by the number of solves.** Two configurations rarely take the same number of Newton
iterations, so summed outer iteration counts are not comparable.

**Establish a noise floor, and match it to what you vary.** Two runs of the same build differ slightly in iteration
counts; two builds of nominally equivalent code differ more. A floor measured by rerunning one build does not license
conclusions about a code change, and neither floor is the right comparator for a *preconditioner* change, where the
Newton path necessarily differs.

**A preconditioner change cannot be validated by comparing solutions.** It changes the path of the iterates by
construction. What must hold is that every increment still meets the same convergence criterion, and that the number of
cutbacks does not grow.

**Do not trust a share measured in another regime.** The share of a component of the linear solve depends on how much
of the total the Krylov loop takes, which changes completely with the support radius.

A worked configuration
**********************

For an RKPM problem above the crossover, on the deformed states where the work actually happens:

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
                    "tolerance": 1e-4,  # fixed; note this switches Eisenstat--Walker forcing off
                    "symmetric": False,  # the reverse sweep is a wasted cycle for a single field
                    "dropTolerance": 1e-4,  # cheaper iterations; verify on your own operator first
                },
            ),
        )
    }

    solver = NISTParallel(
        jobInfo,
        journal,
        linsolver="amg",
        **meshfreeCriterionOptions | {"systemMatrixAssembly": "direct"},
    )

Then check three things in the performance table before tuning further: the share of the linear solve that is actually
the outer Krylov iteration, the outer iteration counts in the *late* Newton iterations of an increment, and whether the
peak memory is set by a setup transient rather than by the solve.

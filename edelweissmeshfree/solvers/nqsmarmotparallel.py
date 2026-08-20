# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _
# | ____|__| | ___| |_      _____(_)___ ___
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/
# |  \/  | ___  ___| |__  / _|_ __ ___  ___
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \
# | |  | |  __/\__ \ | | |  _| | |  __/  __/
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
#
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------
import os
from multiprocessing import cpu_count
from time import perf_counter

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix

from edelweissmeshfree.solvers.base.parallelization import (
    computeMarmotCellsInParallel,
    computeMarmotMaterialPointsInParallel,
    computeMarmotParticlesInParallel,
    computeMarmotParticlesIntoCSR,
)
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver


class NQSParallelForMarmot(NonlinearQuasistaticSolver):
    """This is a parallel implemenntation of the NonlinearQuasistaticSolver.
    It only works with MarmotCells and MarmotElements, as it directly accesses and exploits
    the background Marmot C++ objects.

    It uses Cython/OpenMP for evaluation those MarmotCells and MarmotMaterialPoints in a prange loop,
    allowing to bypass the GIL and get decent performance.

    The number of threads for the OpenMP loop is determined based on the cpu count,
    or (higher priority) based on the environment variable OMP_NUM_THREADS.

    Parameters
    ----------
    journal
        The Journal instance for loggin purposes.
    verifyDirectCSRAssembly
        If True, every iteration assembles the particle contributions *twice* -- once through the
        VIJ staging array, which stays the path the solve actually uses, and once straight into CSR
        -- and compares the two results. Diagnostic only, and it roughly doubles the particle
        assembly cost. See :meth:`_computeParticlesAndVerifyDirectCSR`.
    """

    identification = "NQSParallelForMarmot"

    #: Emit a per-entry breakdown of any disagreement: how many entries differ and by how much.
    #: The distinction that matters is a handful of entries off by O(1) -- a structural bug -- versus
    #: every entry off by a little, which points at the arithmetic instead.
    directCSRDiagnostics = True

    #: Floor below which a relative difference between assembly paths is not interpreted at all.
    directCSRVerificationTolerance = 1e-12

    #: Relative agreement demanded of the ``scatter`` comparison, which sums identical values in two
    #: different orders. Rounding on a reduction of this size lands near 1e-16; anything at 1e-13 or
    #: above is a misaddressed row, column or transposition rather than accumulated error.
    directCSRScatterTolerance = 1e-13

    #: How far above the measured re-evaluation noise floor the fused path may land before it counts
    #: as a real disagreement. Re-evaluating a particle is not bit-reproducible, so the fused kernel
    #: -- which re-evaluates -- can only be held to that floor, not to zero. Its *addressing* is
    #: pinned separately and exactly by the ``scatter`` comparison, which involves no re-evaluation.
    directCSRNoiseFactor = 4.0

    #: Repeats per iteration in the assembly benchmark. Both paths are run every repeat and the
    #: order is alternated, so a warm-up or a drifting machine biases neither.
    directCSRTimingRepeats = 5

    def __init__(
        self,
        journal: Journal,
        verifyDirectCSRAssembly: bool = False,
        timeDirectCSRAssembly: bool = False,
    ):
        self.numThreads = cpu_count()

        if "OMP_NUM_THREADS" in os.environ:
            self.numThreads = int(os.environ["OMP_NUM_THREADS"])

        self.verifyDirectCSRAssembly = verifyDirectCSRAssembly
        self.timeDirectCSRAssembly = timeDirectCSRAssembly

        super().__init__(journal)

    @performancetiming.timeit("computation material points")
    def _computeMaterialPoints(self, materialPoints_, time: float, dT: float):
        return computeMarmotMaterialPointsInParallel(materialPoints_, time, dT, self.numThreads)

    @performancetiming.timeit("computation active cells")
    def _computeCells(
        self,
        activeCells_: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        return computeMarmotCellsInParallel(
            activeCells_,
            dU,
            P,
            F,
            K_VIJ,
            time,
            dT,
            theDofManager,
            self.numThreads,
        )

    @performancetiming.timeit("computation particles")
    def _computeParticles(
        self,
        particles_: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        if self.verifyDirectCSRAssembly:
            return self._computeParticlesAndVerifyDirectCSR(
                particles_, dU, P, F, K_VIJ, time, dT, theDofManager
            )

        if self.timeDirectCSRAssembly:
            return self._computeParticlesAndTimeDirectCSR(
                particles_, dU, P, F, K_VIJ, time, dT, theDofManager
            )

        return computeMarmotParticlesInParallel(
            particles_,
            dU,
            P,
            F,
            K_VIJ,
            time,
            dT,
            theDofManager,
            self.numThreads,
        )

    @performancetiming.timeit("verification direct csr")
    def _computeParticlesAndVerifyDirectCSR(
        self,
        particles_: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Assemble the particles several ways on identical state, and cross-check the results.

        Why this lives in the solver rather than in a unit test: evaluating a particle needs a fully
        prepared model -- kernel supports resolved, shape functions reconstructed, state variables
        allocated -- and hand-building that state is how three earlier attempts died, each one
        segfaulting inside Marmot's shape-function evaluation. Decisively, the *existing* VIJ kernel
        segfaults identically in the same harness, which places the fault in the harness. Running
        here instead means the comparison sees exactly the state the production path sees.

        Three comparisons, because one alone cannot separate an addressing bug from evaluation noise:

        ``scatter``
            The blocks the VIJ path just produced, pushed through :meth:`scatterBlock` entity by
            entity, against the same blocks pushed through ``assembleFromVIJ``. Identical numbers
            through the identical offset map, so this isolates the addressing from any physics noise
            -- it is the test of the entity registration and of the block layout. It agrees to
            *rounding*, not bit-for-bit: the two sum the same contributions in different orders
            (entity by entity on one thread, versus pair order across sixteen), so a difference at
            the 1e-16 relative level is expected and a difference above it is an addressing error.
        ``noise``
            The VIJ kernel run a *second* time on the same state, against its own first result. This
            is the floor: re-evaluating a particle is not guaranteed to reproduce bit-for-bit, and
            without measuring that floor there is no way to read the third number.
        ``fused``
            The direct-to-CSR kernel, against the first VIJ result. It re-evaluates the physics, so
            it can only ever be as good as ``noise``. Judged against that floor, not against zero.

        A wrong row, column or transposition in the fused scatter shows up as an O(1) relative
        difference, which no amount of evaluation noise can disguise.

        Parameters
        ----------
        particles_
            The particles to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """

        assembler = self._directCSRAssembler
        particles = list(particles_)
        entityIds = np.array([self._directCSREntityIds[p] for p in particles], dtype=np.intc)

        VBefore = np.array(K_VIJ, copy=True)
        PBefore = np.array(P, copy=True)

        # ---- evaluation 1: the production path, untouched -------------------------------------
        computeMarmotParticlesInParallel(particles_, dU, P, F, K_VIJ, time, dT, theDofManager, self.numThreads)

        # cells and elements may have written before us, so the particle contribution is the delta
        VParticles = np.ascontiguousarray(np.asarray(K_VIJ) - VBefore)
        PParticles = np.asarray(P) - PBefore

        staged = np.array(assembler.assembleFromVIJ(VParticles).data, copy=True)

        # ---- comparison "scatter": same values, entity-by-entity through scatterBlock ----------
        assembler.beginAssembly()
        for particle in particles:
            start = theDofManager.idcsOfHigherOrderEntitiesInVIJ[particle]
            nDof = particle.nDof
            assembler.scatterBlock(0, self._directCSREntityIds[particle], VParticles[start : start + nDof * nDof])
        scattered = np.array(assembler.reduce().data, copy=True)

        errScatter = np.abs(scattered - staged).max()

        # ---- comparison "noise": the same kernel, evaluated a second time ----------------------
        VSecond = theDofManager.constructVIJSystemMatrix()
        VSecond[:] = 0.0
        PSecond = theDofManager.constructDofVector()
        FSecond = theDofManager.constructDofVector()
        computeMarmotParticlesInParallel(
            particles_, dU, PSecond, FSecond, VSecond, time, dT, theDofManager, self.numThreads
        )
        noise = np.array(assembler.assembleFromVIJ(np.ascontiguousarray(np.asarray(VSecond))).data, copy=True)

        # ---- comparison "fused": the direct-to-CSR kernel ---------------------------------------
        PFused = theDofManager.constructDofVector()
        FFused = theDofManager.constructDofVector()
        assembler.beginAssembly()
        computeMarmotParticlesIntoCSR(
            particles_,
            dU,
            PFused,
            FFused,
            entityIds,
            assembler.corePointer,
            time,
            dT,
            theDofManager,
            self.numThreads,
        )
        fused = np.array(assembler.reduce().data, copy=True)

        # ---- comparison "fused1": the same kernel on one thread -------------------------------
        # Discriminates a threading or privatisation fault from a fault in the scratch buffer: if one
        # thread reproduces the VIJ result and sixteen do not, the loop is at fault, not the buffer.
        PFused1 = theDofManager.constructDofVector()
        FFused1 = theDofManager.constructDofVector()
        assembler.beginAssembly()
        computeMarmotParticlesIntoCSR(
            particles_, dU, PFused1, FFused1, entityIds, assembler.corePointer, time, dT, theDofManager, 1
        )
        fused1 = np.array(assembler.reduce().data, copy=True)

        scaleK = np.abs(staged).max()
        relScatter = errScatter / scaleK
        relNoise = np.abs(noise - staged).max() / scaleK
        relFused = np.abs(fused - staged).max() / scaleK

        scaleP = np.abs(PParticles).max()
        relNoiseP = np.abs(np.asarray(PSecond) - PParticles).max() / scaleP
        relFusedP = np.abs(np.asarray(PFused) - PParticles).max() / scaleP

        # one comparison per line: the journal wraps, and a single line wide enough to hold all six
        # numbers loses its tail exactly when the numbers matter
        self.journal.message(
            "direct CSR check: {:} nnz, |K| {:.3e}, |P| {:.3e}".format(fused.shape[0], scaleK, scaleP),
            self.identification,
            level=1,
        )
        self.journal.message(
            "  scatter: K rel {:.3e} (addressing, expected ~1e-16)".format(relScatter),
            self.identification,
            level=1,
        )
        self.journal.message(
            "  noise  : K rel {:.3e}, P rel {:.3e} (re-evaluation floor)".format(relNoise, relNoiseP),
            self.identification,
            level=1,
        )
        self.journal.message(
            "  fused  : K rel {:.3e}, P rel {:.3e} ({:} threads)".format(relFused, relFusedP, self.numThreads),
            self.identification,
            level=1,
        )
        self.journal.message(
            "  fused1 : K rel {:.3e}, P rel {:.3e} (1 thread)".format(
                np.abs(fused1 - staged).max() / scaleK,
                np.abs(np.asarray(PFused1) - PParticles).max() / scaleP,
            ),
            self.identification,
            level=1,
        )

        if self.directCSRDiagnostics:
            self._reportDirectCSRDifference("K", staged, fused)
            self._reportDirectCSRDifference("P", PParticles, np.asarray(PFused))

        if relScatter > self.directCSRScatterTolerance:
            raise RuntimeError(
                "direct CSR scatter disagrees with assembleFromVIJ on identical values: rel {:.3e}."
                " Both sum the same contributions, so this is an addressing error.".format(relScatter)
            )

        # judged against the re-evaluation floor, with an absolute tolerance for the case where
        # re-evaluation happens to be exact and the floor is therefore zero
        limitK = max(self.directCSRVerificationTolerance, self.directCSRNoiseFactor * relNoise)
        limitP = max(self.directCSRVerificationTolerance, self.directCSRNoiseFactor * relNoiseP)

        if relFused > limitK or relFusedP > limitP:
            raise RuntimeError(
                "direct CSR assembly exceeds the re-evaluation noise floor: K rel {:.3e} (limit"
                " {:.3e}), P rel {:.3e} (limit {:.3e})".format(relFused, limitK, relFusedP, limitP)
            )

    def _reportDirectCSRDifference(self, label, reference, candidate, nWorst: int = 3):
        """Report how a candidate array differs from a reference, entry by entry.

        Parameters
        ----------
        label
            Name of the quantity, for the log line.
        reference
            The values the VIJ path produced.
        candidate
            The values the direct-to-CSR path produced.
        nWorst
            How many of the largest relative differences to list.
        """

        reference = np.asarray(reference)
        candidate = np.asarray(candidate)

        diff = np.abs(candidate - reference)
        scale = np.maximum(np.abs(reference), np.abs(candidate))
        rel = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0.0)

        nDiffering = int((rel > 1e-14).sum())
        self.journal.message(
            "  {:} diag: {:}/{:} entries differ, {:.4f}% ".format(
                label, nDiffering, rel.shape[0], 100.0 * nDiffering / rel.shape[0]
            ),
            self.identification,
            level=1,
        )

        if nDiffering == 0:
            return

        for k in np.argsort(rel)[::-1][:nWorst]:
            self.journal.message(
                "    [{:}] ref {: .8e} cand {: .8e} rel {:.3e}".format(k, reference[k], candidate[k], rel[k]),
                self.identification,
                level=1,
            )

    @performancetiming.timeit("benchmark direct csr")
    def _computeParticlesAndTimeDirectCSR(
        self,
        particles_: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Benchmark the two assembly paths against each other on identical state.

        Both paths are timed component by component, every repeat, in the same process and on the
        same increment, with the order alternated between repeats. That is deliberate: an earlier
        attempt at this comparison used a separate harness which sliced blocks out of the VIJ array
        with numpy, and the resulting 3.83 GB of streaming inverted the ranking of the two. Timing
        both paths side by side against the state the solver actually has removes the harness from
        the comparison entirely.

        The components, and why each belongs to the path it is charged to:

        VIJ path
            ``zero`` the whole staging array must be cleared every iteration, and particles are what
            make it large, so the memset is charged here rather than treated as overhead.
            ``eval`` the particle loop, writing each block into its slab.
            ``gather`` ``CSRGenerator.updateCSR``, the production gather.

        direct path
            ``begin`` clearing the private CSR buffers.
            ``eval`` the particle loop, writing into a cached scratch block and scattering it.
            ``reduce`` summing the private buffers.

        Medians are reported, with the spread, because the scatter was previously measured varying by
        +-18% run to run -- wide enough that a single sample cannot support a ratio.

        The solve continues on the VIJ path: K_VIJ is restored to its pre-benchmark contents and the
        production evaluation is run last, so the increment is unaffected apart from taking longer.
        """

        assembler = self._directCSRAssembler
        particles = list(particles_)
        entityIds = np.array([self._directCSREntityIds[p] for p in particles], dtype=np.intc)

        # cells and elements may already have contributed, and the benchmark clears the array
        VBefore = np.array(K_VIJ, copy=True)

        PDiscard = theDofManager.constructDofVector()
        FDiscard = theDofManager.constructDofVector()

        samples = {k: [] for k in ("zero", "vijEval", "gather", "begin", "fusedEval", "reduce")}

        def timeVIJ():
            t0 = perf_counter()
            K_VIJ[:] = 0.0
            t1 = perf_counter()
            computeMarmotParticlesInParallel(
                particles_, dU, PDiscard, FDiscard, K_VIJ, time, dT, theDofManager, self.numThreads
            )
            t2 = perf_counter()
            self._VIJtoCSR(K_VIJ, self._csrGenerator, useInPlace=True)
            t3 = perf_counter()
            samples["zero"].append(t1 - t0)
            samples["vijEval"].append(t2 - t1)
            samples["gather"].append(t3 - t2)

        def timeFused():
            t0 = perf_counter()
            assembler.beginAssembly()
            t1 = perf_counter()
            computeMarmotParticlesIntoCSR(
                particles_,
                dU,
                PDiscard,
                FDiscard,
                entityIds,
                assembler.corePointer,
                time,
                dT,
                theDofManager,
                self.numThreads,
            )
            t2 = perf_counter()
            assembler.reduce()
            t3 = perf_counter()
            samples["begin"].append(t1 - t0)
            samples["fusedEval"].append(t2 - t1)
            samples["reduce"].append(t3 - t2)

        for repeat in range(self.directCSRTimingRepeats):
            if repeat % 2 == 0:
                timeVIJ()
                timeFused()
            else:
                timeFused()
                timeVIJ()

        def med(key):
            return float(np.median(samples[key]))

        def spread(key):
            v = samples[key]
            return (max(v) - min(v)) / np.median(v) if np.median(v) > 0.0 else 0.0

        vijTotal = med("zero") + med("vijEval") + med("gather")
        fusedTotal = med("begin") + med("fusedEval") + med("reduce")

        gib = 1024.0**3
        self.journal.message(
            "assembly benchmark, {:} repeats, {:} threads, {:} particles |"
            " VIJ array {:.2f} GiB, assembler {:.2f} GiB".format(
                self.directCSRTimingRepeats,
                self.numThreads,
                len(particles),
                np.asarray(K_VIJ).nbytes / gib,
                assembler.memoryBytes / gib,
            ),
            self.identification,
            level=1,
        )
        for label, key in (
            ("VIJ   zero  ", "zero"),
            ("VIJ   eval  ", "vijEval"),
            ("VIJ   gather", "gather"),
            ("direct begin", "begin"),
            ("direct eval ", "fusedEval"),
            ("direct reduc", "reduce"),
        ):
            self.journal.message(
                "  {:} {:8.4f} s  (spread {:5.1%})".format(label, med(key), spread(key)),
                self.identification,
                level=1,
            )
        self.journal.message(
            "  TOTAL  VIJ {:8.4f} s   direct {:8.4f} s   speedup {:.3f}x".format(
                vijTotal, fusedTotal, vijTotal / fusedTotal if fusedTotal > 0.0 else float("nan")
            ),
            self.identification,
            level=1,
        )

        # restore what the benchmark clobbered, then do the real evaluation
        np.asarray(K_VIJ)[:] = VBefore
        return computeMarmotParticlesInParallel(
            particles_, dU, P, F, K_VIJ, time, dT, theDofManager, self.numThreads
        )

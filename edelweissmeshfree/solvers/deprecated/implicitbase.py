from abc import abstractmethod
from typing import Iterable

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    getThreadPool,
    isFreeThreadingSupported,
)
from edelweissfe.numerics.systemmatrixassembly import (
    SystemMatrixAssembly,
    VIJSystemMatrixAssembly,
)
from edelweissfe.solvers.base.parallelelementcomputation import chunked_iterable
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.stepactions.base.loadbase import LoadBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import DivergingSolution
from numpy import ndarray
from scipy.sparse import csr_matrix

from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle
from edelweissmeshfree.solvers.deprecated.nonlinearsolverbase import BaseNonlinearSolver


class BaseNonlinearImplicitSolver(BaseNonlinearSolver):

    @abstractmethod
    def _newtonSolve(
        self,
        dirichlets: list[DirichletBase],
        bodyLoads: list,
        distributedLoads: list,
        particleDistributedLoads: list,
        elements: list,
        Un: DofVector,
        activeCells: list,
        cellElements: list,
        materialPoints: list,
        particles: list,
        constraints: list,
        theDofManager: DofManager,
        linearSolver,
        iterationOptions: dict,
        timeStep: TimeStep,
        model: MPMModel,
        newtonCache: tuple = None,
    ) -> tuple[DofVector, DofVector, dict, tuple]:
        pass

    @performancetiming.timeit("assemble loads")
    def _assembleLoads(
        self,
        loads: list[LoadBase],
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        K: SystemMatrixAssembly,
        timeStep: TimeStep,
        theDofManager: DofManager,
    ):
        """Assemble the residual contributions of all loads, the negative of the external loads, and their tangents:
        every load knows where it acts -- on cells, on material points, on particle faces -- and assembles itself, see
        :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.assembleLoad`.

        Parameters
        ----------
        loads
            The loads, in the order in which their contributions are summed.
        U_np
            The current global solution vector.
        dU
            The current global solution increment vector.
        P
            The load vector to be augmented.
        K
            Where each entity writes its stiffness contribution.
        timeStep
            The current time increment.
        theDofManager
            The DofManager instance.
        """

        for load in loads:
            load.assembleLoad(P, K, U_np, dU, timeStep, theDofManager)

    @performancetiming.timeit("dirichlet on CSR")
    def _applyDirichletKCsr(
        self,
        K: VIJSystemMatrix,
        dirichlets: list[DirichletBase],
        theDofManager: DofManager,
    ) -> VIJSystemMatrix:
        """Apply the dirichlet bcs on the global stiffness matrix
        Is called by solveStep() before solving the global system.
        http://stackoverflux.com/questions/12129948/scipy-sparse-set-row-to-zeros

        Parameters
        ----------
        K
            The system matrix.
        dirichlets
            The list of dirichlet boundary conditions.

        Returns
        -------
        VIJSystemMatrix
            The modified system matrix.
        """

        if dirichlets:
            for dirichlet in dirichlets:
                for row in self._findDirichletIndices(theDofManager, dirichlet):
                    K.data[K.indptr[row] : K.indptr[row + 1]] = 0.0

            # K[row, row] = 1.0 @ once, faster than within the loop above:
            diag = K.diagonal()
            diag[np.concatenate([self._findDirichletIndices(theDofManager, d) for d in dirichlets])] = 1.0
            K.setdiag(diag)

            # NOTE: the zeroed off-diagonals are deliberately kept as explicitly stored zeros.
            # This preserves the sparsity pattern, so the assembled CSR matrix can be updated
            # in place across iterations and direct solvers can reuse their symbolic
            # factorization. (Direct solvers handle stored zeros without issues.)

        return K

    @performancetiming.timeit("evaluation residuals")
    def _computeResiduals(
        self,
        R: DofVector,
        ddU: DofVector,
        dU: DofVector,
        F: DofVector,
        residualHistory: dict,
        theDofManager: DofManager,
    ) -> tuple[bool, dict]:
        """Compute the current residuals and relative flux residuals flux (R) and effort correction (ddU).

        Parameters
        ----------
        R
            The current residual.
        ddU
            The current correction increment.
        dU
            The current solution increment.
        F
            The accumulated fluxes.
        residualHistory
            The previous residuals.
        theDofManager
            The DofManager instance.

        Returns
        -------
        tuple[bool,dict]
            - True if converged.
            - The residual histories field wise.

        """

        if np.isnan(R).any():
            raise DivergingSolution("NaN obtained in residual.")

        spatialAveragedFluxes = self._computeSpatialAveragedFluxes(F, theDofManager)

        for field, fieldIndices in theDofManager.idcsOfFieldsInDofVector.items():
            fieldResidualAbs = np.abs(R[fieldIndices])

            indexOfMax = np.argmax(fieldResidualAbs)
            fluxResidualAbsolute = fieldResidualAbs[indexOfMax]

            fluxResidualRelative = fluxResidualAbsolute / max(spatialAveragedFluxes[field], 1e-16)
            nodeWithLargestResidual = theDofManager.getNodeForIndexInDofVector(indexOfMax)

            maxIncrement = np.linalg.norm(dU[fieldIndices], np.inf)
            correctionAbsolute = np.linalg.norm(ddU[fieldIndices], np.inf) if ddU is not None else 0.0
            correctionRelative = correctionAbsolute / max(maxIncrement, 1e-16)

            residualHistory[field].append(
                {
                    "absolute flux residual": fluxResidualAbsolute,
                    "spatial average flux": spatialAveragedFluxes[field],
                    "relative flux residual": fluxResidualRelative,
                    "max. increment": maxIncrement,
                    "absolute correction": correctionAbsolute,
                    "relative correction": correctionRelative,
                    "node with largest residual": nodeWithLargestResidual,
                }
            )

        return residualHistory

    @performancetiming.timeit("convergence check")
    def _checkConvergence(self, iterations: int, incrementResidualHistory: dict, iterationOptions: dict) -> bool:
        """Check the status of convergence.

        Parameters
        ----------
        iterations
            The current number of iterations.
        incrementResidualHistory
            The dictionary containing information about all residuals (and history) for the current Newton cycle.
        iterationOptions
            The dictionary containing settings controlling the convergence tolerances.

        Returns
        -------
        bool
            The truth value of convergence."""

        # iterationMessage = ""
        # convergedAtAll = True

        # iterationMessageTemplate = "{:11.2e}{:1}{:11.2e}{:1} "

        useStrictFluxTolerances = iterations < iterationOptions["iterations for alt. tolerances"]

        for field, fieldIncrementResidualHistory in incrementResidualHistory.items():
            lastResults = fieldIncrementResidualHistory[-1]
            correctionAbs = lastResults["absolute correction"]
            correctionRel = lastResults["relative correction"]

            fluxResidualAbs = lastResults["absolute flux residual"]
            fluxResidualRel = lastResults["relative flux residual"]
            spatialAveragedFlux = lastResults["spatial average flux"]

            if useStrictFluxTolerances:
                fluxTolRel = iterationOptions["spec. relative flux residual tolerances"].get(
                    field, iterationOptions["default relative flux residual tolerance"]
                )
            else:
                fluxTolRel = iterationOptions["spec. relative flux residual tolerances alt."].get(
                    field, iterationOptions["default relative flux residual tolerance alt."]
                )

            fluxTolAbs = iterationOptions["spec. absolute flux residual tolerances"].get(
                field, iterationOptions["default absolute flux residual tolerance"]
            )

            correctionTolRel = iterationOptions["spec. relative field correction tolerances"].get(
                field, iterationOptions["default relative field correction tolerance"]
            )

            correctionTolAbs = iterationOptions["spec. absolute field correction tolerances"].get(
                field, iterationOptions["default absolute field correction tolerance"]
            )

            nonZeroIncrement = lastResults["max. increment"] > iterationOptions["zero increment threshhold"]
            convergedCorrection = correctionRel < correctionTolRel if nonZeroIncrement else True
            convergedCorrection = convergedCorrection or correctionAbs < correctionTolAbs

            nonZeroFlux = spatialAveragedFlux > iterationOptions["zero flux threshhold"]
            convergedFlux = fluxResidualRel < fluxTolRel if nonZeroFlux else True
            convergedFlux = convergedFlux or fluxResidualAbs < fluxTolAbs

            if iterations == 0:
                convergedFlux = False

            fieldIncrementResidualHistory[-1]["converged flux"] = convergedFlux
            fieldIncrementResidualHistory[-1]["converged correction"] = convergedCorrection

            # iterationMessage += iterationMessageTemplate.format(
            #     fluxResidualAbs,
            #     "✓" if convergedFlux else " ",
            #     correctionAbs,
            #     "✓" if convergedCorrection else " ",
            # )
            # convergedAtAll = convergedAtAll and convergedCorrection and convergedFlux

        # self.journal.message(iterationMessage, self.identification)

        return incrementResidualHistory

    @performancetiming.timeit("linear solve")
    def _linearSolve(self, A: csr_matrix, b: DofVector, linearSolver) -> ndarray:
        """Solve the linear equation system.

        Parameters
        ----------
        A
            The system matrix in compressed spare row format.
        b
            The right hand side.

        Returns
        -------
        ndarray
            The solution 'x'.
        """

        ddU = linearSolver(A, b)

        if np.isnan(ddU).any():
            raise DivergingSolution("Obtained NaN in linear solve")

        return ddU

    @performancetiming.timeit("conversion VIJ to CSR")
    def _VIJtoCSR(self, KCoo: VIJSystemMatrix, csrGenerator, useInPlace: bool = False) -> csr_matrix:
        """Construct a CSR matrix from VIJ (COO)format.

        Parameters
        ----------
        K
            The system matrix in VIJ format.
        csrGenerator
            The CSR generator holding the sparsity pattern.
        useInPlace
            If True, update and return the generator's internal CSR matrix directly
            (no copy). Only safe if the caller does not retain the returned matrix
            across a subsequent conversion using the *same* ``csrGenerator``
            instance (e.g. a second matrix such as a mass matrix converted with the
            same generator would alias and overwrite it). Defaults to False, which
            always returns an independent copy.

        Returns
        -------
        csr_matrix
            The system matrix in compressed sparse row format.
        """
        if useInPlace:
            # In-place update: the returned matrix is the generator's internal CSR matrix.
            # The subsequent Dirichlet application only modifies values (the pattern is
            # preserved), which are fully overwritten again on the next update.
            KCsr = csrGenerator.updateInPlace(KCoo)
        else:
            KCsr = csrGenerator.updateCSR(KCoo)

        return KCsr

    @performancetiming.timeit("computation spatial fluxes")
    def _computeSpatialAveragedFluxes(self, F: DofVector, theDofManager) -> float:
        """Compute the spatial averaged flux for every field
        Is usually called by checkConvergence().

        Parameters
        ----------
        F
            The accumulated flux vector.

        Returns
        -------
        dict[str,float]
            A dictioary containg the spatial average fluxes for every field.
        """
        spatialAveragedFluxes = dict.fromkeys(theDofManager.idcsOfFieldsInDofVector, 0.0)
        for field, nDof in theDofManager.nAccumulatedNodalFluxesFieldwise.items():
            spatialAveragedFluxes[field] = np.linalg.norm(F[theDofManager.idcsOfFieldsInDofVector[field]], 1) / nDof

        return spatialAveragedFluxes

    def _checkDivergingSolution(self, incrementResidualHistory: dict, maxGrowingIter: int) -> bool:
        """Check if the iterative solution scheme is diverging.

        Parameters
        ----------
        incrementResidualHistory
            The dictionary containing the residual history of all fields.
        maxGrowingIter
            The maximum allows number of growths of a residual during the iterative solution scheme.

        Returns
        -------
        bool
            True if solution is diverging.
        """

        for history in incrementResidualHistory.values():
            nGrew = 0
            for i in range(len(history) - 1):
                if (
                    history[i + 1]["converged flux"] and history[i + 1]["converged correction"]
                ):  # don't count converged iterations
                    continue
                if history[i + 1]["absolute flux residual"] > history[i]["absolute flux residual"]:
                    nGrew += 1

            if nGrew > maxGrowingIter:
                return True

        return False

    def _printResidualOutlierNodes(self, incrementResidualHistory: dict):
        """Print which nodes have the largest residuals.

        Parameters
        ----------
        residualOutliers
            The dictionary containing the outlier nodes for every field.
        """
        self.journal.message(
            "Residual outliers:",
            self.identification,
            level=1,
        )
        for field, hist in incrementResidualHistory.items():
            self.journal.message(
                "|{:20}|node {:10}|".format(field, hist[-1]["node with largest residual"].label),
                self.identification,
                level=2,
            )

    @performancetiming.timeit("computation active cells")
    def _computeCells(
        self,
        activeCells: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K: SystemMatrixAssembly,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        activeCells
            The list of (active) cells to be evaluated.
        U_np
            The current global solution vector.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K
            Where each entity writes its stiffness contribution.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        for c in activeCells:
            Pc = np.zeros(c.nDof)
            Kc = K[c]
            c.computeKernels(Kc, Pc, U_np[c], dU[c], time, dT)
            P[c] += Pc
            F[c] += abs(Pc)

    @performancetiming.timeit("computation elements")
    def _computeElements(
        self,
        elements: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K: SystemMatrixAssembly,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all finite elements.

        An element contributes its internal force, which is the sign convention of cells and
        particles as well, so it is accumulated into the same vector.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        U_np
            The current global solution vector.
        dU
            The current global solution increment vector.
        P
            The current global internal force vector.
        F
            The accumulated nodal fluxes vector.
        K
            Where each entity writes its stiffness contribution.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """

        for el in elements:
            PEl = np.zeros(el.nDof)
            el.computeKernels(K[el], PEl, U_np[el], dU[el], time, dT)
            P[el] += PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation cell elements")
    def _computeCellElements(
        self,
        elements: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K: SystemMatrixAssembly,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cell elements.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        U_np
            The current global solution vector.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K
            Where each entity writes its stiffness contribution.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """

        for el in elements:
            PEl = np.zeros(el.nDof)
            KEl = K[el]
            el.computeKernels(KEl, PEl, U_np[el], dU[el], time, dT)
            P[el] += PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation particles")
    def _computeParticles(
        self,
        particles: Iterable[BaseParticle],
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K: SystemMatrixAssembly,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all particles.

        Parameters
        ----------
        particles
            The list of particles to be evaluated.
        U_np
            The current global solution vector.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        K
            Where each entity writes its stiffness contribution.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """

        if not particles:
            return
        particles = list(particles)  # Ensure we have a list

        scatter_P = (
            P.createScatterVector()
        )  # make a scatter vector; which gives 1) contiguous memory access and 2) thread safety

        # Process a CHUNK of particles per task, not just one, to keep the per-task
        # dispatch overhead negligible compared to the actual particle computation.
        def computeParticlesWorker(particleChunk):
            for particle in particleChunk:
                PP = scatter_P[particle]
                KP = K[particle]

                particle.computeKernels(KP, PP, U_np[particle], dU[particle], time, dT)

        numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

        chunkSize = max(1, len(particles) // (numThreads * 4))
        chunks = chunked_iterable(particles, chunkSize)

        executor = getThreadPool(numThreads)
        list(executor.map(computeParticlesWorker, chunks))

        scatter_P.assembleInto(P)
        scatter_P.assembleInto(F, absolute=True)

    @performancetiming.timeit("computation constraints")
    def _computeConstraints(
        self,
        constraints: list,
        U_np: DofVector,
        dU: DofVector,
        P: DofVector,
        K: SystemMatrixAssembly,
        timeStep: TimeStep,
    ):
        """Evaluate all active constraints: each adds its residual contribution and its tangent, see
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.computeKernels`.

        Parameters
        ----------
        constraints
            The list of constraints to be evaluated.
        U_np
            The current global solution vector.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        K
            Where each entity writes its stiffness contribution.
        timeStep
            The current time increment.
        """
        for c in constraints:
            if c.active:
                Pc = np.zeros(c.nDof)
                Kc = K[c]
                c.computeKernels(U_np[c], dU[c], Pc, Kc, timeStep)
                P[c] += Pc

    @performancetiming.timeit("creation newton cache")
    def _createNewtonCache(self, theDofManager):
        """Create expensive objects, which may be reused if the global system does not change.

        Parameters
        ----------
        theDofManager
            The DofManager instance.

        Returns
        -------
        tuple
            The collection of expensive objects.
        """

        K = VIJSystemMatrixAssembly(theDofManager, 1)
        csrGenerator = K.csrGenerator

        dU = theDofManager.constructDofVector()
        Rhs = theDofManager.constructDofVector()
        F = theDofManager.constructDofVector()
        PInt = theDofManager.constructDofVector()
        PExt = theDofManager.constructDofVector()

        newtonCache = (K, csrGenerator, dU, Rhs, F, PInt, PExt)

        return newtonCache

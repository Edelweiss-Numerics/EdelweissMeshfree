import concurrent.futures
from abc import abstractmethod
from typing import Iterable

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
from edelweissfe.numerics.csrgeneratorv2 import CSRGenerator
from edelweissfe.numerics.dofmanager import DofManager, DofVector, VIJSystemMatrix
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    isFreeThreadingSupported,
)
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.stepactions.base.stepactionbase import StepActionBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import DivergingSolution
from numpy import ndarray
from scipy.sparse import csr_matrix

from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.particles.base.baseparticle import BaseParticle
from edelweissmpm.solvers.base.nonlinearsolverbase import BaseNonlinearSolver
from edelweissmpm.stepactions.base.mpmdistributedloadbase import MPMDistributedLoadBase
from edelweissmpm.stepactions.particledistributedload import ParticleDistributedLoad


class BaseNonlinearImplicitSolver(BaseNonlinearSolver):

    @abstractmethod
    def _newtonSolve(
        self,
        dirichlets: list[DirichletBase],
        bodyLoads: list,
        distributedLoads: list,
        particleDistributedLoads: list,
        reducedNodeSets: list,
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

    @performancetiming.timeit("compute body loads")
    def _computeBodyLoads(
        self,
        bodyForces: list[StepActionBase],
        PExt: DofVector,
        K: VIJSystemMatrix,
        timeStep: TimeStep,
        theDofManager,
        activeCells,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all body forces loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        U_np
            The current solution vector.
        PExt
            The external load vector to be augmented.
        K
            The system matrix to be augmented.
        timeStep
            The current time increment.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        for bForce in bodyForces:
            loadVector = bForce.getCurrentLoad(timeStep)
            bLoadType = bForce.loadType

            for cl in bForce.cellSet:
                if cl in activeCells:
                    Pc = np.zeros(cl.nDof)
                    Kc = K[theDofManager.idcsOfCellsInDofVector[cl]]

                    cl.computeBodyLoad(bLoadType, loadVector, Pc, Kc, timeStep.totalTime, timeStep.timeIncrement)

                    PExt[cl] += Pc

        return PExt, K

    @performancetiming.timeit("compute distributed loads")
    def _computeCellDistributedLoads(
        self,
        distributedLoads: list[MPMDistributedLoadBase],
        PExt: DofVector,
        K_VIJ: VIJSystemMatrix,
        timeStep: TimeStep,
        theDofManager,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all body forces loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        PExt
            The external load vector to be augmented.
        K_VIJ
            The system matrix to be augmented.
        timeStep
            The current time increment.
        theDofManager
            The DofManager instance.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        for distributedLoad in distributedLoads:
            for mp in distributedLoad.mpSet:
                surfaceID, loadVector = distributedLoad.getCurrentMaterialPointLoad(mp, timeStep)

                for cl in mp.assignedCells:
                    Pc = np.zeros(cl.nDof)
                    Kc = K_VIJ[cl]
                    cl.computeDistributedLoad(
                        distributedLoad.loadType,
                        surfaceID,
                        mp,
                        loadVector,
                        Pc,
                        Kc,
                        timeStep.totalTime,
                        timeStep.timeIncrement,
                    )
                    PExt[cl] += Pc

        return PExt, K_VIJ

    @performancetiming.timeit("compute distributed loads")
    def _computeParticleDistributedLoads(
        self,
        distributedLoads: list[ParticleDistributedLoad],
        PExt: DofVector,
        K_VIJ: VIJSystemMatrix,
        timeStep: TimeStep,
        theDofManager,
    ) -> tuple[DofVector, VIJSystemMatrix]:
        """Loop over all body forces loads acting on elements, and evaluate them.
        Assembles into the global external load vector and the system matrix.

        Parameters
        ----------
        distributedLoads
            The list of distributed loads.
        PExt
            The external load vector to be augmented.
        K_VIJ
            The system matrix to be augmented.
        timeStep
            The current time increment.
        theDofManager
            The DofManager instance.

        Returns
        -------
        tuple[DofVector,VIJSystemMatrix]
            The augmented load vector and system matrix.
        """

        for distributedLoad in distributedLoads:

            # surfaceID, loadVector = distributedLoad.getCurrentLoad(None, timeStep)
            # for p in distributedLoad.particles:
            for p, surfaceID, loadVector in distributedLoad.getCurrentParticleLoads(timeStep):
                Pc = np.zeros(p.nDof)
                Kc = K_VIJ[p]
                p.computeDistributedLoad(
                    distributedLoad.loadType,
                    surfaceID,
                    loadVector,
                    Pc,
                    Kc,
                    timeStep.totalTime,
                    timeStep.timeIncrement,
                )

                PExt[p] += Pc

        return PExt, K_VIJ

    @performancetiming.timeit("dirichlet on CSR")
    def _applyDirichletKCsr(
        self,
        K: VIJSystemMatrix,
        dirichlets: list[DirichletBase],
        theDofManager: DofManager,
        reducedNodeSets: dict[NodeSet, NodeSet],
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
                reducedNodeSet = reducedNodeSets[dirichlet.nSet]
                for row in self._findDirichletIndices(theDofManager, dirichlet, reducedNodeSet):  # dirichlet.indices:
                    K.data[K.indptr[row] : K.indptr[row + 1]] = 0.0

            # K[row, row] = 1.0 @ once, faster than within the loop above:
            diag = K.diagonal()
            diag[
                np.concatenate(
                    [self._findDirichletIndices(theDofManager, d, reducedNodeSets[d.nSet]) for d in dirichlets]
                )
            ] = 1.0
            K.setdiag(diag)

            K.eliminate_zeros()

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
    def _VIJtoCSR(self, KCoo: VIJSystemMatrix, csrGenerator) -> csr_matrix:
        """Construct a CSR matrix from VIJ (COO)format.

        Parameters
        ----------
        K
            The system matrix in VIJ format.
        Returns
        -------
        csr_matrix
            The system matrix in compressed sparse row format.
        """
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
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        activeCells
            The list of (active) cells to be evaluated.
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
        for c in activeCells:
            dUc = dU[c]
            Pc = np.zeros(c.nDof)
            Kc = K_VIJ[c]
            c.computeMaterialPointKernels(dUc, Pc, Kc, time, dT)
            P[c] += Pc
            F[c] += abs(Pc)

    @performancetiming.timeit("computation active cells")
    def _computeCellsWithLumpedInertia(
        self,
        activeCells: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        M: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        activeCells
            The list of (active) cells to be evaluated.
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
        for c in activeCells:
            dUc = dU[c]
            Pc = np.zeros(c.nDof)
            Mc = np.zeros(c.nDof)
            Kc = K_VIJ[c]
            c.computeMaterialPointKernels(dUc, Pc, Kc, time, dT)
            c.computeLumpedInertia(Mc)
            M[c] += Mc
            P[c] += Pc
            F[c] += abs(Pc)

    @performancetiming.timeit("computation active cells")
    def _computeCellsWithConsistentInertia(
        self,
        activeCells: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        M: VIJSystemMatrix,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        activeCells
            The list of (active) cells to be evaluated.
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
        for c in activeCells:
            dUc = dU[c]
            Pc = np.zeros(c.nDof)
            Mc = np.zeros((c.nDof, c.nDof))
            Kc = K_VIJ[c]
            c.computeMaterialPointKernels(dUc, Pc, Kc, time, dT)
            c.computeConsistentInertia(Mc)
            M[c] += Mc
            P[c] += Pc
            F[c] += abs(Pc)

    @performancetiming.timeit("computation elements")
    def _computeElements(
        self,
        elements: list,
        dU: DofVector,
        Un: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        Un
            The previous global solution vector.
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
        time_ = np.array([time, time])

        for el in elements:
            dUEl = dU[el]
            UEln = Un[el]
            UElnp = UEln + dUEl
            PEl = np.zeros(el.nDof)
            KEl = K_VIJ[el]
            el.computeYourself(KEl, PEl, UElnp, dUEl, time_, dT)
            P[el] -= PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation elements")
    def _computeElementsWithConsistentInertia(
        self,
        elements: list,
        dU: DofVector,
        Un: DofVector,
        P: DofVector,
        F: DofVector,
        M_VIJ: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all elements with consistent inertia.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        Un
            The previous global solution vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        M_VIJ
            The global consistent mass matrix in VIJ (COO) format.
        K_VIJ
            The global system stiffness matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        time_ = np.array([time, time])

        for el in elements:
            dUEl = dU[el]
            UEln = Un[el]
            UElnp = UEln + dUEl
            PEl = np.zeros(el.nDof)
            MEl = np.zeros_like(M_VIJ[el])
            KEl = K_VIJ[el]
            el.computeYourself(KEl, PEl, UElnp, dUEl, time_, dT)
            el.computeConsistentInertia(MEl)
            M_VIJ[el] += MEl
            P[el] -= PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation elements")
    def _computeElementsWithLumpedInertia(
        self,
        elements: list,
        dU: DofVector,
        Un: DofVector,
        P: DofVector,
        F: DofVector,
        M: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all elements with lumped inertia.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        Un
            The previous global solution vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        M
            The lumped mass matrix as vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        time_ = np.array([time, time])

        for el in elements:
            dUEl = dU[el]
            UEln = Un[el]
            UElnp = UEln + dUEl
            PEl = np.zeros(el.nDof)
            MEl = np.zeros(el.nDof)
            KEl = K_VIJ[el]
            el.computeYourself(KEl, PEl, UElnp, dUEl, time_, dT)
            el.computeLumpedInertia(MEl)
            M[el] += MEl
            P[el] -= PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation cell elements")
    def _computeCellElements(
        self,
        elements: list,
        dU: DofVector,
        Un: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all cells.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        Un
            The previous global solution vector.
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

        for el in elements:
            dUEl = dU[el]
            UEln = Un[el]
            UElnp = UEln + dUEl
            PEl = np.zeros(el.nDof)
            KEl = K_VIJ[el]
            el.computeMaterialPointKernels(UElnp, PEl, KEl, time, dT)
            P[el] += PEl
            F[el] += abs(PEl)

    @performancetiming.timeit("computation particles")
    def _computeParticles(
        self,
        particles: Iterable[BaseParticle],
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all particles.

        Parameters
        ----------
        particles
            The list of particles to be evaluated.
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

        if not particles:
            return
        particles = list(particles)  # Ensure we have a list

        scatter_P = (
            P.createScatterVector()
        )  # make a scatter vector; which gives 1) contiguous memory access and 2) thread safety

        def computeParticleWorker(particle: BaseParticle):
            """
            Worker function to compute physics kernels for a single particle.

            Parameters
            ----------
            particle
                The particle to be processed.
            """
            PP = scatter_P[particle]
            dUP = dU[particle]
            KP = K_VIJ[particle]

            particle.computePhysicsKernels(dUP, PP, KP, time, dT)

        numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=numThreads) as executor:
            executor.map(computeParticleWorker, particles)

        scatter_P.assembleInto(P)
        scatter_P.assembleInto(F, absolute=True)

    @performancetiming.timeit("computation particles")
    def _computeParticlesWithConsistentInertia(
        self,
        particles: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        M_VIJ: VIJSystemMatrix,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all particles with consistent inertia.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        M_VIJ
            The global mass matrix in VIJ (COO) format.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        for p in particles:
            dUP = dU[p]
            PP = np.zeros(p.nDof)
            MP = np.zeros_like(M_VIJ[p])
            KP = K_VIJ[p]
            p.computePhysicsKernels(dUP, PP, KP, time, dT)
            p.computeConsistentInertia(MP)
            MP_ = M_VIJ[p]
            MP_ += MP
            P[p] += PP
            F[p] += abs(PP)

    @performancetiming.timeit("computation particles")
    def _computeParticlesWithLumpedInertia(
        self,
        particles: list,
        dU: DofVector,
        P: DofVector,
        F: DofVector,
        M: DofVector,
        K_VIJ: VIJSystemMatrix,
        time: float,
        dT: float,
        theDofManager: DofManager,
    ):
        """Evaluate all particles with consistent inertia.

        Parameters
        ----------
        elements
            The list of elements to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        F
            The accumulated nodal fluxes vector.
        M
            The global lumped mass matrix as vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        time
            The current time.
        dT
            The increment of time.
        theDofManager
            The DofManager instance.
        """
        for p in particles:
            dUP = dU[p]
            PP = np.zeros(p.nDof)
            MP = np.zeros(p.nDof)
            KP = K_VIJ[p]
            p.computePhysicsKernels(dUP, PP, KP, time, dT)
            p.computeLumpedInertia(MP)
            M[p] += MP
            P[p] += PP
            F[p] += abs(PP)

    @performancetiming.timeit("computation constraints")
    def _computeConstraints(
        self, constraints: list, dU: DofVector, P: DofVector, K_VIJ: VIJSystemMatrix, timeStep: TimeStep
    ):
        """Evaluate all constraints.

        Parameters
        ----------
        constraints
            The list of constraints to be evaluated.
        dU
            The current global solution increment vector.
        P
            The current global flux vector.
        K_VIJ
            The global system matrix in VIJ (COO) format.
        timeStep
            The current time increment.
        """
        for c in constraints:
            if c.active:
                dUc = dU[c]
                Pc = np.zeros(c.nDof)
                Kc = K_VIJ[c]
                c.applyConstraint(dUc, Pc, Kc, timeStep)
                P[c] += Pc

    @performancetiming.timeit("instancing csr generator")
    def _makeCachedCOOToCSRGenerator(self, K_VIJ):
        return CSRGenerator(K_VIJ)

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

        K_VIJ = theDofManager.constructVIJSystemMatrix()
        csrGenerator = self._makeCachedCOOToCSRGenerator(K_VIJ)
        dU = theDofManager.constructDofVector()
        Rhs = theDofManager.constructDofVector()
        F = theDofManager.constructDofVector()
        PInt = theDofManager.constructDofVector()
        PExt = theDofManager.constructDofVector()

        newtonCache = (K_VIJ, csrGenerator, dU, Rhs, F, PInt, PExt)

        return newtonCache

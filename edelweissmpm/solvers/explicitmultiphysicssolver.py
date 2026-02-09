# -*- coding: utf-8 -*-
import concurrent.futures
from typing import Iterable

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    isFreeThreadingSupported,
)
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import StepFailed
from edelweissfe.utils.fieldoutput import FieldOutputController
from prettytable import PrettyTable

from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.mpmmanagers.base.mpmmanagerbase import MPMManagerBase
from edelweissmpm.particlemanagers.base.baseparticlemanager import BaseParticleManager
from edelweissmpm.particles.base.baseparticle import BaseParticle
from edelweissmpm.solvers.base.nonlinearsolverbase import (
    BaseNonlinearSolver,
    RestartHistoryManager,
)


class ExplicitMultiphysicsSolver(BaseNonlinearSolver):
    """
    Explicit solver for multiphysics RKPM problems.
    """

    identification = "Explicit-MP-Solver"

    validOptions = {
        "field orders": {"displacement": 2, "temperature": 1},  # 2: Central Diff, 1: Forward Euler
        "damping factor": 0.0,
        "write restart interval": 0,
    }

    def __init__(self, journal: Journal):
        super().__init__(journal)

    @performancetiming.timeit("solve step")
    def solveStep(
        self,
        timeStepper,
        model: MPMModel,
        fieldOutputController: FieldOutputController,
        mpmManagers: list[MPMManagerBase] = [],
        particleManagers: list[BaseParticleManager] = [],
        dirichlets: list = [],
        bodyLoads: list = [],
        distributedLoads: list = [],
        particleDistributedLoads: list = [],
        outputManagers: list = [],
        userIterationOptions: dict = {},
        vciManagers: list = [],
        restartWriteInterval: int = 0,
        allowFallBackToRestart: bool = False,
        numberOfRestartsToStore=3,
        restartBaseName: str = "restart",
    ) -> tuple[bool, MPMModel]:

        options = self.validOptions.copy()
        options.update(userIterationOptions)

        table = PrettyTable(("Solver option", "value"))
        table.add_rows([(k, v) for k, v in options.items()])
        self.journal.printPrettyTable(table, self.identification)

        self._applyStepActionsAtStepStart(model, dirichlets + bodyLoads + distributedLoads)

        restartHistoryManager = RestartHistoryManager(restartBaseName, numberOfRestartsToStore)

        if not timeStepper.doesZeroIncrement():
            raise ValueError("The first time increment must be zero for explicit time integration.")

        particles = list(model.particles.values())
        constraints = list(model.constraints.values())

        try:
            for timeStep in timeStepper.generateTimeStep():
                dT = timeStep.timeIncrement
                self.journal.message(
                    f"Step {timeStep.number}: Time {timeStep.totalTime:.6e}, dt {dT:.6e}", self.identification
                )

                if dT == 0.0:
                    # +----------------+
                    # | zero increment |
                    # +----------------+
                    self._updateModelConnectivity(
                        list(), particles, constraints, model, timeStep, list(), particleManagers
                    )

                    (activeNodesPersistent, _, reducedNodeFields, reducedNodeSets) = self._assembleActiveDomain(
                        list(), model
                    )

                    activeConstraints = [c for c in constraints if c.active]

                    theDofManager = self._createDofManager(
                        reducedNodeFields.values(),
                        list(),
                        list(),
                        activeConstraints,
                        list(),
                        list(),
                        particles,
                        initializeVIJPattern=False,
                    )

                    (M, dU_np, P_Int, P_Ext, v_np_one_half, momentum) = self.getDiscretization(
                        theDofManager, model, mpmManagers, constraints
                    )
                    self.updateSystem(model, timeStep.totalTime, dT, dU_np)
                    self.computeSystem(particles, activeConstraints, P_Int, P_Ext, M, momentum, timeStep)
                    M_inv = np.reciprocal(M)
                    v_np_one_half[:] = momentum * M_inv

                else:
                    # +---------------------+
                    # | any other increment |
                    # +---------------------+
                    Rhs_n = P_Ext - P_Int
                    a_n = Rhs_n * M_inv

                    for field_name, order in options["field orders"].items():
                        if field_name not in theDofManager.idcsOfFieldsInDofVector:
                            continue
                        indices = theDofManager.idcsOfFieldsInDofVector[field_name]

                        if order == 2:  # Central Difference
                            v_np_one_half[indices] += a_n[indices] * dT  # v_(n+1/2) = v_(n-1/2) + a_n * dT
                            dU_np[indices] = v_np_one_half[indices] * dT  # (U_np - U_n) = v_(n+1/2) * dT
                        elif order == 1:  # Forward Euler
                            dU_np[indices] += a_n[indices] * dT

                self._applyStepActionsAtIncrementStart(model, timeStep, dirichlets + bodyLoads)

                # the solution increment to t_np is formulated in terms of the old discretization at t_n
                # so for MPM and RKPM this call connects the old discretization with the shift to the new positions
                # This is on contrast to FE, which can be exclusively computed in the new configuration using computeSystem(...) only
                self.updateSystem(model, timeStep.totalTime, dT, dU_np)
                model.advanceToTime(timeStep.totalTime)

                # A
                # |
                # |
                # |  +---------------------------+
                # +--| old discretization at t_n |
                #    +---------------------------+
                #
                connectivityHasChanged = self._updateModelConnectivity(
                    list(), model.particles.values(), constraints, model, timeStep, list(), particleManagers
                )
                if connectivityHasChanged:

                    self._updateDofManager(theDofManager, constraints, model.particles.values())

                    (M, dU_np, P_Int, P_Ext, _, momentum) = self.getDiscretization(
                        theDofManager, model, mpmManagers, constraints
                    )
                #    +-------------------------------+
                # +--| new discretization at t_(n+1) |
                # |  +-------------------------------+
                # |
                # |
                # V

                P_Int[:] = P_Ext[:] = M[:] = momentum[:] = 0.0
                self.computeSystem(particles, activeConstraints, P_Int, P_Ext, M, momentum, timeStep)
                M_inv = np.reciprocal(M)

                # v_np_one_half[:] = momentum * M_inv # omitting this line and simple taking v_np_one_half from previous step leads to way less dissipative results

                self._finalizeIncrementOutput(fieldOutputController, outputManagers)

                if restartWriteInterval and timeStep.number % restartWriteInterval == 0:
                    fn = restartHistoryManager.getNextRestartFileName()
                    self._writeRestart(model, timeStepper, fn)
                    restartHistoryManager.append(fn)

        except StepFailed:
            self.journal.errorMessage("Step Failed", self.identification)
            return False, model

        self._applyStepActionsAtStepEnd(model, dirichlets + bodyLoads + distributedLoads)
        fieldOutputController.finalizeStep()
        for man in outputManagers:
            man.finalizeStep()

        return True, model

    @performancetiming.timeit("computation particles")
    def _computeParticlesExplicit(
        self,
        particles: Iterable[BaseParticle],
        P: DofVector,
        M: DofVector,
        Mv: DofVector,
    ):
        """Evaluate all particles.

        Parameters
        ----------
        particles
            The list of particles to be evaluated.
        P
            The current global flux vector.
        M
            The current global lumped inertia vector.
        Mv
            The current global lumped momentum vector.
        """

        if not particles:
            return
        particles = list(particles)  # Ensure we have a list

        scatter_P = P.createScatterVector()
        scatter_M = M.createScatterVector()
        scatter_Mv = Mv.createScatterVector()

        def computeParticleWorker(particle: BaseParticle):
            """
            Worker function to compute physics kernels for a single particle.

            Parameters
            ----------
            particle
                The particle to be processed.
            """
            PP = scatter_P[particle]
            MP = scatter_M[particle]
            MVP = scatter_Mv[particle]

            particle.computePhysicsKernelsExplicit(PP)
            particle.computeLumpedInertia(MP)
            particle.computeLumpedMomentum(MVP)

        numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=numThreads) as executor:
            executor.map(computeParticleWorker, particles)

        scatter_P.assembleInto(P)
        scatter_M.assembleInto(M)
        scatter_Mv.assembleInto(Mv)

    @performancetiming.timeit("computation particles")
    def _updateParticlesExplicit(
        self,
        particles: Iterable[BaseParticle],
        dU: DofVector,
        time: float,
        dT: float,
    ):
        """Evaluate all particles.

        Parameters
        ----------
        particles
            The list of particles to be evaluated.
        dU
            The current global solution increment vector.
        time
            The current time.
        dT
            The increment of time.
        """

        if not particles:
            return

        particles = list(particles)  # Ensure we have a list

        def computeParticleWorker(particle: BaseParticle):
            """
            Worker function to compute physics kernels for a single particle.

            Parameters
            ----------
            particle
                The particle to be processed.
            """
            dUP = dU[particle]
            particle.updatePhysicsExplicit(dUP, time, dT)

        numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=numThreads) as executor:
            executor.map(computeParticleWorker, particles)

    @performancetiming.timeit("build discretization")
    def getDiscretization(self, theDofManager, model: MPMModel, mpmManagers: list[MPMManagerBase], constraints: list):
        """Assemble the system discretization.

        Parameters
        ----------
        model
            The MPM model to be discretized.
        mpmManagers
            The list of MPM managers handling the discretization.
        constraints
            The list of constraints to be applied.

        Returns
        -------
        theDofManager
            The assembled DOF manager.
        M
            The global lumped inertia vector.
        dU
            The global solution increment vector.
        P_Int
            The global internal flux vector.
        P_Ext
            The global external flux vector.
        v_np_one_half
            The global velocity vector at time n+1/2.
        Mv
            The global momentum vector.
        reducedNodeSets
            The reduced node sets for the active domain.
        """

        M = theDofManager.constructDofVector()
        dU = theDofManager.constructDofVector()
        P_Int = theDofManager.constructDofVector()
        P_Ext = theDofManager.constructDofVector()
        v_np_one_half = np.zeros_like(dU)
        Mv = theDofManager.constructDofVector()

        return M, dU, P_Int, P_Ext, v_np_one_half, Mv

    @performancetiming.timeit("compute system")
    def computeSystem(
        self,
        particles: list,
        constraints: list,
        P_Int: DofVector,
        P_Ext: DofVector,
        M: DofVector,
        Mv: DofVector,
        timeStep: TimeStep,
    ):
        """Compute the system vectors.

        Parameters
        ----------
        particles
            The list of particles to be evaluated.
        constraints
            The list of constraints to be applied.
        P_Int
            The global internal flux vector.
        P_Ext
            The global external flux vector.
        M
            The global lumped inertia vector.
        Mv
            The global momentum vector.
        timeStep
            The current time increment.
        """

        self._computeParticlesExplicit(
            particles,
            P_Int,
            M,
            Mv,
        )

        # self._computeParticleDistributedLoads(particleDistributedLoads, P_Ext_n, None, timeStep)

        self._computeConstraints(constraints, P_Ext, timeStep)

        return P_Int, P_Ext, M, Mv

    @performancetiming.timeit("update system")
    def updateSystem(self, model, totalTime, dT, dU: DofVector):
        """Update the system state.

        For RKPM, this involves applying the solution increment to the particles using the old discretization,
        before we update the connectivity for the new configuration.

        Parameters
        ----------
        model
            The MPM model to be updated.
        totalTime
            The current total time.
        dT
            The time increment.
        dU
            The current global solution increment vector.
        """

        self._updateParticlesExplicit(
            model.particles.values(),
            dU,
            totalTime,
            dT,
        )

    @performancetiming.timeit("computation constraints")
    def _computeConstraints(self, constraints: list, P: DofVector, timeStep: TimeStep):
        """Evaluate all constraints.

        Parameters
        ----------
        constraints
            The list of constraints to be evaluated.
        P
            The current global flux vector.
        timeStep
            The current time increment.
        """
        for c in constraints:
            if c.active:
                Pc = np.zeros(c.nDof)
                c.applyConstraint(Pc, timeStep)
                P[c] += Pc

    @performancetiming.timeit("updating dof structure")
    def _updateDofManager(self, theDofManager, constraints: list, particles: list):
        """
        Update the DOF manager with the current active constraints and particles.

        Parameters
        ----------
        theDofmanager
            The DofManager instance to be updated.
        constraints
            The list of constraints to be evaluated.
        particles
            The list of particles to be evaluated.
        """

        activeConstraints = [c for c in constraints if c.active]
        theDofManager.updateParticles(particles)
        theDofManager.updateConstraints(activeConstraints)

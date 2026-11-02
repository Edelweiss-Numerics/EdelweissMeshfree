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

from abc import abstractmethod
from collections import deque

import edelweissfe.utils.performancetiming as performancetiming
import h5py
import numpy as np
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    getThreadPool,
    isFreeThreadingSupported,
)
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.checkpoint import readRestartDataInto, writeRestartDataOf
from edelweissfe.utils.exceptions import ReachedMinIncrementSize, StepFailed
from edelweissfe.utils.fieldoutput import FieldOutputController

from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.base.mpmmanagerbase import MPMManagerBase
from edelweissmeshfree.numerics.dofmanager import MPMDofManager
from edelweissmeshfree.particlemanagers.base.baseparticlemanager import (
    BaseParticleManager,
)
from edelweissmeshfree.stepactions.base.mpmbodyloadbase import MPMBodyLoadBase
from edelweissmeshfree.stepactions.base.mpmdistributedloadbase import (
    MPMDistributedLoadBase,
)
from edelweissmeshfree.stepactions.particledistributedload import (
    ParticleDistributedLoad,
)


class RestartHistoryManager(deque):

    def __init__(self, restartBaseName, maxsize):
        super().__init__(maxlen=maxsize)
        self._restartBaseName = restartBaseName
        self._maxsize = maxsize
        self._currentCount = 0

    def append(self, item):
        super().append(item)
        self._currentCount = (self._currentCount + 1) % self._maxsize

    def pop(self):
        self._currentCount = self._currentCount - 1 if self._currentCount > 0 else self._maxsize - 1
        return super().pop()

    def getNextRestartFileName(
        self,
    ):
        theFileName = "{:}_{:}.h5".format(self._restartBaseName, self._currentCount)
        return theFileName


class BaseNonlinearSolver:
    """This is the base class for nonlinear implicit solvers.


    Parameters
    ----------
    journal
        The journal instance for logging.
    """

    #: Cache for :meth:`_findDirichletIndices`; lazily initialized since not all
    #: subclasses call ``super().__init__()``.
    _dirichletIndicesCache = None

    def __init__(self, journal: Journal):
        self.journal = journal

    @abstractmethod
    def solveStep(
        self,
        timeStepper,
        linearSolver,
        model: MPMModel,
        fieldOutputController: FieldOutputController,
        mpmManagers: list[MPMManagerBase] = [],
        particleManagers: list[BaseParticleManager] = [],
        dirichlets: list[DirichletBase] = [],
        bodyLoads: list[MPMBodyLoadBase] = [],
        distributedLoads: list[MPMDistributedLoadBase] = [],
        particleDistributedLoads: list[ParticleDistributedLoad] = [],
        constraints: list[ConstraintBase] = [],
        outputManagers: list[OutputManagerBase] = [],
        userIterationOptions: dict = {},
        vciManagers: list = [],
    ) -> tuple[bool, MPMModel]:
        pass

    @performancetiming.timeit("dirichlet on R")
    def _applyDirichlet(
        self,
        timeStep: TimeStep,
        R: DofVector,
        dirichlets: list[DirichletBase],
        theDofManager: DofManager,
    ):
        """Apply the dirichlet bcs on the residual vector
        Is called by solveStep() before solving the global equatuon system.

        Parameters
        ----------
        timeStep
            The time increment.
        R
            The residual vector of the global equation system to be modified.
        dirichlets
            The list of dirichlet boundary conditions.
        theDofManager
            The DofManager instance.

        Returns
        -------
        DofVector
            The modified residual vector.
        """

        for dirichlet in dirichlets:
            dofIndices, nodePositions = self._locateDirichlet(theDofManager, dirichlet)
            R[dofIndices] = dirichlet.getPrescribedIncrement(timeStep)[nodePositions].flatten()

        return R

    @performancetiming.timeit("step actions")
    def _applyStepActionsAtStepStart(self, model: MPMModel, actions):
        """Called when all step actions should be appliet at the start a step.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The dictionary of active step actions.
        """

        for action in actions:
            action.applyAtStepStart(model)

    @performancetiming.timeit("step actions")
    def _applyStepActionsAtStepEnd(self, model: MPMModel, actions):
        """Called when all step actions should finish a step.

        Parameters
        ----------
        model
            The model tree.
        stepActions
            The dictionary of active step actions.
        """

        for action in actions:
            action.applyAtStepEnd(model)

    @performancetiming.timeit("step actions")
    def _applyStepActionsAtIncrementStart(self, model: MPMModel, timeStep: TimeStep, actions):
        """Called when all step actions should be applied at the start of a step.

        Parameters
        ----------
        model
            The model tree.
        increment
            The time increment.
        stepActions
            The dictionary of active step actions.
        """

        for action in actions:
            action.applyAtIncrementStart(model, timeStep)

    def _findDirichletIndices(self, theDofManager: DofManager, dirichlet: DirichletBase) -> np.ndarray:
        """The DOF indices a Dirichlet boundary condition prescribes, node after node, see :meth:`_locateDirichlet`.

        Parameters
        ----------
        theDofManager
            The DofManager instance.
        dirichlet
            The boundary condition.

        Returns
        -------
        np.ndarray
            The prescribed DOF indices.
        """

        return self._locateDirichlet(theDofManager, dirichlet)[0]

    def _locateDirichlet(self, theDofManager: DofManager, dirichlet: DirichletBase) -> tuple[np.ndarray, np.ndarray]:
        """Where a Dirichlet boundary condition acts in the equation system: the DOF indices of its prescribed
        components on the nodes of its node set that lie in the active domain, and the positions of these nodes in
        the set, which select the rows of
        :meth:`~edelweissfe.stepactions.base.dirichletbase.DirichletBase.getPrescribedIncrement`.

        Memoized: the result is requested several times per Newton iteration (residual and system matrix), but
        changes only when the equation system is rebuilt or the boundary condition is updated between steps.

        Parameters
        ----------
        theDofManager
            The DofManager instance.
        dirichlet
            The boundary condition.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The prescribed DOF indices, and the positions of their nodes in the node set.
        """

        cache = self._dirichletIndicesCache
        if cache is None:
            cache = self._dirichletIndicesCache = {}

        key = (dirichlet, theDofManager, tuple(dirichlet.components))
        located = cache.get(key)
        if located is None:
            fieldIndices, nodePositions = theDofManager.locateFieldOnNodeSet(dirichlet.field, dirichlet.nSet)
            dofIndices = fieldIndices.reshape((-1, dirichlet.fieldSize))[:, dirichlet.components].flatten()
            located = cache[key] = (dofIndices, nodePositions)

        return located

    @performancetiming.timeit("assembly active domain")
    def _assembleActiveDomain(self, activeCells, model: MPMModel) -> tuple[NodeSet, NodeSet, dict, dict]:
        """Gather the Nodes, active NodeFields and NodeSets.

        Parameters
        ----------
        activeCells
            The currently active cells.
        model
            The full MPMModel.

        Returns
        -------
        tuple
            The tuple containing:
                - The set of active Nodes with persistent field values (FEM).
                - The set of active Nodes with volatile field values (MPM).
                - The dict of NodeFields on the active Nodes.
                - The dict of reduced NodeSets on the active Nodes.
        """
        return model.assembleActiveDomain(activeCells)

    @performancetiming.timeit("preparation material points")
    def _prepareMaterialPoints(self, materialPoints: list, time: float, dT: float):
        """Let the material points know that a new time step begins.

        Parameters
        ----------
        materialPoints
            The list of material points to be prepared.
        time
            The current time.
        dT
            The current time increment.
        """
        for mp in materialPoints:
            mp.prepareYourself(time, dT)

    @performancetiming.timeit("preparation particles")
    def _prepareParticles(self, particles: list, time: float, dT: float):
        """Let the material points know that a new time step begins.

        Parameters
        ----------
        particles
            The list of particles to be prepared.
        time
            The current time.
        dT
            The current time increment.
        """
        for p in particles:
            p.prepareYourself(time, dT)

    @performancetiming.timeit("interpolation to mps")
    def _interpolateFieldsToMaterialPoints(self, activeCells: list, dU: DofVector):
        """Let the solution be interpolated to all material points using the cells.

        Parameters
        ----------
        activeCells
            The list of active cells, which contain material points.
        dU
            The current solution increment to be interpolated.
        """
        for c in activeCells:
            dUCell = dU[c]
            c.interpolateFieldsToMaterialPoints(dUCell)

    @performancetiming.timeit("computation material points")
    def _computeMaterialPoints(self, materialPoints: list, time: float, dT: float):
        """Evaluate all material points' physics.

        Parameters
        ----------
        materialPonts
            The list material points to  evaluated.
        time
            The current time.
        dT
            The increment of time.
        """
        for mp in materialPoints:
            mp.computeYourself(time, dT)

    @performancetiming.timeit("instancing dof manager")
    def _createDofManager(self, *args, **kwargs):
        return MPMDofManager(*args, **kwargs)

    @performancetiming.timeit("update connectivity")
    def _updateManagedConnectivity(self, managers: list[MPMManagerBase] | list[BaseParticleManager]) -> bool:
        """Update the connectivity of all MPMManagers or particle managers.

        Parameters
        ----------
        managers
            The list of managers to update.
        """
        connectivityHasChanged = False
        for man in managers:
            connectivityHasChanged |= man.updateConnectivity()

        return connectivityHasChanged

    @performancetiming.timeit("postprocessing & output")
    def _finalizeIncrementOutput(self, fieldOutputController, outputmanagers):
        fieldOutputController.finalizeIncrement()
        for man in outputmanagers:
            man.finalizeIncrement()

    @performancetiming.timeit("writing restart")
    def _writeRestart(self, model: MPMModel, timeStepper, fileName):
        """Write the restart file.

        Parameters
        ----------
        model
            The model to be written.
        timeStepper
            The timeStepper to be written.
        fileName
            The name of the restart file.
        """
        theRestartFile = h5py.File(fileName, "w")

        model.writeRestart(theRestartFile)
        writeRestartDataOf(theRestartFile, {"timestepper": timeStepper})

    def readRestart(
        self,
        restartFile,
        timeStepper,
        model: MPMModel,
    ):
        """Read a restart file.

        Parameters
        ----------
        restartFile
            The name of the restart file.
        timeStepper
            The timeStepper instance to be read from the restart file.
        model
            The full MPMModel instance to be read from the restart file.
        """
        theRestartFile = h5py.File(restartFile, "r")

        model.readRestart(theRestartFile)
        readRestartDataInto(theRestartFile, {"timestepper": timeStepper})

    def _tryFallbackWithRestartFiles(
        self, writtenRestarts: RestartHistoryManager, timeStepper, model: MPMModel, iterationOptions: dict
    ):
        """Fallback to a previous converged increment using the written restart files.

        Parameters
        ----------
        writtenRestarts
            The list of written restart files.
        timeStepper
            The timeStepper instance.
        model
            The full MPMModel instance.
        iterationOptions
            The dictionary containing the iteration options.
        """

        while True:
            try:
                previousRestartFile = writtenRestarts.pop()
            except IndexError:
                raise StepFailed("No more restart files available for fallback")

            self.readRestart(previousRestartFile, timeStepper, model)
            self.journal.message(
                "Reverting to last successful increment at time {:}".format(model.time), self.identification
            )

            try:
                timeStepper.rejectTimeStep(iterationOptions["failed increment cutback factor"])
            except ReachedMinIncrementSize:
                continue

            break

    def _updateModelConnectivity(
        self, materialPoints, particles, constraints, model, timeStep, mpmManagers, particleManagers
    ):

        connectivityHasChanged = False

        if materialPoints:
            self.journal.message(
                "updating material point - cell connectivity",
                self.identification,
                level=1,
            )
            self._prepareMaterialPoints(materialPoints, timeStep.totalTime, timeStep.timeIncrement)
            connectivityHasChanged |= self._updateManagedConnectivity(mpmManagers)

        if particleManagers:
            self.journal.message(
                "updating particle kernel connectivity",
                self.identification,
                level=1,
            )
            self._prepareParticles(particles, timeStep.totalTime, timeStep.timeIncrement)
            connectivityHasChanged |= self._updateManagedConnectivity(particleManagers)

        connectivityHasChanged |= self._updateConstraintConnectivity(constraints, model)

        return connectivityHasChanged

    @performancetiming.timeit("constraint connectivity")
    def _updateConstraintConnectivity(self, constraints: list, model) -> bool:
        """Update the connectivity of all constraints.

        In a particle simulation there is typically one constraint per particle, so this loop is
        long enough to be worth spreading over the available threads.

        That is safe here for two reasons, both worth being explicit about:

        1. Every ``updateConnectivity`` implementation writes only to the constraint it belongs to.
           None of them writes to the model, and none of them creates nodes or variables, so two
           constraints can never touch the same memory.
        2. The individual answers are combined with a logical *or*. That is independent of the order
           in which the answers arrive, so the *value* returned does not depend on how the work was
           split, nor on how the threads happened to be scheduled.

        Point 2 is about the value only, and it is not on its own enough: ``any`` short-circuits, so
        reducing a lazy ``Executor.map`` would also make this function *return early*, while the
        chunks after the first changed one are still writing. The caller rebuilds the dof manager
        immediately afterwards, from the very node lists those threads are still rewriting. Hence
        the results are materialised before being reduced -- see the comment at the call below.

        Parameters
        ----------
        constraints
            The list of constraints to be updated.
        model
            The model the constraints act on.

        Returns
        -------
        bool
            True if the connectivity of at least one constraint has changed.
        """

        # Callers legitimately hand over a dict view rather than a list, which the serial loop this
        # replaced was happy with; chunking needs indexing, so materialise it once.
        constraints = list(constraints)

        numberOfThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1

        # Below this size the cost of handing work to the thread pool outweighs the work itself.
        minimumNumberOfConstraintsForThreading = 500

        if numberOfThreads == 1 or len(constraints) < minimumNumberOfConstraintsForThreading:
            return self._updateConnectivityOfConstraintChunk(constraints, model)

        chunkSize = len(constraints) // numberOfThreads + 1
        constraintChunks = [constraints[i : i + chunkSize] for i in range(0, len(constraints), chunkSize)]

        threadPool = getThreadPool(numberOfThreads)

        # Materialised before it is reduced, and that is the whole point of the ``list``:
        # ``Executor.map`` returns a *lazy* iterator, and ``any`` stops consuming at the first
        # ``True``. Reducing the iterator directly therefore returns while the chunks after the
        # first changed one are still running -- and the caller goes straight on to rebuild the dof
        # manager from constraint node lists that those threads are still rewriting. A constraint
        # updated after that snapshot keeps a dof block sized for a kernel support its particle no
        # longer has, and the next force evaluation fails to broadcast.
        #
        # Every ``executor.map`` in the particle managers is wrapped the same way, for the same
        # reason. Do not "simplify" this back into ``any(threadPool.map(...))``; the test
        # ``solvers/test/constraintconnectivity_test.py`` fails if it is.
        connectivityHasChangedPerChunk = list(
            threadPool.map(lambda chunk: self._updateConnectivityOfConstraintChunk(chunk, model), constraintChunks)
        )

        return any(connectivityHasChangedPerChunk)

    def _updateConnectivityOfConstraintChunk(self, constraints: list, model) -> bool:
        """Update the connectivity of one chunk of constraints.

        Parameters
        ----------
        constraints
            The chunk of constraints to be updated.
        model
            The model the constraints act on.

        Returns
        -------
        bool
            True if the connectivity of at least one constraint in this chunk has changed.
        """

        connectivityHasChanged = False
        for constraint in constraints:
            connectivityHasChanged |= constraint.updateConnectivity(model)

        return connectivityHasChanged

    def _getActiveCellsFromManagers(self, mpmManagers: list[MPMManagerBase]) -> list:
        """Collect the cells that are active in any of the MPM managers, sorted by their number; see
        :meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.activeCellsOf`.

        Parameters
        ----------
        mpmManagers
            The list of MPMManagerBase instances.

        Returns
        -------
        list
            The active cells, each once, sorted by cell number.
        """

        return MPMModel.activeCellsOf(mpmManagers)

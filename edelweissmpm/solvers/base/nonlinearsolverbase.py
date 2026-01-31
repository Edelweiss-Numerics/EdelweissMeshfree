# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissMPM.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMPM.
#  ---------------------------------------------------------------------

from abc import abstractmethod
from collections import deque

import edelweissfe.utils.performancetiming as performancetiming
import h5py
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.outputmanagers.base.outputmanagerbase import OutputManagerBase
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.stepactions.base.dirichletbase import DirichletBase
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.exceptions import ReachedMinIncrementSize, StepFailed
from edelweissfe.utils.fieldoutput import FieldOutputController

from edelweissmpm.fields.nodefield import MPMNodeField
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.mpmmanagers.base.mpmmanagerbase import MPMManagerBase
from edelweissmpm.numerics.dofmanager import MPMDofManager
from edelweissmpm.particlemanagers.base.baseparticlemanager import BaseParticleManager
from edelweissmpm.stepactions.base.mpmbodyloadbase import MPMBodyLoadBase
from edelweissmpm.stepactions.base.mpmdistributedloadbase import MPMDistributedLoadBase
from edelweissmpm.stepactions.particledistributedload import ParticleDistributedLoad


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
        reducedNodeSets,
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
        activeNodeSets
            The sets with active nodes only.
        theDofManager
            The DofManager instance.

        Returns
        -------
        DofVector
            The modified residual vector.
        """

        for dirichlet in dirichlets:
            dirichletNodes = reducedNodeSets[dirichlet.nSet]
            R[self._findDirichletIndices(theDofManager, dirichlet, dirichletNodes)] = dirichlet.getDelta(
                timeStep, dirichletNodes
            ).flatten()

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

    def _findDirichletIndices(self, theDofManager, dirichlet, reducedNodeSet):
        fieldIndices = theDofManager.idcsOfFieldsOnNodeSetsInDofVector[dirichlet.field][reducedNodeSet]

        return fieldIndices.reshape((-1, dirichlet.fieldSize))[:, dirichlet.components].flatten()

    @performancetiming.timeit("assembly active domain")
    def _assembleActiveDomain(self, activeCells, model: MPMModel) -> tuple[NodeSet, NodeSet, list, list]:
        """Gather the Nodes, active NodeFields and NodeSets.

        Parameters
        ----------
        model
            The full MPMModel.
        mpmManager
            The MPMManager intance.

        Returns
        -------
        tuple
            The tuple containing:
                - The set of active Nodes with persistent field values (FEM).
                - The set of active Nodes with volatile field values (MPM).
                - the list of NodeFields on the active Nodes.
                - the list of reduced NodeSets on the active Nodes.
        """
        # TODO: This method should be part of Model, in the spirit of 'getReducedModel()' or similar

        activeNodesWithPersistentFieldValues = set(
            n for element in model.elements.values() for n in element.nodes
        ) | set(n for element in model.cellElements.values() for n in element.nodes)

        activeNodesWithVolatileFieldValues = set(n for cell in activeCells for n in cell.nodes)

        activeNodesWithVolatileFieldValues |= set(
            kf.node for particle in model.particles.values() for kf in particle.kernelFunctions
        )

        activeNodes = activeNodesWithVolatileFieldValues | activeNodesWithPersistentFieldValues

        activeNodes = NodeSet("activeNodes", activeNodes)
        activeNodesWithPersistentFieldValues = NodeSet(
            "activeNodesWithPersistentFieldvalues", activeNodesWithPersistentFieldValues
        )
        activeNodesWithVolatileFieldValues = NodeSet(
            "activeNodesWithVolatileFieldValues", activeNodesWithVolatileFieldValues
        )

        reducedNodeFields = {
            nodeField.name: MPMNodeField(nodeField.name, nodeField.dimension, activeNodes)
            for nodeField in model.nodeFields.values()
        }

        reducedNodeSets = {
            nodeSet: NodeSet(nodeSet.name, set(activeNodes).intersection(nodeSet))
            for nodeSet in model.nodeSets.values()
        }

        return (
            activeNodesWithPersistentFieldValues,
            activeNodesWithVolatileFieldValues,
            reducedNodeFields,
            reducedNodeSets,
        )

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
        timeStepper.writeRestart(theRestartFile)

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
        timeStepper.readRestart(theRestartFile)

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
                timeStepper.reduceNextIncrement(iterationOptions["failed increment cutback factor"])
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

        for c in constraints:
            connectivityHasChanged |= c.updateConnectivity(model)

        return connectivityHasChanged

    def _getActiveCellsFromManagers(self, mpmManagers):

        activeCells = set()
        for man in mpmManagers:
            activeCells |= man.getActiveCells()
        return activeCells

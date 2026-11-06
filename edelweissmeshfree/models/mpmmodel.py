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

import copy

import h5py
from edelweissfe.config.phenomena import getFieldSize, phenomena
from edelweissfe.journal.journal import Journal
from edelweissfe.models.femodel import FEModel
from edelweissfe.sets.elementset import ElementSet
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.exceptions import RestartError
from edelweissfe.variables.fieldvariable import FieldVariable
from prettytable import PrettyTable

from edelweissmeshfree.fields.nodefield import MPMNodeField
from edelweissmeshfree.models.mpmequationsystemlayout import MPMEquationSystemLayout
from edelweissmeshfree.numerics.dofmanager import MPMDofManager
from edelweissmeshfree.sets.cellelementset import CellElementSet
from edelweissmeshfree.sets.cellset import CellSet
from edelweissmeshfree.sets.materialpointset import MaterialPointSet
from edelweissmeshfree.sets.particleset import ParticleSet


class MPMModel(FEModel):
    """This is is a standard mpm model tree.
    It takes care of the correct number of variables,
    for nodes and scalar degrees of freedom, and it manages the fields.

    Parameters
    ----------
    dimension
        The dimension of the model.
    """

    identification = "MPMModel"

    def __init__(self, dimension: int):
        self.cells = {}  #: The collection of Cells in the present model.
        self.cellSets = {}  #: The collection of CellSets in the present model.
        self.materialPoints = {}  #: The collection of MaterialPoints in the present model.
        self.materialPointSets = {}  #: The collection of MaterialPointSets in the present model.
        self.cellElements = {}  #: The collection of CellElements in the present model.
        self.cellElementSets = {}  #: The collection of CellElementSets in the present model.
        self.particles = {}  #: The collection of Particles in the present model.
        self.particleSets = {}  #: The collection of ParticleSets in the present model.
        self.meshfreeKernelFunctions = {}  #: The collection of MeshfreeKernelFunctions in the present model.
        self.particleKernelDomains = {}  #: The collection of ParticleKernelDomains in the present model.
        self.vertexSets = {}  #: The collection of VertexSets for per particle in the present model.
        #: The managers that assign the material points to the cells they lie in, see :meth:`updateConnectivity`.
        self.mpmManagers = []
        #: The managers that assign the particles to the kernel functions they reach, see :meth:`updateConnectivity`.
        self.particleManagers = []
        #: The variationally consistent integration managers, which correct the particles' test functions after every
        #: update of the connectivity, see :meth:`updateConnectivity`.
        self.vciManagers = []

        super().__init__(dimension)

    def _populateNodeFieldVariablesFromCells(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all
        MaterialPointCells .
        """
        for cell in self.cells.values():
            for node, nodeFields in zip(cell.nodes, cell.fields):
                for field in nodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node, field)

    def _populateNodeFieldVariablesFromCellElements(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all
        MaterialPointCells .
        """
        for cellElement in self.cellElements.values():
            for node, nodeFields in zip(cellElement.nodes, cellElement.fields):
                for field in nodeFields:
                    if field not in node.fields:
                        node.fields[field] = FieldVariable(node, field)

    def _populateNodeFieldVariablesFromParticleKernelDomains(
        self,
    ):
        """Creates FieldVariables on Nodes depending on the all defined particle-kernel interactions.
        For particles and kernels, the situation is a bit more complicated compared to elements or cells:
        Unlike for elements or cells, there is no direct connection between a particle and a node.
        Hence, the fields on the nodes are not necessarily known at the beginning of simulation.

        For this reason, we force the user to tell us which particles (which have fields) are interacting with which kernelfunctions during the simulation.
        """

        for particleKernelInteraction in self.particleKernelDomains.values():
            theFields = particleKernelInteraction.particles[
                0
            ].baseFields  # TODO: We assume that all particles have the same fields; let's change this later
            for field in theFields:
                for kf in particleKernelInteraction.meshfreeKernelFunctions:
                    if field not in kf.node.fields:
                        kf.node.fields[field] = FieldVariable(kf.node, field)

    def _prepareVariablesAndFields(self, journal):
        """Prepare all variables and fields for a simulation.

        Parameters
        ----------
        journal
            The journal instance.
        """
        if self.cells:
            journal.message("Activating fields on Nodes from Cells", self.identification)
            self._populateNodeFieldVariablesFromCells()
        if self.cellElements:
            journal.message("Activating fields on Nodes from CellElements", self.identification)
            self._populateNodeFieldVariablesFromCellElements()
        if self.particleKernelDomains:
            journal.message("Activating fields on Nodes from Particles", self.identification)
            self._populateNodeFieldVariablesFromParticleKernelDomains()

        return super()._prepareVariablesAndFields(journal)

    def _prepareMaterialPoints(self, journal: Journal):
        """Initialize the material points for a simulation. They carry their material from their creation; the
        sections are assigned to the finite elements by :meth:`~edelweissfe.models.femodel.FEModel.prepareYourself`.

        Parameters
        ----------
        journal
            The journal instance.
        """
        for mp in self.materialPoints.values():
            mp.initializeYourself()

    def _prepareParticles(self, journal: Journal):
        """Prepare elements for a simulation.
        In detail, sections are assigned.


        Parameters
        ----------
        journal
            The journal instance.
        """
        # for section in self.sections.values():
        #     section.assignSectionPropertiesToModel(self)

        for p in self.particles.values():
            p.initializeYourself()

    def prepareYourself(self, journal: Journal):
        """Prepare the model for a simulation.
        Creates the variables, bundles the fields,
        and initializes elements.


        Parameters
        ----------
        journal
            The journal instance.
        """

        self.nodeSets["all"] = NodeSet("all", self.nodes.values())
        self.elementSets["all"] = ElementSet("all", self.elements.values())
        self.materialPointSets["all"] = MaterialPointSet("all", self.materialPoints.values())
        self.cellSets["all"] = CellSet("all", self.cells.values())
        self.cellElementSets["all"] = CellElementSet("all", self.cellElements.values())
        self.particleSets["all"] = ParticleSet("all", self.particles.values())

        self._prepareMaterialPoints(journal)
        self._prepareParticles(journal)

        super().prepareYourself(journal)

    def advanceToTime(self, time: float):
        """Accept the current state of the model and sub instances, and
        set the new time.

        Parameters
        ----------
        time
            The new time.
        """

        for mp in self.materialPoints.values():
            mp.acceptStateAndPosition()

        for p in self.particles.values():
            p.acceptStateAndPosition()

        self.previousTime = self.time

        return super().advanceToTime(time)

    def goToPreviousTimeStep(self):

        for p in self.particles.values():
            p.revertToPreviousState()

        self.time = self.previousTime

    def _createNodeFieldsFromNodes(self, nodes: list, nodeSets: list) -> dict[str, MPMNodeField]:
        """Bundle nodal FieldVariables together in contiguous NodeFields.

        Parameters
        ----------
        nodes
            The list of Nodes from which the NodeFields should be created.
        nodeSets
            The list of NodeSets, which should be considered in the index map of the NodeFields.

        Returns
        -------
        dict[str,NodeField]
            The dictionary containing the NodeField instances for every active field."""

        domainSize = self.domainSize

        theNodeFields = dict()
        for field in phenomena.keys():
            theNodeField = MPMNodeField(field, getFieldSize(field, domainSize), nodes)

            if theNodeField.nodes:
                theNodeFields[field] = theNodeField

        return theNodeFields

    def updateConnectivity(self) -> bool:
        """Bring the coupling of the degrees of freedom up to date with the converged state, before an increment; see
        :meth:`~edelweissfe.models.femodel.FEModel.updateConnectivity`. Called by EdelweissFE's solvers at the start
        of every increment and of every retry after a cutback.

        First the managers assign the material points to the cells they lie in (:attr:`mpmManagers`) and the particles
        to the kernel functions they reach (:attr:`particleManagers`), each from its converged state: a failed attempt
        may have left a trial state behind. Then the constraints follow, which couple the nodes the managers have just
        assigned. Finally, the test functions of the particles are corrected for the new connectivity
        (:attr:`vciManagers`).

        A change of the active domain is a change of the coupling, not of the mesh: no node is created or moved, so the
        solver rebuilds its equation system, but does not re-equilibrate.

        Returns
        -------
        bool
            True if a coupling changed, so that the equation system has to be rebuilt.
        """

        connectivityChanged = False

        if self.mpmManagers:
            # The arguments are those of the converged state; no material point reads them here.
            for materialPoint in self.materialPoints.values():
                materialPoint.prepareYourself(self.time, 0.0)
            for mpmManager in self.mpmManagers:
                connectivityChanged |= mpmManager.updateConnectivity()

        if self.particleManagers:
            for particle in self.particles.values():
                particle.prepareYourself(self.time, 0.0)
            for particleManager in self.particleManagers:
                connectivityChanged |= particleManager.updateConnectivity()

        connectivityChanged |= super().updateConnectivity()

        for vciManager in self.vciManagers:
            vciManager.computeVCICorrections()

        return connectivityChanged

    @property
    def activeNodesAreFixed(self) -> bool:
        """Whether a change of the connectivity leaves the nodes of the active domain as they are, so that an equation
        system can follow it by locating the changed particles and the constraints anew
        (:meth:`~edelweissmeshfree.models.mpmequationsystemlayout.MPMEquationSystemLayout.relocateParticlesAndConstraints`)
        instead of being rebuilt: without MPM managers, whose cells become active and inactive, and with every particle
        manager bonding its kernel functions to the particles, so that a kernel function always covers its own
        particle.

        Returns
        -------
        bool
            True if the active nodes cannot change.
        """

        return not self.mpmManagers and all(
            particleManager.kernelFunctionsAreBondedToParticles for particleManager in self.particleManagers
        )

    def buildEquationSystem(self, withSystemMatrix: bool = True) -> MPMEquationSystemLayout:
        """Lay out the equation system of the model on its active domain; see
        :meth:`~edelweissfe.models.femodel.FEModel.buildEquationSystem` and :class:`MPMEquationSystemLayout`.

        The active domain consists of the cells that hold material points, the nodes these cells and the particles'
        kernel functions reach, and the finite elements and cell elements with their nodes. Its degrees of freedom are
        numbered on the active nodes; the model's node sets are located on their nodes in the active domain. The
        elements the solver computes are the finite elements, the active cells, the cell elements and the particles.

        Parameters
        ----------
        withSystemMatrix
            False for an explicit solver, which assembles no system matrix.

        Returns
        -------
        MPMEquationSystemLayout
            The layout of the equation system.
        """

        activeCells = self.activeCellsOf(self.mpmManagers)
        nodesWithPersistentFieldValues, _, activeNodeFields, activeNodeSets = self.assembleActiveDomain(activeCells)
        constraints = self.activeConstraints()

        dofManager = MPMDofManager(
            activeNodeFields.values(),
            [scalarVariable for constraint in constraints for scalarVariable in constraint.scalarVariables],
            elements=self.elements.values(),
            constraints=constraints,
            nodeSets=self.nodeSets.values(),
            cells=activeCells,
            cellElements=self.cellElements.values(),
            particles=self.particles.values(),
            initializeVIJPattern=withSystemMatrix,
            determiningIndexToHostObjectMappping=withSystemMatrix,
        )

        return MPMEquationSystemLayout(
            self,
            dofManager,
            [*self.elements.values(), *activeCells, *self.cellElements.values(), *self.particles.values()],
            activeCells,
            nodesWithPersistentFieldValues,
            activeNodeFields,
            self._reducedToActiveDomain(activeCells, activeNodeFields, activeNodeSets),
        )

    @staticmethod
    def activeCellsOf(mpmManagers: list) -> list:
        """The cells that are active in any of the MPM managers.

        Cells define no ``__hash__``, so a set of them iterates in an address-dependent order that differs between runs.
        The order of the active cells fixes their blocks in the system matrix and the order in which their
        contributions are summed, so the cells are sorted by their unique number.

        Parameters
        ----------
        mpmManagers
            The MPM managers.

        Returns
        -------
        list
            The active cells, each once, sorted by cell number.
        """

        activeCells = set()
        for mpmManager in mpmManagers:
            activeCells.update(mpmManager.getActiveCells())

        return sorted(activeCells, key=lambda cell: cell.cellNumber)

    def getActiveSubModel(self):
        """
        For many applications only a part of the model is active.
        For instance, in MPM simulations only cells occupied by material points are active.
        In order to create a numerical model, we need to know which part of the model is active, and which is not.
        This function returns the active sub model, based on the current state of the model.

        Returns
        -------
        MPMModel
            The active sub model.
        """

        activeCells = sorted(
            {cell for mp in self.materialPoints.values() for cell in mp.cells}, key=lambda cell: cell.cellNumber
        )

        _, _, reducedNodeFields, reducedNodeSets = self.assembleActiveDomain(activeCells)

        return self._reducedToActiveDomain(activeCells, reducedNodeFields, reducedNodeSets)

    def _reducedToActiveDomain(self, activeCells: list, activeNodeFields: dict, activeNodeSets: dict):
        """A shallow copy of the model, reduced to its active domain: its cells, node fields and node sets.

        Parameters
        ----------
        activeCells
            The active cells.
        activeNodeFields
            The node fields on the active nodes.
        activeNodeSets
            The node sets reduced onto the active nodes.

        Returns
        -------
        MPMModel
            The reduced model.
        """

        activeModel = copy.copy(self)
        activeModel.nodeSets = activeNodeSets
        activeModel.nodeFields = activeNodeFields
        activeModel.cells = {cell.cellNumber: cell for cell in activeCells}

        return activeModel

    def assembleActiveDomain(self, activeCells) -> tuple[NodeSet, NodeSet, dict, dict]:
        """Gather the active Nodes, and the NodeFields and NodeSets reduced onto them.

        The single implementation of this; :meth:`getActiveSubModel` wraps it into a reduced model,
        and the meshfree solvers call it directly (they already know their active cells, which need
        not agree with the cells reachable from ``self.materialPoints``).

        Parameters
        ----------
        activeCells
            The currently active cells, as determined by the caller.

        Returns
        -------
        tuple
            The tuple containing:
                - The NodeSet of active Nodes with persistent field values (FEM).
                - The NodeSet of active Nodes with volatile field values (MPM).
                - The dict of NodeFields on the active Nodes.
                - The dict of NodeSets reduced onto the active Nodes.
        """

        cells = activeCells.values() if isinstance(activeCells, dict) else activeCells

        activeNodesWithPersistentFieldValues = set(
            n for element in self.elements.values() for n in element.nodes
        ) | set(n for element in self.cellElements.values() for n in element.nodes)

        activeNodesWithVolatileFieldValues = set(n for cell in cells for n in cell.nodes)

        activeNodesWithVolatileFieldValues |= set(
            kf.node for particle in self.particles.values() for kf in particle.kernelFunctions
        )

        activeNodes = activeNodesWithVolatileFieldValues | activeNodesWithPersistentFieldValues

        # Node defines no __hash__, so a plain set of Node objects iterates in an address-dependent
        # order that differs between runs. DofManager assigns DOF indices strictly in
        # nodeField.nodes order, so leaving this unsorted made the global sparsity pattern
        # irreproducible -- and with it direct-solver fill-in and AMG aggregation, i.e. solve times
        # and iteration counts. Labels are unique, so sorting on them pins the numbering down
        # without changing which nodes are active. Guarded by dof_ordering_test.py.
        activeNodes = NodeSet("activeNodes", sorted(activeNodes, key=lambda n: n.label))
        activeNodesWithPersistentFieldValues = NodeSet(
            "activeNodesWithPersistentFieldvalues",
            sorted(activeNodesWithPersistentFieldValues, key=lambda n: n.label),
        )
        activeNodesWithVolatileFieldValues = NodeSet(
            "activeNodesWithVolatileFieldValues",
            sorted(activeNodesWithVolatileFieldValues, key=lambda n: n.label),
        )

        reducedNodeFields = {
            nodeField.name: MPMNodeField(nodeField.name, nodeField.dimension, activeNodes)
            for nodeField in self.nodeFields.values()
        }

        reducedNodeSets = {
            nodeSet: NodeSet(nodeSet.name, sorted(set(activeNodes).intersection(nodeSet), key=lambda n: n.label))
            for nodeSet in self.nodeSets.values()
        }

        return (
            activeNodesWithPersistentFieldValues,
            activeNodesWithVolatileFieldValues,
            reducedNodeFields,
            reducedNodeSets,
        )

    def makePrettyTableSummary(self):
        """Create a pretty table with a summary of the model properties."""
        prettytable = PrettyTable(("model property", ""))

        prettytable.add_row(("domain dim.", self.domainSize))
        prettytable.add_row(("time", self.time))
        prettytable.add_row(("nodes", len(self.nodes)))
        prettytable.add_row(("elements", len(self.elements)))
        prettytable.add_row(("node sets", list(self.nodeSets.keys())))
        prettytable.add_row(("node fields", list(self.nodeFields.keys())))
        prettytable.add_row(("element sets", list(self.elementSets.keys())))
        prettytable.add_row(("sections", list(self.sections.keys())))
        prettytable.add_row(("surfaces", list(self.surfaces.keys())))
        prettytable.add_row(("constraints", len(self.constraints.keys())))
        prettytable.add_row(("constraint sets", list(self.constraintSets.keys())))
        prettytable.add_row(("materials", list(self.materials.keys())))
        prettytable.add_row(("analytical fields", list(self.analyticalFields.keys())))
        prettytable.add_row(("scalar vars.", len(self.scalarVariables)))
        prettytable.add_row(("material points", len(self.materialPoints)))
        prettytable.add_row(("cells", len(self.cells)))
        prettytable.add_row(("cell elements", len(self.cellElements)))
        prettytable.add_row(("material point sets", list(self.materialPointSets.keys())))
        prettytable.add_row(("cell sets", list(self.cellSets.keys())))
        prettytable.add_row(("cell element sets", list(self.cellElementSets.keys())))
        prettytable.add_row(("particles", len(self.particles)))
        prettytable.add_row(("particle sets", list(self.particleSets.keys())))
        prettytable.add_row(("meshfree kernel functions", len(self.meshfreeKernelFunctions.keys())))
        prettytable.add_row(("particle kernel domains", list(self.particleKernelDomains.keys())))

        prettytable.min_width["model property"] = 80
        prettytable.align = "l"

        return prettytable

    def writeRestart(self, restartFile: h5py.File):
        """Write the converged state of the model to a restart checkpoint: EdelweissFE's model state (see
        :meth:`~edelweissfe.models.femodel.FEModel.writeRestart`), and the state of every material point and particle.

        The material point method and RKPM are updated-Lagrangian: the material points and particles carry the state,
        and their positions follow from it. The connectivity -- which cells hold which material points, which kernel
        functions reach which particles, where bonded kernel functions sit -- follows from the positions, and is
        rebuilt by the first connectivity update of the resumed run (:meth:`updateConnectivity`), as before every
        increment.

        Parameters
        ----------
        restartFile
            An open, writable :class:`h5py.File` (or group) to write the checkpoint into.
        """

        super().writeRestart(restartFile)

        materialPointsGroup = restartFile.create_group("materialPoints")
        for materialPoint in self.materialPoints.values():
            materialPointsGroup.create_dataset(str(materialPoint.number), data=materialPoint.getRestartData())

        particlesGroup = restartFile.create_group("particles")
        for particle in self.particles.values():
            particlesGroup.create_dataset(str(particle.number), data=particle.getRestartData())

    def readRestart(self, restartFile: h5py.File, journal: Journal = None):
        """Read the state of the model from a restart checkpoint written by :meth:`writeRestart`.

        The model must have been built as for the checkpointed run, and :meth:`prepareYourself` called.

        Parameters
        ----------
        restartFile
            An open, readable :class:`h5py.File` (or group) to read the checkpoint from.
        journal
            The journal, for the progress messages of the topology replay.
        """

        super().readRestart(restartFile, journal)

        for entities, groupName in ((self.materialPoints, "materialPoints"), (self.particles, "particles")):
            if groupName not in restartFile:
                raise RestartError("the checkpoint holds no state of the {:} of an MPMModel".format(groupName))
            group = restartFile[groupName]
            for number, entity in entities.items():
                key = str(entity.number)
                if key not in group:
                    raise RestartError("the checkpoint holds no state for {:} {:}".format(groupName, number))
                entity.readRestartData(group[key][()])

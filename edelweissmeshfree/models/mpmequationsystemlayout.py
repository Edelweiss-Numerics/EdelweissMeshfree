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

"""What the equation system of a material point or RKPM model is made of: the answer an
:class:`~edelweissmeshfree.models.mpmmodel.MPMModel` gives EdelweissFE's implicit solvers and
:class:`~edelweissmeshfree.solvers.meshfreeexplicitdynamic.MeshfreeExplicitDynamic` whenever they (re)build
their equation system, see :meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.buildEquationSystem`."""

from collections.abc import Callable

from edelweissfe.models.equationsystemlayout import EquationSystemLayout
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep


class MPMEquationSystemLayout(EquationSystemLayout):
    """The layout of the equation system of a meshfree model, on its active domain: the background cells that hold
    material points, the nodes these cells and the particles' kernel functions reach, and the finite elements with
    their nodes.

    Two kinds of nodes take part. The nodes of finite elements (and of cell elements) carry their total solution from
    increment to increment, as in a finite element model. The nodes of the grid and of the kernel functions do not: the
    material point method and RKPM are incremental on the current configuration -- the cells are reset after every
    increment, and the kernel functions move with their particles, while the state lives in the material points and
    particles. Their nodal values are the coefficients of one increment, and start every rebuilt equation system at
    zero.

    Parameters
    ----------
    model
        The model.
    dofManager
        The numbering of the degrees of freedom of the active domain.
    elements
        The finite elements, the active cells, the cell elements and the particles, in the order in which the solver
        computes them.
    activeCells
        The cells that hold material points, sorted by their number.
    nodesWithPersistentFieldValues
        The nodes that carry their total solution from increment to increment: those of the finite elements and of
        the cell elements.
    activeNodeFields
        The node fields on the nodes of the active domain, which the DOF manager numbers.
    modelForLinearSolver
        The model reduced to its active domain, for the linear solver.
    """

    def __init__(
        self,
        model,
        dofManager: DofManager,
        elements: list,
        activeCells: list,
        nodesWithPersistentFieldValues: NodeSet,
        activeNodeFields: dict,
        modelForLinearSolver,
    ):
        super().__init__(model, dofManager, elements)
        #: The cells that hold material points, sorted by their number.
        self.activeCells = activeCells
        #: The nodes that carry their total solution from increment to increment.
        self.nodesWithPersistentFieldValues = nodesWithPersistentFieldValues
        #: The node fields on the nodes of the active domain.
        self.activeNodeFields = activeNodeFields
        self.modelForLinearSolver = modelForLinearSolver
        #: The material points of the model, a list the solver hands out in chunks.
        self.materialPoints = list(model.materialPoints.values())
        #: The active cells and the cell elements, which interpolate the solution to the material points they hold.
        self.cellsInterpolatingToMaterialPoints = activeCells + list(model.cellElements.values())

    def numbersTheSameNodesAs(self, other: "MPMEquationSystemLayout") -> bool:
        """Whether this layout numbers the same nodes, with the same fields, in the same order as another one, so that
        a vector of the other layout means the same here, entry by entry.

        Parameters
        ----------
        other
            The other layout.

        Returns
        -------
        bool
            True if the numbering of the nodes is the same.
        """

        def numbering(layout):
            return [(name, list(nodeField.nodes)) for name, nodeField in layout.activeNodeFields.items()]

        return self.dofManager.nDof == other.dofManager.nDof and numbering(self) == numbering(other)

    def relocateParticlesAndConstraints(self):
        """Follow a change of the connectivity that leaves the active nodes as they are (see
        :attr:`~edelweissmeshfree.models.mpmmodel.MPMModel.activeNodesAreFixed`): the numbering of the nodes stands, so
        only the particles whose kernel functions changed, and the active constraints, are located anew in the DOF
        vector. That is what rebuilding would give, at a fraction of its cost.
        """

        self.dofManager.updateParticles(
            [
                particle
                for particleManager in self.model.particleManagers
                for particle in particleManager.particlesWithChangedKernelFunctions
            ]
        )
        self.dofManager.updateConstraints(self.model.activeConstraints())

    def readStateFromModel(self, U: DofVector, V: DofVector = None, A: DofVector = None, dynamicFields: list = None):
        """Read the state the model carries from increment to increment into the vectors of a (re)built equation
        system: the entry ``U`` of the node fields at the nodes of finite elements and cell elements, and the values of
        the scalar variables. The nodes of the grid and of the kernel functions start at zero; see
        :class:`MPMEquationSystemLayout`.

        Parameters
        ----------
        U
            The solution vector of the equation system, zero, overwritten.
        V
            The velocity vector of a solver that integrates in time; not supported, must be None.
        A
            The acceleration vector of a solver that integrates in time; not supported, must be None.
        dynamicFields
            The names of the fields whose velocity and acceleration are read; unused.

        Raises
        ------
        NotImplementedError
            If a velocity or an acceleration is to be read.
        """

        self._refuseVelocityAndAcceleration(V, A)

        if self.nodesWithPersistentFieldValues:
            for nodeField in self.model.nodeFields.values():
                self.dofManager.writeNodeFieldToDofVector(U, nodeField, "U", self.nodesWithPersistentFieldValues)

        self.dofManager.writeScalarVariablesToDofVector(U)

    def writeResultsToModel(
        self,
        U: DofVector,
        dU: DofVector,
        P: DofVector,
        V: DofVector = None,
        A: DofVector = None,
        dynamicFields: list = None,
    ):
        """Write the results of an accepted increment to the model: the solution ``U``, its increment ``dU`` and the
        internal force ``P`` to the node fields of the active domain, from there to the node fields of the model, whose
        nodes outside the active domain get zero, and the solution to the scalar variables.

        Parameters
        ----------
        U
            The solution.
        dU
            The increment of the solution.
        P
            The internal force, the sum of the elements', cells' and particles' residual contributions.
        V
            The velocity of a solver that integrates in time; not supported, must be None.
        A
            The acceleration of a solver that integrates in time; not supported, must be None.
        dynamicFields
            The names of the fields whose velocity and acceleration are written; unused.

        Raises
        ------
        NotImplementedError
            If a velocity or an acceleration is to be written.
        """

        self._refuseVelocityAndAcceleration(V, A)

        for activeNodeField in self.activeNodeFields.values():
            self.dofManager.writeDofVectorToNodeField(U, activeNodeField, "U")
            self.dofManager.writeDofVectorToNodeField(P, activeNodeField, "P")
            self.dofManager.writeDofVectorToNodeField(dU, activeNodeField, "dU")
            self.model.nodeFields[activeNodeField.name].copyEntriesFromOther(activeNodeField)

        self.dofManager.writeDofVectorToScalarVariables(U)

    def writeVelocityAndAccelerationToModel(self, V: DofVector, A: DofVector, dynamicFields: list = None):
        """Not supported: EdelweissFE's solvers do not integrate a meshfree model in time.

        Parameters
        ----------
        V
            The velocity; must be None.
        A
            The acceleration; must be None.
        dynamicFields
            The names of the fields whose velocity and acceleration are written; unused.

        Raises
        ------
        NotImplementedError
            If a velocity or an acceleration is to be written.
        """

        self._refuseVelocityAndAcceleration(V, A)

    @staticmethod
    def _refuseVelocityAndAcceleration(V: DofVector | None, A: DofVector | None):
        """Refuse a velocity or an acceleration: the velocity and the acceleration of a meshfree model do not pass
        through this layout.

        Parameters
        ----------
        V
            The velocity, or None.
        A
            The acceleration, or None.

        Raises
        ------
        NotImplementedError
            If either is given.
        """

        if V is not None or A is not None:
            raise NotImplementedError(
                "the velocity and the acceleration of a meshfree model do not pass through its equation system "
                "layout: EdelweissFE's dynamic solvers do not integrate a meshfree model in time"
            )

    def updateMaterialPoints(self, dU: DofVector, timeStep: TimeStep, forEachChunk: Callable[[list, Callable], None]):
        """Update the material points with the trial solution increment, before the solver computes the cells, which
        integrate them: the material points return to their converged state, every active cell (and cell element)
        interpolates the solution increment to the material points it holds, and the material points evaluate their
        material. Each of the three is a loop over independent entities -- a material point lies in one cell -- in the
        solver's loop over chunks, on its threads for a parallel solver.

        Parameters
        ----------
        dU
            The trial solution increment.
        timeStep
            The time step.
        forEachChunk
            The solver's loop over chunks of independent entities, see
            :meth:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST.forEachChunk`.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        def prepare(materialPoints):
            for materialPoint in materialPoints:
                materialPoint.prepareYourself(time, dT)

        def interpolate(cells):
            for cell in cells:
                cell.interpolateFieldsToMaterialPoints(dU[cell])

        def compute(materialPoints):
            for materialPoint in materialPoints:
                materialPoint.computeYourself(time, dT)

        forEachChunk(self.materialPoints, prepare)
        forEachChunk(self.cellsInterpolatingToMaterialPoints, interpolate)
        forEachChunk(self.materialPoints, compute)

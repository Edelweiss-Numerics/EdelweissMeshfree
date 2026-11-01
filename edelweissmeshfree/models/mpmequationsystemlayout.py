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
:class:`~edelweissmeshfree.models.mpmmodel.MPMModel` gives EdelweissFE's implicit solvers whenever they (re)build
their equation system, see :meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.buildEquationSystem`."""

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

    def writeSolutionToDofVector(self, U: DofVector) -> DofVector:
        """Write the solution the model carries from increment to increment into the solution vector of a (re)built
        equation system: the entry ``U`` of the node fields at the nodes of finite elements and cell elements, and the
        values of the scalar variables. The nodes of the grid and of the kernel functions start at zero; see
        :class:`MPMEquationSystemLayout`.

        Parameters
        ----------
        U
            The solution vector of the equation system, zero, overwritten.

        Returns
        -------
        DofVector
            The solution vector.
        """

        if self.nodesWithPersistentFieldValues:
            for nodeField in self.model.nodeFields.values():
                U = self.dofManager.writeNodeFieldToDofVector(U, nodeField, "U", self.nodesWithPersistentFieldValues)

        return self.dofManager.writeScalarVariablesToDofVector(U)

    def writeResultsToModel(self, U: DofVector, dU: DofVector, P: DofVector):
        """Write the results of a converged increment to the model: the solution ``U``, its increment ``dU`` and the
        reaction ``P`` to the node fields of the active domain, from there to the node fields of the model, whose nodes
        outside the active domain get zero, and the solution to the scalar variables.

        Parameters
        ----------
        U
            The solution.
        dU
            The increment of the solution.
        P
            The reaction, the sum of the elements', cells' and particles' internal forces.
        """

        for activeNodeField in self.activeNodeFields.values():
            self.dofManager.writeDofVectorToNodeField(U, activeNodeField, "U")
            self.dofManager.writeDofVectorToNodeField(P, activeNodeField, "P")
            self.dofManager.writeDofVectorToNodeField(dU, activeNodeField, "dU")
            self.model.nodeFields[activeNodeField.name].copyEntriesFromOther(activeNodeField)

        self.dofManager.writeDofVectorToScalarVariables(U)

    def updateMaterialPoints(self, dU: DofVector, timeStep: TimeStep):
        """Update the material points with the trial solution increment, before the solver computes the cells, which
        integrate them: the material points return to their converged state, every active cell (and cell element)
        interpolates the solution increment to the material points it holds, and the material points evaluate their
        material.

        Parameters
        ----------
        dU
            The trial solution increment.
        timeStep
            The time step.
        """

        time = timeStep.totalTime
        dT = timeStep.timeIncrement

        for materialPoint in self.model.materialPoints.values():
            materialPoint.prepareYourself(time, dT)

        for cell in self.activeCells:
            cell.interpolateFieldsToMaterialPoints(dU[cell])

        for cellElement in self.model.cellElements.values():
            cellElement.interpolateFieldsToMaterialPoints(dU[cellElement])

        for materialPoint in self.model.materialPoints.values():
            materialPoint.computeYourself(time, dT)

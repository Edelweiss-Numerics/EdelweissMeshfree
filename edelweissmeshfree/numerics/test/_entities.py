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
"""Stand-ins for the entities of a meshfree discrete system -- cells, cell elements, particles and constraints -- and
what compares the DOF managers that locate them."""

import numpy as np
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.variables.fieldvariable import FieldVariable

from edelweissmeshfree.numerics.dofmanager import MPMDofManager

FIELD = "displacement"
DIMENSION = 2


class Entity:
    """A cell, a cell element, a particle or a constraint: it couples the given nodes, with the field on each."""

    def __init__(self, nodes: list):
        self.dofIndicesPermutation = None
        self.scalarVariables = []
        self.couple(nodes)

    def couple(self, nodes: list):
        """Couple the given nodes, with the field on each: as a particle whose kernel functions changed, or a
        constraint that is re-assigned."""

        self.nodes = nodes
        self.fields = self.fieldsOnNodes = [[FIELD] for _ in nodes]
        self.nDof = DIMENSION * len(nodes)


def makeNodes(number: int) -> list[Node]:
    """Nodes carrying the field, labelled from zero."""

    nodes = [Node(label, np.zeros(3)) for label in range(number)]
    for node in nodes:
        node.fields[FIELD] = FieldVariable(node, FIELD)

    return nodes


def makeNodeField(nodes: list[Node]) -> NodeField:
    """The node field of all nodes."""

    return NodeField(FIELD, DIMENSION, NodeSet("all", nodes))


def makeDofManager(
    nodeField: NodeField,
    constraints: list,
    cells: list,
    cellElements: list,
    particles: list,
    initializeAccumulatedNodalFluxesFieldwise: bool = True,
) -> MPMDofManager:
    """The DOF manager of the entities."""

    return MPMDofManager(
        [nodeField],
        constraints=constraints,
        cells=cells,
        cellElements=cellElements,
        particles=particles,
        initializeAccumulatedNodalFluxesFieldwise=initializeAccumulatedNodalFluxesFieldwise,
    )


def assertEqualMappings(mapping: dict, expected: dict):
    """The mappings hold the same entities in the same order, with the same indices."""

    assert list(mapping) == list(expected)
    for entity in expected:
        np.testing.assert_array_equal(mapping[entity], expected[entity])


def assertEqualDofManagers(refreshed: MPMDofManager, fresh: MPMDofManager):
    """What a refreshed DOF manager shares with one built for the same entities."""

    for mapping, expected in (
        (refreshed.idcsOfConstraintsInDofVector, fresh.idcsOfConstraintsInDofVector),
        (refreshed.idcsOfCellsInDofVector, fresh.idcsOfCellsInDofVector),
        (refreshed.idcsOfCellElementsInDofVector, fresh.idcsOfCellElementsInDofVector),
        (refreshed.idcsOfParticlesInDofVector, fresh.idcsOfParticlesInDofVector),
        (refreshed.idcsOfHigherOrderEntitiesInDofVector, fresh.idcsOfHigherOrderEntitiesInDofVector),
    ):
        assertEqualMappings(mapping, expected)
    assert refreshed.accumulatedConstraintNDof == fresh.accumulatedConstraintNDof
    assert refreshed.largestNumberOfConstraintNDof == fresh.largestNumberOfConstraintNDof

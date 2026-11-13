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
what compares the locations of them."""

import numpy as np
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.numerics.dofnumbering import DofNumbering
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.variables.fieldvariable import FieldVariable

from edelweissmeshfree.numerics.doflocations import MPMDofLocations

FIELD = "displacement"
DIMENSION = 2


class Entity:
    """A cell, a cell element, a particle or a constraint: it couples the given nodes, with the field on each."""

    connectivityChangesWithMotion = False

    def __init__(self, nodes: list):
        self.dofIndicesPermutation = None
        self.scalarVariables = []
        self.couple(nodes)

    def couple(self, nodes: list):
        """Couple the given nodes, with the field on each: as a particle whose kernel functions changed, or a
        constraint that is re-assigned."""

        self.nodes = nodes
        self.fields = [[FIELD] for _ in nodes]
        self.nDof = DIMENSION * len(nodes)


class Particle(Entity):
    """A particle: the nodes it couples change with the motion."""

    connectivityChangesWithMotion = True


def makeNodes(number: int) -> list[Node]:
    """Nodes carrying the field, labelled from zero."""

    nodes = [Node(label, np.zeros(3)) for label in range(number)]
    for node in nodes:
        node.fields[FIELD] = FieldVariable(node, FIELD)

    return nodes


def makeNodeField(nodes: list[Node]) -> NodeField:
    """The node field of all nodes."""

    return NodeField(FIELD, DIMENSION, NodeSet("all", nodes))


def makeDofLocations(
    nodeField: NodeField,
    constraints: list,
    cells: list,
    cellElements: list,
    particles: list,
    numbering: DofNumbering = None,
) -> MPMDofLocations:
    """The locations of the entities, in the given numbering of the node field, or in a new one."""

    return MPMDofLocations(
        DofNumbering([nodeField]) if numbering is None else numbering,
        constraints=constraints,
        cells=cells,
        cellElements=cellElements,
        particles=particles,
    )


def assertEqualMappings(mapping: dict, expected: dict):
    """The mappings hold the same entities in the same order, with the same indices."""

    assert list(mapping) == list(expected)
    for entity in expected:
        np.testing.assert_array_equal(mapping[entity], expected[entity])


def assertEqualDofLocations(relocated: MPMDofLocations, fresh: MPMDofLocations):
    """What relocated locations share with locations built for the same entities."""

    assertEqualMappings(relocated.ofEntity, fresh.ofEntity)
    assert relocated.numbering.nDof == fresh.numbering.nDof
    assert relocated.elements == fresh.elements
    assert relocated.constraints == fresh.constraints
    assert relocated.cells == fresh.cells
    assert relocated.cellElements == fresh.cellElements
    assert relocated.particles == fresh.particles
    assert relocated.nAccumulatedNodalFluxesFieldwise == fresh.nAccumulatedNodalFluxesFieldwise

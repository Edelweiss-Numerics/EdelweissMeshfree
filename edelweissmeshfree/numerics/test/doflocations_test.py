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
"""MPMDofLocations whose constraints are located anew equal locations built for the same entities and constraints: the
cells, the cell elements and the particles keep what they were given, and the merged entity mapping and the accumulated
nodal fluxes follow the constraints, also if none is left. Moved particles are located anew on the
same numbering. The locations of cells, cell elements and particles are those the numbering gives their nodes."""

import numpy as np
import pytest

from edelweissmeshfree.numerics.test._entities import (
    Entity,
    assertEqualDofLocations,
    makeDofLocations,
    makeNodeField,
    makeNodes,
)


@pytest.mark.parametrize(
    "nodesOfTheConstraintsAfterwards",
    [[[3, 4]], [[0, 5], [2, 3, 4]], []],
    ids=["moved", "one more", "none left"],
)
def test_relocated_locations_equal_fresh_ones(nodesOfTheConstraintsAfterwards):
    nodes = makeNodes(6)
    nodeField = makeNodeField(nodes)
    cells = [Entity(nodes[0:3]), Entity(nodes[2:5])]
    cellElements = [Entity(nodes[0:2]), Entity(nodes[4:6])]
    particles = [Entity(nodes[1:4]), Entity(nodes[3:6])]
    constraintsBefore = [Entity([nodes[0], nodes[1]])]
    constraintsAfterwards = [Entity([nodes[i] for i in indices]) for indices in nodesOfTheConstraintsAfterwards]

    before = makeDofLocations(nodeField, constraintsBefore, cells, cellElements, particles)
    mappingBefore = before.ofEntity
    relocated = before.withNewLocationsOf(constraintsAfterwards)

    fresh = makeDofLocations(
        nodeField, constraintsAfterwards, cells, cellElements, particles, numbering=before.numbering
    )

    assertEqualDofLocations(relocated, fresh)
    assert relocated.numbering is before.numbering
    assert relocated.ofEntity is not mappingBefore, "the old locations stay valid for whoever holds them"
    assert before.ofEntity is mappingBefore and before.constraints == constraintsBefore


def test_moved_particles_are_located_anew_on_the_same_numbering_and_the_others_keep_their_locations():
    nodes = makeNodes(6)
    nodeField = makeNodeField(nodes)
    particles = [Entity(nodes[1:4]), Entity(nodes[3:6]), Entity(nodes[0:2])]
    before = makeDofLocations(nodeField, [], [], [], particles)
    locationsOfUnmoved = {p: before[p] for p in particles[1:]}

    particles[0].couple(nodes[0:3])
    relocated = before.withNewLocationsOf([], [particles[0]])

    np.testing.assert_array_equal(relocated[particles[0]], np.arange(6))
    for particle, indices in locationsOfUnmoved.items():
        assert relocated[particle] is indices
    assert relocated.particles == particles, "in the order of the particles"
    assert relocated.elements == particles
    fresh = makeDofLocations(nodeField, [], [], [], particles, numbering=before.numbering)
    assertEqualDofLocations(relocated, fresh)
    # the locations before are not changed
    assert before[particles[0]].tolist() == [2, 3, 4, 5, 6, 7]


def test_the_locations_of_cells_cell_elements_and_particles_are_those_of_their_nodes_in_the_numbering():
    """Each node numbers its two displacement components, node after node: node ``i`` has the indices ``2i, 2i + 1``."""

    nodes = makeNodes(6)
    nodeField = makeNodeField(nodes)
    cells = [Entity(nodes[0:3]), Entity(nodes[2:5])]
    cellElements = [Entity(nodes[4:6])]
    particles = [Entity(nodes[1:3])]
    constraints = [Entity([nodes[0], nodes[5]])]

    locations = makeDofLocations(nodeField, constraints, cells, cellElements, particles)

    np.testing.assert_array_equal(locations[cells[0]], [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(locations[cells[1]], [4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(locations[cellElements[0]], [8, 9, 10, 11])
    np.testing.assert_array_equal(locations[particles[0]], [2, 3, 4, 5])
    np.testing.assert_array_equal(locations[constraints[0]], [0, 1, 10, 11])
    assert locations.elements == [*cells, *cellElements, *particles]
    assert locations.constraints == constraints
    assert locations.cells == cells and locations.cellElements == cellElements and locations.particles == particles
    assert list(locations.ofEntity) == [*constraints, *cells, *cellElements, *particles]
    # Σ_entities Σ_nodes nDof(field): 3 + 3 nodes of the cells, 2 of the cell element, 2 of the particle, 2 of the
    # constraint, each with two components
    assert locations.nAccumulatedNodalFluxesFieldwise == {"displacement": 2 * (3 + 3 + 2 + 2 + 2)}

    vector = locations.constructDofVector()
    vector[cellElements[0]] = 1.0
    np.testing.assert_array_equal(np.flatnonzero(vector), [8, 9, 10, 11])

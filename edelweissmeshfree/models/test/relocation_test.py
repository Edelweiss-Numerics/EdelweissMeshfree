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
"""A meshfree discrete system follows a change of the connectivity that leaves the active nodes as they are by locating
the particles whose kernel functions changed, and the constraints, anew, in a new discrete system on the same
numbering: its locations equal those built for the changed entities, and its constraints are those of the model that
are active, none if all have left."""

from types import SimpleNamespace

import pytest
from edelweissfe.sets.nodeset import NodeSet

from edelweissmeshfree.models.mpmdiscretesystem import MPMDiscreteSystem
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
    ids=["moved", "one more", "all left"],
)
def test_the_system_locates_particles_and_constraints_anew_as_a_fresh_one_has_them(nodesOfTheConstraintsAfterwards):
    nodes = makeNodes(6)
    nodeField = makeNodeField(nodes)
    cells = [Entity(nodes[0:3]), Entity(nodes[2:5])]
    particles = [Entity(nodes[1:4]), Entity(nodes[3:6])]
    constraints = [Entity([nodes[0], nodes[1]])]

    model = SimpleNamespace(
        materialPoints={},
        particles=dict(enumerate(particles)),
        cellElements={},
        particleManagers=[SimpleNamespace(particlesWithChangedKernelFunctions=[])],
        activeConstraints=lambda: constraints,
    )
    system = MPMDiscreteSystem(
        model,
        makeDofLocations(nodeField, constraints, cells, [], particles),
        NodeSet("persistent", []),
        NodeSet("volatile", []),
        None,
    )
    assert system.constraints == constraints
    constraintsBefore = list(constraints)

    # the kernel functions of one particle now reach other nodes, and the constraints change
    particles[0].couple(nodes[0:3])
    model.particleManagers[0].particlesWithChangedKernelFunctions = [particles[0]]
    constraints[:] = [Entity([nodes[i] for i in indices]) for indices in nodesOfTheConstraintsAfterwards]

    relocated = system.withNewLocations()

    assert relocated.constraints == constraints
    assert relocated.elements == [*cells, *particles]
    assert relocated.numbering is system.numbering
    assertEqualDofLocations(
        relocated.locations,
        makeDofLocations(nodeField, constraints, cells, [], particles, numbering=system.numbering),
    )
    # the discrete system it follows from is not changed
    assert system.constraints == constraintsBefore

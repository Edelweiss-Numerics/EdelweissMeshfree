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
"""An MPMDofManager whose constraints are located anew equals one built for the same entities and constraints: the
cells, the cell elements and the particles keep what they were given, and the merged entity mapping, the accumulated
nodal fluxes and the DOF counts follow the constraints, also if none is left."""

import pytest

from edelweissmeshfree.numerics.test._entities import (
    Entity,
    assertEqualDofManagers,
    makeDofManager,
    makeNodeField,
    makeNodes,
)


@pytest.mark.parametrize("initializeAccumulatedNodalFluxesFieldwise", [True, False])
@pytest.mark.parametrize(
    "nodesOfTheConstraintsAfterwards",
    [[[3, 4]], [[0, 5], [2, 3, 4]], []],
    ids=["moved", "one more", "none left"],
)
def test_a_refreshed_dof_manager_equals_a_fresh_one(
    nodesOfTheConstraintsAfterwards, initializeAccumulatedNodalFluxesFieldwise
):
    nodes = makeNodes(6)
    nodeField = makeNodeField(nodes)
    cells = [Entity(nodes[0:3]), Entity(nodes[2:5])]
    cellElements = [Entity(nodes[0:2]), Entity(nodes[4:6])]
    particles = [Entity(nodes[1:4]), Entity(nodes[3:6])]
    constraintsBefore = [Entity([nodes[0], nodes[1]])]
    constraintsAfterwards = [Entity([nodes[i] for i in indices]) for indices in nodesOfTheConstraintsAfterwards]

    refreshed = makeDofManager(
        nodeField, constraintsBefore, cells, cellElements, particles, initializeAccumulatedNodalFluxesFieldwise
    )
    mappingBefore = refreshed.idcsOfHigherOrderEntitiesInDofVector
    refreshed.refreshConstraintIndices(constraintsAfterwards)

    fresh = makeDofManager(
        nodeField, constraintsAfterwards, cells, cellElements, particles, initializeAccumulatedNodalFluxesFieldwise
    )

    assertEqualDofManagers(refreshed, fresh)
    assert refreshed.idcsOfHigherOrderEntitiesInDofVector is not mappingBefore, "consumers cache plans on its identity"
    if initializeAccumulatedNodalFluxesFieldwise:
        assert refreshed.nAccumulatedNodalFluxesFieldwise == fresh.nAccumulatedNodalFluxesFieldwise

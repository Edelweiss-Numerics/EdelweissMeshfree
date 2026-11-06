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
"""Guards the order in which material points are handed to the cells.

The material point managers build each cell's list of material points by iterating the material point set
they were given, and a cell sums the contributions of its material points in that order. Loads and constraints
defined on a material point set sum in its order, too. A plain ``set`` of material points iterates in the order
of their memory addresses, which differs from run to run, and so did the rounding of these sums.

The invariant asserted here -- material points come out in the order they were given -- is what makes these
sums reproducible across processes. It is asserted against the given order, not by building the same set twice:
within one process a ``set`` of the same objects iterates consistently, so such a test would pass on the
address-ordered code as well.
"""

import numpy as np
import pytest
from edelweissfe.points.node import Node

from edelweissmeshfree.cells.test.cell import Cell
from edelweissmeshfree.materialpoints.test.mp import MaterialPoint
from edelweissmeshfree.mpmmanagers.simplempmmanager import SimpleMaterialPointManager
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.sets.materialpointset import MaterialPointSet

_nCellsPerDirection = 4
_nMaterialPointsPerDirection = 16


def _unitCellGrid() -> list[Cell]:
    """A grid of square cells of unit size."""
    n = _nCellsPerDirection
    nodes = {
        (i, j): Node(1 + i + j * (n + 1), np.array([float(i), float(j)])) for j in range(n + 1) for i in range(n + 1)
    }
    return [
        Cell("test", 1 + i + j * n, [nodes[i, j], nodes[i + 1, j], nodes[i + 1, j + 1], nodes[i, j + 1]])
        for j in range(n)
        for i in range(n)
    ]


def _shuffledMaterialPoints() -> list[MaterialPoint]:
    """Material points on a regular grid strictly inside the cells, several per cell, in a shuffled order."""
    spacing = _nCellsPerDirection / _nMaterialPointsPerDirection
    n = _nMaterialPointsPerDirection
    coordinates = [np.array([(i + 0.5) * spacing, (j + 0.5) * spacing]) for j in range(n) for i in range(n)]
    materialPoints = [MaterialPoint("test", number, xy, spacing**2, None) for number, xy in enumerate(coordinates, 1)]
    np.random.default_rng(20261031).shuffle(materialPoints)
    return materialPoints


def test_material_points_iterate_in_the_given_order():
    """A material point set must iterate in the order its material points were given, not in memory order."""

    givenOrder = _shuffledMaterialPoints()
    assert [mp.number for mp in givenOrder] != sorted(mp.number for mp in givenOrder)

    materialPointSet = MaterialPointSet("shuffled", givenOrder)

    assert [mp.number for mp in materialPointSet] == [mp.number for mp in givenOrder]


@pytest.mark.parametrize("managerType", [SimpleMaterialPointManager, SmartMaterialPointManager])
def test_managers_hand_material_points_to_cells_in_the_order_of_the_set(managerType):
    """Each cell must receive its material points in the order of the material point set."""

    cells = _unitCellGrid()
    givenOrder = _shuffledMaterialPoints()
    materialPointSet = MaterialPointSet("shuffled", givenOrder)

    if managerType is SmartMaterialPointManager:
        manager = SmartMaterialPointManager(cells, materialPointSet, 2, options={"KDTreeLevels": 2})
    else:
        manager = SimpleMaterialPointManager(cells, materialPointSet)
    manager.updateConnectivity()

    positionInGivenOrder = {mp: position for position, mp in enumerate(givenOrder)}
    activeCells = list(manager.getActiveCells())
    assert len(activeCells) == len(cells)
    for cell in activeCells:
        # the test cell keeps the list handed over by the manager
        positions = [positionInGivenOrder[mp] for mp in cell.materialPoints]
        assert len(positions) == (_nMaterialPointsPerDirection // _nCellsPerDirection) ** 2
        assert positions == sorted(positions)

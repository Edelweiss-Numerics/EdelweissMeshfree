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
"""A cell is an entity of the discrete system like an element: it is derived from the same base, and is located by the
same protocol."""

import numpy as np
from edelweissfe.nodecouplingentity.base.nodecouplingentity import (
    BaseNodeCouplingEntity,
)
from edelweissfe.numerics.dofnumbering import DofNumbering

from edelweissmeshfree.cells.pythoncell.cell import PythonCell
from edelweissmeshfree.numerics.doflocations import MPMDofLocations
from edelweissmeshfree.numerics.test._entities import makeNodeField, makeNodes


def _unitSquareCell():
    nodes = makeNodes(4)
    for node, coordinates in zip(nodes, [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]):
        node.coordinates = np.array([*coordinates, 0.0])

    return PythonCell("quad4", 1, nodes), nodes


def test_a_cell_is_located_by_the_protocol_of_an_entity():
    cell, nodes = _unitSquareCell()

    locations = MPMDofLocations(DofNumbering([makeNodeField(nodes)]), cells=[cell])

    assert isinstance(cell, BaseNodeCouplingEntity)
    assert cell.nodes == nodes and cell.nNodes == 4 and cell.fields == [["displacement"]] * 4
    assert list(cell.scalarVariables) == []
    np.testing.assert_array_equal(locations[cell], np.arange(8))
    assert locations.elements == [cell]

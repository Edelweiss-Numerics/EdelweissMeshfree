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
"""The aggregation of the background grid (AgMPM, Coombs, Bird and Pretti, CMAME 442 (2025) 118012): the classification
of cells and nodes, the extension of interior cells to exterior nodes (Fig. 5 of the paper), and the extension matrix
on a numbering."""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofnumbering import DofNumbering

from edelweissmeshfree.fields.nodefield import MPMNodeField
from edelweissmeshfree.generators import rectangulargridgenerator
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.numerics.aggregation import (
    CellAggregation,
    GridFaceAdjacency,
    extensionMatrix,
)

NX = NY = 4


def _grid(nX=NX, nY=NY, cellType="Displacement/Quad4"):
    """A grid of unit square cells, cell (i, j) spanning [i, i + 1] x [j, j + 1]."""

    model = MPMModel(2)
    rectangulargridgenerator.generateModelData(
        model,
        Journal(),
        x0=0.0,
        l=float(nX),
        y0=0.0,
        h=float(nY),
        nX=nX,
        nY=nY,
        cellProvider="LagrangianMarmotCell",
        cellType=cellType,
    )
    model.prepareYourself(Journal())
    return model


def _cell(i, j, nY=NY):
    """The number of cell (i, j) of the rectangular grid generator."""

    return 1 + i * nY + j


def _node(model, x, y):
    return next(n for n in model.nodes.values() if np.allclose(n.coordinates, (x, y)))


#: A staircase: four cells in the bottom row, three in the second, two in the third, none in the top row.
STAIRCASE = [(i, 0) for i in range(4)] + [(i, 1) for i in range(3)] + [(i, 2) for i in range(2)]


@pytest.fixture
def staircase():
    model = _grid()
    activeCells = [model.cells[_cell(i, j)] for i, j in STAIRCASE]
    return model, CellAggregation(GridFaceAdjacency(model.cells.values()), activeCells)


def test_face_neighbours_of_a_grid():
    model = _grid()
    adjacency = GridFaceAdjacency(model.cells.values())
    # a corner cell has two face neighbours, an inner cell four; cells sharing a corner only are no face neighbours
    assert adjacency.faceNeighbours[_cell(0, 0)] == sorted([_cell(1, 0), _cell(0, 1)])
    assert adjacency.faceNeighbours[_cell(1, 1)] == sorted([_cell(0, 1), _cell(2, 1), _cell(1, 0), _cell(1, 2)])


def test_classification_of_cells_and_nodes(staircase):
    model, aggregation = staircase

    # faces on the outer boundary of the grid do not count, faces towards inactive cells do
    assert aggregation.boundaryCells == sorted(_cell(i, j) for i, j in [(3, 0), (2, 1), (0, 2), (1, 2)])
    assert aggregation.interiorCells == sorted(_cell(i, j) for i, j in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)])

    exterior = sorted(_node(model, x, y).label for x, y in [(4, 0), (4, 1), (3, 2), (0, 3), (1, 3), (2, 3)])
    assert [node.label for node in aggregation.exteriorNodes] == exterior

    # an interior face neighbour; of two, the lower cell number
    assert aggregation.connectedInteriorCell == {
        _cell(3, 0): _cell(2, 0),
        _cell(2, 1): _cell(1, 1),
        _cell(0, 2): _cell(0, 1),
        _cell(1, 2): _cell(1, 1),
    }


def test_corner_case_takes_the_closest_interior_cell():
    model = _grid(3, 3)
    # a 2 x 2 block and an isolated cell, which shares no face with an interior cell
    active = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)]
    aggregation = CellAggregation(
        GridFaceAdjacency(model.cells.values()), [model.cells[_cell(i, j, 3)] for i, j in active]
    )
    assert aggregation.interiorCells == [_cell(0, 0, 3)]
    assert aggregation.connectedInteriorCell[_cell(2, 2, 3)] == _cell(0, 0, 3)


def _weights(model, aggregation, x, y):
    node = _node(model, x, y)
    _, weights = aggregation.extension[node.label]
    return {tuple(np.round(master.coordinates, 12)): w for master, w in weights}


def test_extension_of_figure_5(staircase):
    """Fig. 5 of the paper: a node one cell beyond its interior cell, at xi = 3, takes N = -1 and 2 of the two nodes
    in line with it."""

    model, aggregation = staircase

    # the interior cell (2, 0) extended to the node (4, 0) at (xi, eta) = (3, -1)
    assert _weights(model, aggregation, 4, 0) == pytest.approx({(2.0, 0.0): -1.0, (3.0, 0.0): 2.0})
    assert _weights(model, aggregation, 4, 1) == pytest.approx({(2.0, 1.0): -1.0, (3.0, 1.0): 2.0})
    # the interior cell (1, 1) extended upwards and to the right
    assert _weights(model, aggregation, 3, 2) == pytest.approx({(1.0, 2.0): -1.0, (2.0, 2.0): 2.0})
    assert _weights(model, aggregation, 2, 3) == pytest.approx({(2.0, 1.0): -1.0, (2.0, 2.0): 2.0})
    # equally close interior cells (0, 1) and (1, 1): the lower cell number
    assert _weights(model, aggregation, 1, 3) == pytest.approx({(1.0, 1.0): -1.0, (1.0, 2.0): 2.0})


def test_extrapolation_to_a_diagonal_corner():
    """Node 8 of Fig. 5: at (xi, eta) = (3, 3) all four nodes of the interior cell take part."""

    model = _grid(2, 2)
    cell = model.cells[_cell(0, 0, 2)]
    N = cell.getInterpolationVector(np.array([2.0, 2.0]))
    # nodes (0,0), (1,0), (1,1), (0,1): N = (1 -+ 3)(1 -+ 3) / 4
    assert N == pytest.approx([1.0, -2.0, 4.0, -2.0])


def _numbering(model, aggregation):
    activeNodes = sorted(
        {node for c in aggregation.activeCells for node in model.cells[c].nodes}, key=lambda node: node.label
    )
    field = MPMNodeField("displacement", 2, activeNodes)
    return DofNumbering([field]), activeNodes


def test_extension_matrix_is_a_partition_of_unity_and_reproduces_linear_fields(staircase):
    model, aggregation = staircase
    numbering, activeNodes = _numbering(model, aggregation)
    records = aggregation.recordsIn(numbering)

    # every component of every exterior node
    assert len(records) == 2 * len(aggregation.exteriorNodes)
    for _, masters in records:
        assert sum(w for _, w in masters) == pytest.approx(1.0)

    E, independent = extensionMatrix(records, numbering.nDof)
    assert E.shape == (numbering.nDof, numbering.nDof - len(records))
    # the identity on the independent degrees of freedom
    assert np.allclose(E[independent].toarray(), np.eye(len(independent)))

    # a rigid body motion and a general linear field, given on the independent degrees of freedom, are reproduced on
    # all nodes
    indices = numbering.idcsOfFieldVariablesInDofVector
    theta = 0.3
    rigid = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) - np.eye(2)
    for A, b in ((rigid, np.array([0.5, -2.0])), (np.array([[0.1, -0.7], [1.3, 0.2]]), np.array([3.0, 1.0]))):
        u = np.zeros(numbering.nDof)
        for node in activeNodes:
            u[indices[node.fields["displacement"]]] = A @ node.coordinates + b
        assert np.allclose(E @ u[independent], u)


def test_records_keep_prescribed_dofs_independent(staircase):
    model, aggregation = staircase
    numbering, _ = _numbering(model, aggregation)
    records = aggregation.recordsIn(numbering)
    prescribed = [records[0][0]]

    E, independent = extensionMatrix(records, numbering.nDof, keptSlaveDofs=prescribed)
    assert prescribed[0] in independent
    assert E.shape[1] == numbering.nDof - len(records) + 1


def test_hexa8():
    """Hexahedra: a 3 x 3 x 3 block of a 4 x 4 x 4 grid; the interior cells are those not on the faces of the block
    that point into the grid, i.e. the 2 x 2 x 2 cells at the origin."""

    from edelweissmeshfree.generators import boxgridgenerator

    model = MPMModel(3)
    boxgridgenerator.generateModelData(
        model,
        Journal(),
        x0=0.0,
        y0=0.0,
        z0=0.0,
        l=4.0,
        h=4.0,
        d=4.0,
        nX=4,
        nY=4,
        nZ=4,
        cellClass="LagrangianMarmotCell",
        cellType="Displacement/Hexa8",
    )
    model.prepareYourself(Journal())

    centres = {c.cellNumber: np.mean([n.coordinates for n in c.nodes], axis=0) for c in model.cells.values()}
    activeCells = [model.cells[c] for c, x in centres.items() if np.all(x < 3.0)]
    aggregation = CellAggregation(GridFaceAdjacency(model.cells.values()), activeCells)

    assert len(aggregation.activeCells) == 27
    assert sorted(aggregation.interiorCells) == sorted(c for c, x in centres.items() if np.all(x < 2.0))
    # the nodes of the block that lie on its faces x = 3, y = 3 or z = 3
    assert len(aggregation.exteriorNodes) == 4**3 - 3**3

    numbering, activeNodes = _numbering(model, aggregation)
    numbering = DofNumbering([MPMNodeField("displacement", 3, activeNodes)])
    records = aggregation.recordsIn(numbering)
    E, independent = extensionMatrix(records, numbering.nDof)

    indices = numbering.idcsOfFieldVariablesInDofVector
    A = np.array([[0.1, -0.7, 0.2], [1.3, 0.2, -0.4], [0.3, 0.5, 0.9]])
    u = np.zeros(numbering.nDof)
    for node in activeNodes:
        u[indices[node.fields["displacement"]]] = A @ node.coordinates + 1.0
    assert np.allclose(E @ u[independent], u)

    # the node at the far corner of the block, (3, 3, 3), two cells beyond the interior cell (1, 1, 1): xi = eta =
    # zeta = 3 with all eight nodes of that cell
    corner = next(n for n in aggregation.exteriorNodes if np.allclose(n.coordinates, 3.0))
    _, weights = aggregation.extension[corner.label]
    assert len(weights) == 8
    assert sum(w for _, w in weights) == pytest.approx(1.0)


def test_bspline_cells_are_not_supported():
    from edelweissmeshfree.generators import rectangularbsplinegridgenerator

    model = MPMModel(2)
    rectangularbsplinegridgenerator.generateModelData(
        model,
        Journal(),
        x0=0.0,
        l=4.0,
        y0=0.0,
        h=4.0,
        nX=4,
        nY=4,
        cellProvider="BSplineMarmotCell",
        cellType="Displacement/BSpline/2",
        order=2,
    )
    with pytest.raises(NotImplementedError, match="B-spline"):
        GridFaceAdjacency(model.cells.values())


@pytest.mark.parametrize(
    "cellType",
    [
        "Displacement/Quad4",
        "DisplacementFBar/Quad4",
        "GradientEnhancedFiniteStrain/Quad4",
        "GradientEnhancedMicropolar/Quad4",
    ],
)
def test_every_field_of_every_cell_type_is_aggregated(cellType):
    """The extension is geometric: the same scalar weights act on every field the nodes carry, whatever the physics
    of the cell (displacement only, also with F-bar; displacement and a nonlocal field; displacement, rotation and a
    nonlocal field)."""

    model = _grid(cellType=cellType)
    activeCells = [model.cells[_cell(i, j)] for i, j in STAIRCASE]
    aggregation = CellAggregation(GridFaceAdjacency(model.cells.values()), activeCells)

    activeNodes = sorted({node for cell in activeCells for node in cell.nodes}, key=lambda node: node.label)
    fields = [MPMNodeField(f.name, f.dimension, activeNodes) for f in model.nodeFields.values()]
    numbering = DofNumbering(fields)
    records = aggregation.recordsIn(numbering)

    nDofPerNode = sum(f.dimension for f in model.nodeFields.values())
    assert len(records) == nDofPerNode * len(aggregation.exteriorNodes)

    # every field of the nodes reproduces a linear field
    E, independent = extensionMatrix(records, numbering.nDof)
    indices = numbering.idcsOfFieldVariablesInDofVector
    u = np.zeros(numbering.nDof)
    for node in activeNodes:
        for k, field in enumerate(model.nodeFields):
            dofs = indices[node.fields[field]]
            u[dofs] = (k + 1) * (0.3 * node.coordinates[0] - 0.8 * node.coordinates[1]) + np.arange(len(dofs))
    assert np.allclose(E @ u[independent], u)


class _Manager:
    """An MPM manager whose active cells the test sets."""

    def __init__(self, cells):
        self.cells = cells

    def getActiveCells(self):
        return self.cells


def test_constraint_without_interior_cell_warns_and_aggregates_nothing():
    from edelweissmeshfree.constraints.aggregatedmpm import (
        AggregatedMaterialPointMethod,
    )

    model = _grid()
    messages = []
    journal = Journal()
    journal.message = lambda message, *args, **kwargs: messages.append(message)

    # a single active cell is a boundary cell: nothing to extend
    manager = _Manager([model.cells[_cell(1, 1)]])
    model.mpmManagers.append(manager)
    constraint = AggregatedMaterialPointMethod("agmpm", model, journal)
    aggregation = CellAggregation(constraint.adjacency, manager.cells)
    numbering, _ = _numbering(model, aggregation)

    assert constraint.getMultiPointConstraints(numbering) == []
    assert any("WARNING" in message and "no interior cell" in message for message in messages)

    # the staircase: records, computed once per numbering and active cells
    manager.cells = [model.cells[_cell(i, j)] for i, j in STAIRCASE]
    numbering, _ = _numbering(model, CellAggregation(constraint.adjacency, manager.cells))
    records = constraint.getMultiPointConstraints(numbering)
    assert len(records) == 12
    assert constraint.getMultiPointConstraints(numbering) is records
    assert constraint.slavesYieldToDirichlet


class _FailingInverseMap:
    """A cell whose inverse map fails, as Marmot's may far outside a strongly distorted cell."""

    def __init__(self, cell):
        self.cellNumber = cell.cellNumber
        self.nodes = cell.nodes

    def getInterpolationVector(self, x):
        raise RuntimeError("inverse map did not converge")


def test_a_failing_inverse_map_falls_back_to_the_next_interior_cell(staircase):
    model, aggregation = staircase
    node = _node(model, 1, 3)
    boundaryCells = [_cell(0, 2), _cell(1, 2)]
    # the node takes the interior cell (0, 1), equally close as (1, 1), by its lower number
    assert aggregation._extensionTo(node, boundaryCells)[0] == _cell(0, 1)

    aggregation.adjacency.cells[_cell(0, 1)] = _FailingInverseMap(aggregation.adjacency.cells[_cell(0, 1)])
    interiorCell, weights = aggregation._extensionTo(node, boundaryCells)
    assert interiorCell == _cell(1, 1)
    assert {tuple(master.coordinates): w for master, w in weights} == pytest.approx({(1.0, 1.0): -1.0, (1.0, 2.0): 2.0})

    aggregation.adjacency.cells[_cell(1, 1)] = _FailingInverseMap(aggregation.adjacency.cells[_cell(1, 1)])
    with pytest.raises(RuntimeError, match="no interior cell extends"):
        aggregation._extensionTo(node, boundaryCells)

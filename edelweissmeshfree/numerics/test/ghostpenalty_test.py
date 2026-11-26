"""The ghost penalty of the background grid (Coombs, IJNME 124 (2023) 4841-4875): the faces, the penalty matrix of a
two-cell patch against the hand computation, consistency (no penalty on linear fields), symmetry and positive
semi-definiteness, every field of every cell type, and the constraint in the implicit system."""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofnumbering import DofNumbering

from edelweissmeshfree.fields.nodefield import MPMNodeField
from edelweissmeshfree.generators import rectangulargridgenerator
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.numerics.aggregation import GridFaceAdjacency
from edelweissmeshfree.numerics.ghostpenalty import (
    boundaryCellsAndGhostFaces,
    ghostPenaltyInNumbering,
    ghostPenaltyMatrix,
)

NX = NY = 4

#: A staircase: four cells in the bottom row, three in the second, two in the third, none in the top row.
STAIRCASE = [(i, 0) for i in range(4)] + [(i, 1) for i in range(3)] + [(i, 2) for i in range(2)]


def _grid(nX=NX, nY=NY, length=None, h=None, cellType="Displacement/Quad4"):
    model = MPMModel(2)
    rectangulargridgenerator.generateModelData(
        model,
        Journal(),
        x0=0.0,
        l=float(nX) if length is None else length,
        y0=0.0,
        h=float(nY) if h is None else h,
        nX=nX,
        nY=nY,
        cellProvider="LagrangianMarmotCell",
        cellType=cellType,
    )
    model.prepareYourself(Journal())
    return model


def _cell(i, j, nY=NY):
    return 1 + i * nY + j


class _Manager:
    """An MPM manager whose active cells the test sets."""

    def __init__(self, cells):
        self.cells = cells

    def getActiveCells(self):
        return self.cells


def test_cells_give_the_gradients_of_their_extrapolated_basis():
    """Marmot's Cell::getInterpolationGradient, inside and outside of the cell, against central differences of
    getInterpolationVector."""

    model = _grid(nX=1, nY=1, length=2.0, h=1.0)
    cell = model.cells[1]
    for x in ([0.3, 0.7], [2.0, 0.5], [3.5, -1.0]):
        x = np.array(x)
        G = cell.getInterpolationGradient(x)
        assert G.shape == (2, 4)
        eps = 1e-6
        Gfd = np.array(
            [
                (cell.getInterpolationVector(x + eps * e) - cell.getInterpolationVector(x - eps * e)) / (2 * eps)
                for e in np.eye(2)
            ]
        )
        assert np.allclose(G, Gfd, atol=1e-8)
        assert np.allclose(G.sum(axis=1), 0.0)


def test_boundary_cells_and_ghost_faces_of_the_staircase():
    """Section 4.2 of the paper: the boundary cells have an inactive face neighbour (the outer boundary of the grid does
    not count); the ghost faces are their faces with active cells."""

    model = _grid()
    adjacency = GridFaceAdjacency(model.cells.values())
    boundary, faces = boundaryCellsAndGhostFaces(adjacency, [_cell(i, j) for i, j in STAIRCASE])

    assert boundary == sorted(_cell(i, j) for i, j in [(3, 0), (2, 1), (0, 2), (1, 2)])
    expected = [
        ((3, 0), (2, 0)),
        ((2, 1), (1, 1)),
        ((2, 1), (2, 0)),
        ((0, 2), (0, 1)),
        ((0, 2), (1, 2)),
        ((1, 2), (1, 1)),
    ]
    assert faces == sorted(tuple(sorted((_cell(*a), _cell(*b)))) for a, b in expected)


@pytest.mark.parametrize("h", [1.0, 0.5])
def test_penalty_of_two_cells_is_the_hand_computation(h):
    """Two square cells of size h side by side, face x = h, n = (1, 0). With y in [0, 1] along the face, the jumps of
    the normal derivative are, times h, (1 - y) p + y q with, for the nodes (0,0), (h,0), (h,h), (0,h), (2h,0),
    (2h,h), p = (-1, 2, 0, 0, -1, 0) and q = (0, 0, 2, -1, 0, -1); J = h^3/3 * h * h^-2 * (pp/3 + (pq + qp)/6 + qq/3).
    """

    model = _grid(nX=2, nY=1, length=2 * h, h=h)
    adjacency = GridFaceAdjacency(model.cells.values())
    _, faces = boundaryCellsAndGhostFaces(adjacency, [1, 2])
    assert faces == []  # no inactive neighbour: no boundary cell

    J, nodes = ghostPenaltyMatrix(adjacency, [(1, 2)])

    order = [(0, 0), (1, 0), (1, 1), (0, 1), (2, 0), (2, 1)]
    index = {tuple(np.round(np.asarray(n.coordinates) / h).astype(int)): i for i, n in enumerate(nodes)}
    perm = [index[o] for o in order]
    Jh = J.toarray()[np.ix_(perm, perm)]

    p = np.array([-1, 2, 0, 0, -1, 0], dtype=float)
    q = np.array([0, 0, 2, -1, 0, -1], dtype=float)
    expected = h**2 / 3 * (np.outer(p, p) / 3 + (np.outer(p, q) + np.outer(q, p)) / 6 + np.outer(q, q) / 3)
    assert np.allclose(Jh, expected, rtol=1e-12, atol=1e-14)

    # the Dirichlet form of Sticko et al. differs by h^-2
    Jd, _ = ghostPenaltyMatrix(adjacency, [(1, 2)], faceScaling="dirichlet")
    assert np.allclose(Jd.toarray(), J.toarray() / h**2)


def test_penalty_is_consistent_symmetric_and_positive_semidefinite():
    """No penalty on any globally linear field (continuous gradient); a kink over a ghost face is penalised."""

    model = _grid()
    adjacency = GridFaceAdjacency(model.cells.values())
    _, faces = boundaryCellsAndGhostFaces(adjacency, [_cell(i, j) for i, j in STAIRCASE])
    J, nodes = ghostPenaltyMatrix(adjacency, faces)
    J = J.toarray()
    X = np.array([n.coordinates for n in nodes])

    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b, c = rng.normal(size=3)
        assert np.allclose(J @ (a + b * X[:, 0] + c * X[:, 1]), 0.0, atol=1e-12)

    assert np.allclose(J, J.T)
    eigenvalues = np.linalg.eigvalsh(J)
    assert eigenvalues.min() > -1e-12 * eigenvalues.max()
    # the global bilinear field xy is in the space with a continuous gradient: no penalty; the interpolant of x^2 or
    # y^2 kinks over every vertical or horizontal face
    assert abs((X[:, 0] * X[:, 1]) @ J @ (X[:, 0] * X[:, 1])) < 1e-12
    assert (X[:, 0] ** 2) @ J @ (X[:, 0] ** 2) > 1e-3
    assert (X[:, 1] ** 2) @ J @ (X[:, 1] ** 2) > 1e-3
    # a kink: |x - 2| is linear in every cell, its gradient jumps over the ghost face at x = 2
    kink = np.abs(X[:, 0] - 2.0)
    assert kink @ J @ kink > 1e-3


@pytest.mark.parametrize("cellType", ["Displacement/Quad4", "GradientEnhancedFiniteStrain/Quad4"])
def test_every_field_is_penalised_with_its_own_parameter(cellType):
    """Physics-agnostic: the same scalar J for every field, times the field's parameter, component by component."""

    model = _grid(cellType=cellType)
    activeCells = [model.cells[_cell(i, j)] for i, j in STAIRCASE]
    activeNodes = sorted({node for cell in activeCells for node in cell.nodes}, key=lambda node: node.label)
    fields = [MPMNodeField(f.name, f.dimension, activeNodes) for f in model.nodeFields.values()]
    numbering = DofNumbering(fields)
    names = list(model.nodeFields)
    if cellType.startswith("GradientEnhanced"):
        assert len(names) == 2

    parameters = {name: 10.0 ** (k + 1) for k, name in enumerate(names)}
    K = ghostPenaltyInNumbering(model, activeCells, numbering, parameters).toarray()

    with pytest.raises(ValueError, match="no penalty parameter"):
        ghostPenaltyInNumbering(model, activeCells, numbering, {n: 1.0 for n in names[:-1]})

    adjacency = GridFaceAdjacency(model.cells.values())
    _, faces = boundaryCellsAndGhostFaces(adjacency, activeCells)
    J, nodes = ghostPenaltyMatrix(adjacency, faces)
    J = J.toarray()
    indices = numbering.idcsOfFieldVariablesInDofVector
    for name in names:
        dim = model.nodeFields[name].dimension
        dofs = np.array([indices[node.fields[name]] for node in nodes]).reshape(len(nodes), dim)
        for c in range(dim):
            assert np.allclose(K[np.ix_(dofs[:, c], dofs[:, c])], parameters[name] * J)
        # no coupling between components or fields
        allDofs = np.concatenate([np.asarray(indices[node.fields[n]]) for n in names for node in nodes])
        block = K[np.ix_(dofs[:, 0], allDofs)]
        assert np.count_nonzero(np.abs(block) > 0) == np.count_nonzero(np.abs(J) > 0)


def test_hexa8_faces_and_consistency():
    model = MPMModel(3)
    from edelweissmeshfree.generators import boxgridgenerator

    boxgridgenerator.generateModelData(
        model,
        Journal(),
        x0=0.0,
        y0=0.0,
        z0=0.0,
        l=3.0,
        h=3.0,
        d=3.0,
        nX=3,
        nY=3,
        nZ=3,
        cellClass="LagrangianMarmotCell",
        cellType="Displacement/Hexa8",
    )
    model.prepareYourself(Journal())
    adjacency = GridFaceAdjacency(model.cells.values())
    centres = adjacency.centres
    active = [c for c in adjacency.cells if centres[c][2] < 2.0]  # the bottom two layers
    boundary, faces = boundaryCellsAndGhostFaces(adjacency, active)
    assert len(boundary) == 9  # the second layer
    assert len(faces) == 9 + 12  # down to the first layer, and within the second layer
    J, nodes = ghostPenaltyMatrix(adjacency, faces)
    J = J.toarray()
    X = np.array([n.coordinates for n in nodes])
    assert np.allclose(J @ (1.0 + X @ np.array([0.3, -0.2, 0.7])), 0.0, atol=1e-12)
    assert np.allclose(J, J.T)
    assert np.linalg.eigvalsh(J).min() > -1e-12
    for d in range(3):
        assert (X[:, d] ** 2) @ J @ (X[:, d] ** 2) > 1e-3


def test_quadratic_cells_are_not_supported():
    model = _grid(nX=2, nY=1, cellType="Displacement/Quad4")
    adjacency = GridFaceAdjacency(model.cells.values())

    class _Quad8:
        nNodes = 8
        cellNumber = 1

    adjacency.cells[1] = _Quad8()
    with pytest.raises(NotImplementedError, match="multilinear"):
        ghostPenaltyMatrix(adjacency, [(1, 2)])


def test_the_constraint_adds_the_penalty_force_of_the_increment_and_its_tangent():
    from edelweissmeshfree.constraints.ghoststabilisation import GhostStabilisation

    model = _grid(cellType="GradientEnhancedFiniteStrain/Quad4")
    names = list(model.nodeFields)
    manager = _Manager([model.cells[_cell(i, j)] for i, j in STAIRCASE])
    model.mpmManagers.append(manager)
    messages = []
    journal = Journal()
    journal.message = lambda message, *args, **kwargs: messages.append(message)

    ghost = GhostStabilisation("ghost", model, {names[0]: 2.0, names[1]: 0.0}, journal)
    assert not ghost.active
    assert ghost.updateConnectivity(model)
    assert ghost.active and len(ghost.faces) == 6
    assert all(fields == [names[0]] for fields in ghost.fields)  # parameter 0: field left out
    assert ghost.nDof == 2 * len(ghost.nodes)
    assert any("6 ghost faces" in m for m in messages)
    assert not ghost.updateConnectivity(model)  # same active cells

    rows, cols = ghost.localEntriesOfBlock()
    Kdense = ghost.stiffness.toarray()
    assert np.allclose(Kdense, Kdense.T)

    rng = np.random.default_rng(1)
    dU = rng.normal(size=ghost.nDof)
    P = np.zeros(ghost.nDof)
    K = np.zeros(len(rows))
    ghost.computeKernels(np.zeros(ghost.nDof), dU, P, K, None)
    Kfull = np.zeros((ghost.nDof, ghost.nDof))
    np.add.at(Kfull, (rows, cols), K)
    assert np.allclose(Kfull, Kdense)
    assert np.allclose(P, Kfull @ dU)
    # the penalty of the x component is 2 J
    assert np.allclose(Kdense[::2, ::2], 2.0 * ghost.J.toarray())

    # no boundary cell: the constraint takes no part
    manager.cells = list(model.cells.values())
    assert ghost.updateConnectivity(model)
    assert not ghost.active and ghost.faces == []


def _bSplineGrid(order, nX=10, nY=8, h=0.5):
    from edelweissmeshfree.generators import rectangularbsplinegridgenerator

    model = MPMModel(2)
    rectangularbsplinegridgenerator.generateModelData(
        model, Journal(), x0=0.0, l=nX * h, y0=0.0, h=nY * h, nX=nX, nY=nY, cellProvider="BSplineMarmotCell",
        cellType="Displacement/BSpline/{:}".format(order), order=order,
    )  # fmt: skip
    model.prepareYourself(Journal())
    return model


@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("h", [1.0, 0.5])
def test_bspline_penalty_of_one_face_is_the_hand_computation(order, h):
    """A B-spline of degree p is C^(p-1): over a face only the p-th normal derivative jumps, eq. 16 with q = p. For
    u(x, y) = B_j(x) (a uniform B-spline in x, the partition of unity in y) the jump of the p-th derivative at the
    i-th knot of its support is (-1)^i C(p + 1, i) / h^p, so u^T J u = h^(2p+1) / ((2p+1) (p!)^2) h jump^2 for a
    face of length h; the lower derivatives do not jump."""

    from math import comb, factorial

    nX, nY = 10, 8
    model = _bSplineGrid(order, nX, nY, h)
    adjacency = GridFaceAdjacency(model.cells.values())
    face = (_cell(4, 3, nY), _cell(5, 3, nY))  # at x = 5 h
    nodesY = nY + order
    for jx in range(5 - order, 6):  # B_jx(x) has support [(jx - p) h, (jx + 1) h] in the uniform part
        J, nodes = ghostPenaltyMatrix(adjacency, [face])
        u = np.array([1.0 if (n.label - 1) // nodesY == jx else 0.0 for n in nodes])
        i = 5 - (jx - order)
        jump = comb(order + 1, i) / h**order
        expected = h ** (2 * order + 1) / ((2 * order + 1) * factorial(order) ** 2) * h * jump**2
        assert u @ J @ u == pytest.approx(expected, rel=1e-9)
        Jd, _ = ghostPenaltyMatrix(adjacency, [face], faceScaling="dirichlet")
        assert u @ Jd @ u == pytest.approx(expected / h**2, rel=1e-9)


@pytest.mark.parametrize("order", [2, 3])
def test_bspline_penalty_is_consistent_symmetric_and_positive_semidefinite(order):
    """No penalty on a global polynomial of degree p (its coefficients in the B-spline basis); symmetric, positive
    semi-definite, and positive on a spline that is not one polynomial."""

    model = _bSplineGrid(order, 6, 5, 0.5)
    adjacency = GridFaceAdjacency(model.cells.values())
    active = [c for c in model.cells.values() if (lambda x: x[0] + 0.6 * x[1] < 1.6)(adjacency.centres[c.cellNumber])]
    _, faces = boundaryCellsAndGhostFaces(adjacency, active)
    assert faces
    J, nodes = ghostPenaltyMatrix(adjacency, faces)
    J = J.toarray()
    assert np.abs(J - J.T).max() <= 1e-14 * np.abs(J).max()
    assert np.linalg.eigvalsh(J).min() >= -1e-12 * np.abs(J).max()

    # coefficients of x^p + y^p - x y by least squares on all cells
    labels = sorted(model.nodes)
    index = {label: i for i, label in enumerate(labels)}
    rng = np.random.default_rng(2)
    rows, rhs = [], []
    for cell in model.cells.values():
        lo, hi = (np.asarray(v) for v in cell.getBoundingBox())
        for _ in range(3 * cell.nNodes):
            y = lo + rng.random(2) * (hi - lo)
            row = np.zeros(len(labels))
            for node, N in zip(cell.nodes, cell.getInterpolationVector(np.ascontiguousarray(y))):
                row[index[node.label]] += N
            rows.append(row)
            rhs.append(y[0] ** order + y[1] ** order - y[0] * y[1])
    c = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    u = np.array([c[index[n.label]] for n in nodes])
    assert np.linalg.norm(J @ u) <= 1e-9 * np.abs(J).max() * np.linalg.norm(u)
    v = np.random.default_rng(3).standard_normal(len(nodes))
    assert v @ J @ v > 1e-6 * np.abs(J).max()

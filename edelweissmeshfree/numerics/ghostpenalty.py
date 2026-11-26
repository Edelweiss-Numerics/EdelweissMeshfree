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
"""The ghost penalty of the background grid of the material point method, after Coombs, *Ghost stabilisation of the
material point method for stable quasi-static and dynamic analysis of large deformation problems*, IJNME 124 (2023)
4841-4875, Sections 3 and 4.

**Faces** (Section 4.2 of the paper). The cells that hold material points are *active*. An active cell that shares a
face with an inactive cell of the grid is a *boundary* cell (a face on the outer boundary of the grid, which has no
neighbour, does not count; the same classification as the aggregation,
:class:`~edelweissmeshfree.numerics.aggregation.CellAggregation`). The *ghost faces* are the faces of the boundary
cells whose other cell is active too: faces between two boundary cells, and faces between a boundary cell and an
interior cell (the thick red lines of Fig. 5 of the paper).

**Penalty** (eqs. 16, 18 and 22). For a basis of order :math:`q = 1` (bilinear quadrilaterals, trilinear hexahedra),
the jump of the normal derivative of the basis over a ghost face :math:`F` between the cells :math:`K^+` and
:math:`K^-` with the normal :math:`\\boldsymbol{n}`,

.. math::
    [\\![\\partial_n N_a]\\!] = \\nabla N_a|_{K^+} \\cdot \\boldsymbol{n} - \\nabla N_a|_{K^-} \\cdot \\boldsymbol{n} ,

is penalised by the scalar matrix

.. math::
    J_{ab} = \\sum_F \\frac{h_F^{2k+1}}{(2k+1)(k!)^2} \\int_F [\\![\\partial_n N_a]\\!]\\, [\\![\\partial_n N_b]\\!]\\,
    \\mathrm{d}\\Gamma , \\qquad k = 1 ,

with the size :math:`h_F` of the face (its length in 2D, the square root of its area in 3D), by Gauss quadrature on
the face (exact for the multilinear cells). The jump vanishes for every field whose gradient is continuous over the
face, in particular for every globally linear field (the consistency of the penalty). The penalty of a field with
:math:`n_c` components is :math:`\\gamma\\, J \\otimes I_{n_c}`: :math:`[\\![\\partial_n u_i]\\!] = [\\![\\partial_n
N_a]\\!] u_{ai}` for every component (eq. 19), so the penalty is the same scalar matrix for every field, only the
parameter :math:`\\gamma` is the field's. This keeps it physics-agnostic: it acts on the basis.

The factor :math:`h^{2k+1}` is the one Coombs adopts after Hansbo et al. for traction-free (Neumann) boundaries
(:math:`K_{GN} = \\gamma_k J_G`, ``faceScaling="neumann"``); the Dirichlet form of Sticko et al. adds the factor
:math:`h^{-2}` (:math:`K_{GD} = \\gamma_k J_G / h^2`, ``faceScaling="dirichlet"``). Note that with the Neumann form
the penalty of a stiffness parameter :math:`\\gamma \\sim E` scales with :math:`h^2` relative to the stiffness of a
cell (2D), i.e. it weakens with refinement.

Only the multilinear Lagrangian cells (Quad4, Hexa8) are supported: quadratic cells need the jumps of the second normal
derivatives (:math:`k = 2` in eq. 16) and B-spline cells share their control points beyond their faces;
both raise :class:`NotImplementedError`.

The gradients of the basis are those of the cells,
:meth:`~edelweissmeshfree.cells.marmotcell.marmotcell.MarmotCellWrapper.getInterpolationGradient` (Marmot
``Cell::getInterpolationGradient``).
"""

import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from edelweissmeshfree.numerics.aggregation import GridFaceAdjacency

#: The numbers of nodes of the cells the ghost penalty supports, by dimension (multilinear Lagrangian cells).
_MULTILINEAR_NUMBERS_OF_NODES = {2: 4, 3: 8}

#: The face scalings of the penalty: the exponent of the face size relative to that of the Neumann form h^(2k+1).
FACE_SCALINGS = {"neumann": 0, "dirichlet": -2}


def _checkCellIsMultilinear(cell, dimension: int):
    """Raise NotImplementedError for a cell whose ghost penalty is not implemented (only Quad4 and Hexa8)."""

    if cell.nNodes != _MULTILINEAR_NUMBERS_OF_NODES.get(dimension):
        raise NotImplementedError(
            "Ghost penalty: only multilinear Lagrangian cells (Quad4, Hexa8) are supported, not cell {:} ({:}, {:} "
            "nodes in {:}D); a quadratic basis needs the jumps of its second normal derivatives (k = 2).".format(
                cell.cellNumber, getattr(cell, "cellType", type(cell).__name__), cell.nNodes, dimension
            )
        )


def boundaryCellsAndGhostFaces(adjacency: GridFaceAdjacency, activeCells) -> tuple[list, list]:
    """The boundary cells of the active domain and the ghost faces (Section 4.2 of the paper).

    Parameters
    ----------
    adjacency
        The face neighbours of the cells of the grid.
    activeCells
        The active cells (cells, or cell numbers).

    Returns
    -------
    tuple[list, list]
        The numbers of the boundary cells (sorted), and the ghost faces as sorted pairs of cell numbers
        ``(c1, c2)``, ``c1 < c2``, sorted.
    """

    active = {c if isinstance(c, (int, np.integer)) else c.cellNumber for c in activeCells}
    unknown = active.difference(adjacency.cells)
    if unknown:
        raise ValueError("Ghost penalty: active cell(s) {:} are not cells of the grid.".format(sorted(unknown)))

    neighbours = adjacency.faceNeighbours
    boundary = sorted(c for c in active if any(n not in active for n in neighbours[c]))
    faces = sorted({(min(c, n), max(c, n)) for c in boundary for n in neighbours[c] if n in active})
    return boundary, faces


def _faceQuadrature(cellA, cellB, dimension: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The Gauss points and weights (times the surface Jacobian) of the face two cells share, its unit normal and its
    size.

    Parameters
    ----------
    cellA, cellB
        The two cells.
    dimension
        The spatial dimension.

    Returns
    -------
    tuple
        Points ``(nGp, dim)``, weights ``(nGp,)`` (including the surface Jacobian), the normal (outward of
        ``cellA``), the size h of the face.
    """

    labelsB = {node.label for node in cellB.nodes}
    shared = np.array([node.coordinates for node in cellA.nodes if node.label in labelsB], dtype=float)
    nShared = {2: 2, 3: 4}[dimension]
    if len(shared) != nShared:
        raise ValueError(
            "Ghost penalty: cells {:} and {:} share {:} nodes, not a face of {:} nodes.".format(
                cellA.cellNumber, cellB.cellNumber, len(shared), nShared
            )
        )

    g = 1.0 / math.sqrt(3.0)
    centreA = np.mean([node.coordinates for node in cellA.nodes], axis=0)

    if dimension == 2:
        xa, xb = shared
        t = xb - xa
        h = float(np.linalg.norm(t))
        normal = np.array([t[1], -t[0]]) / h
        xi = np.array([-g, g])
        points = 0.5 * (xa + xb) + 0.5 * xi[:, None] * t
        weights = np.full(2, 0.5 * h)
    else:
        centre = shared.mean(axis=0)
        # order the corners of the (planar) face by their angle around its centre
        normal = np.cross(shared[1] - shared[0], shared[2] - shared[0])
        if np.linalg.norm(normal) < 1e-14 * np.linalg.norm(shared[1] - shared[0]) ** 2:
            normal = np.cross(shared[1] - shared[0], shared[3] - shared[0])
        normal /= np.linalg.norm(normal)
        e1 = shared[0] - centre
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)
        angles = np.arctan2((shared - centre) @ e2, (shared - centre) @ e1)
        corners = shared[np.argsort(angles)]
        points, weights = [], []
        area = 0.0
        for xi1 in (-g, g):
            for xi2 in (-g, g):
                Nf = 0.25 * np.array(
                    [(1 - xi1) * (1 - xi2), (1 + xi1) * (1 - xi2), (1 + xi1) * (1 + xi2), (1 - xi1) * (1 + xi2)]
                )
                dN1 = 0.25 * np.array([-(1 - xi2), (1 - xi2), (1 + xi2), -(1 + xi2)])
                dN2 = 0.25 * np.array([-(1 - xi1), -(1 + xi1), (1 + xi1), (1 - xi1)])
                dA = float(np.linalg.norm(np.cross(dN1 @ corners, dN2 @ corners)))
                points.append(Nf @ corners)
                weights.append(dA)
                area += dA
        points, weights = np.array(points), np.array(weights)
        h = math.sqrt(area)

    if np.dot(normal, points.mean(axis=0) - centreA) < 0:
        normal = -normal

    return points, weights, normal, h


def ghostPenaltyMatrix(
    adjacency: GridFaceAdjacency, faces: list, faceScaling: str = "neumann"
) -> tuple[csr_matrix, list]:
    """The scalar ghost penalty matrix :math:`J` of a set of faces (eq. 22 with eq. 16, :math:`k = 1`).

    Parameters
    ----------
    adjacency
        The face neighbours of the cells of the grid.
    faces
        The faces, pairs of cell numbers (see :func:`boundaryCellsAndGhostFaces`).
    faceScaling
        ``"neumann"`` (default, Coombs/Hansbo: :math:`h^3/3`) or ``"dirichlet"`` (Sticko: :math:`h/3`).

    Returns
    -------
    tuple[csr_matrix, list]
        The matrix (symmetric, positive semi-definite), on the nodes of the cells of the faces, and these nodes,
        sorted by label.
    """

    if faceScaling not in FACE_SCALINGS:
        raise ValueError(
            "Ghost penalty: faceScaling must be one of {:}, not {:}.".format(list(FACE_SCALINGS), faceScaling)
        )
    k = 1
    exponent = 2 * k + 1 + FACE_SCALINGS[faceScaling]
    factor = 1.0 / ((2 * k + 1) * math.factorial(k) ** 2)

    dimension = adjacency.dimension
    cells = adjacency.cells
    nodes = {}
    for pair in faces:
        for c in pair:
            _checkCellIsMultilinear(cells[c], dimension)
            for node in cells[c].nodes:
                nodes[node.label] = node
    labels = sorted(nodes)
    index = {label: i for i, label in enumerate(labels)}

    rows, cols, vals = [], [], []
    for cA, cB in faces:
        cellA, cellB = cells[cA], cells[cB]
        points, weights, normal, h = _faceQuadrature(cellA, cellB, dimension)
        idxA = [index[node.label] for node in cellA.nodes]
        idxB = [index[node.label] for node in cellB.nodes]
        local = sorted(set(idxA) | set(idxB))
        position = {i: p for p, i in enumerate(local)}
        pA = [position[i] for i in idxA]
        pB = [position[i] for i in idxB]

        Jf = np.zeros((len(local), len(local)))
        for x, w in zip(points, weights):
            x = np.ascontiguousarray(x, dtype=float)
            jump = np.zeros(len(local))
            np.add.at(jump, pA, normal @ np.asarray(cellA.getInterpolationGradient(x)))
            np.add.at(jump, pB, -(normal @ np.asarray(cellB.getInterpolationGradient(x))))
            Jf += w * np.outer(jump, jump)
        Jf *= factor * h**exponent

        rows.append(np.repeat(local, len(local)))
        cols.append(np.tile(local, len(local)))
        vals.append(Jf.ravel())

    n = len(labels)
    if not faces:
        return csr_matrix((n, n)), [nodes[label] for label in labels]

    J = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    J.sum_duplicates()
    J.sort_indices()
    return J, [nodes[label] for label in labels]


def ghostPenaltyInNumbering(
    model,
    activeCells,
    numbering,
    penaltyParameters: dict,
    faceScaling: str = "neumann",
    adjacency: GridFaceAdjacency = None,
) -> csr_matrix:
    """The ghost penalty stiffness :math:`\\sum_f \\gamma_f J \\otimes I` of the active cells of a model in a numbering of
    the degrees of freedom (a diagnostic, e.g. for the condition number of the stabilised stiffness).

    Parameters
    ----------
    model
        The model, :class:`~edelweissmeshfree.models.mpmmodel.MPMModel`.
    activeCells
        The active cells.
    numbering
        The numbering, :class:`~edelweissfe.numerics.dofnumbering.DofNumbering`.
    penaltyParameters
        The penalty parameter of every field of the cells, ``{field: gamma}``.
    faceScaling
        See :func:`ghostPenaltyMatrix`.
    adjacency
        The face neighbours of the cells of the grid; default those of all cells of the model.

    Returns
    -------
    csr_matrix
        The matrix, ``nDof x nDof``.
    """

    if adjacency is None:
        adjacency = GridFaceAdjacency(model.cells.values())
    _, faces = boundaryCellsAndGhostFaces(adjacency, activeCells)
    J, nodes = ghostPenaltyMatrix(adjacency, faces, faceScaling)
    fieldsOfNode = fieldsOfGhostNodes(adjacency, faces, penaltyParameters)

    nDof = numbering.nDof
    indicesOf = numbering.idcsOfFieldVariablesInDofVector
    Jc = J.tocoo()
    rows, cols, vals = [], [], []
    for field, gamma in penaltyParameters.items():
        if gamma == 0.0:
            continue
        dofsOfNode = []
        for node in nodes:
            dofs = indicesOf.get(node.fields.get(field)) if field in fieldsOfNode[node.label] else None
            dofsOfNode.append(None if dofs is None else np.asarray(dofs, dtype=int).ravel())
        for a, b, v in zip(Jc.row, Jc.col, Jc.data):
            da, db = dofsOfNode[a], dofsOfNode[b]
            if da is None or db is None:
                continue
            rows.append(da)
            cols.append(db)
            vals.append(np.full(len(da), gamma * v))
    if not rows:
        return csr_matrix((nDof, nDof))
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nDof, nDof)).tocsr()


def fieldsOfGhostNodes(adjacency: GridFaceAdjacency, faces: list, penaltyParameters: dict) -> dict:
    """The fields of the nodes of the cells of the ghost faces, by node label, as their cells list them, and check that
    every field has a penalty parameter.

    Parameters
    ----------
    adjacency
        The face neighbours of the cells of the grid.
    faces
        The ghost faces.
    penaltyParameters
        The penalty parameters, ``{field: gamma}``.

    Returns
    -------
    dict
        The fields of every node, by label (a list in the order of the cells).

    Raises
    ------
    ValueError
        If a field of the cells has no penalty parameter (pass 0 to leave a field unpenalised).
    """

    fieldsOfNode = {}
    for pair in faces:
        for c in pair:
            cell = adjacency.cells[c]
            for node, fields in zip(cell.nodes, cell.fields):
                fieldsOfNode.setdefault(node.label, list(fields))
    missing = sorted({f for fields in fieldsOfNode.values() for f in fields}.difference(penaltyParameters))
    if missing:
        raise ValueError(
            "Ghost penalty: no penalty parameter for field(s) {:} of the cells; give one for every field "
            "(0 leaves a field unpenalised).".format(missing)
        )
    return fieldsOfNode

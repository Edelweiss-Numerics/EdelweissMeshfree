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
"""Mesh aggregation of the background grid of the material point method, after Coombs, Bird and Pretti, *The aggregated
material point method (AgMPM)*, CMAME 442 (2025) 118012, Section 3.

The cells that hold material points are *active*. An active cell that shares a face with an inactive cell of the grid is
a *boundary* cell; a face on the outer boundary of the grid, which has no neighbour, does not count. The other active
cells are *interior* cells. The *exterior* (aggregated) nodes are the nodes of the active cells that no interior cell
holds. They are the nodes that may suffer from the small cut problem: the material points reach them only through the
small parts of boundary cells they cover.

Every boundary cell is connected to an interior cell: a face neighbour that is interior, else (the "corner" case) the
interior cell whose centre is closest to its own. An exterior node takes the values of an interior cell, extended to it
(eq. 9 of the paper),

.. math::
    u_b = \\sum_a N_a(\\boldsymbol{x}_b)\\, u_a ,

with the basis :math:`N_a` of the interior cell extrapolated to the position :math:`\\boldsymbol{x}_b` of the node (for
a linear cell, a node one cell beyond has e.g. :math:`N = -1, 2`, see Fig. 5 of the paper). The interior cell is the one
connected to a boundary cell that holds the node; of these, the one with the centre closest to the node is taken, which
keeps the extrapolation as short as possible (the paper leaves the choice open).

Together with the identity on the other degrees of freedom this is the extension matrix :math:`E` of eq. 11. It depends
on the active cells and on the grid only, not on the motion within an increment, so it stays the same for all
iterations of an increment, and it is rebuilt whenever the active cells change.

The aggregation is applied to the implicit solution through EdelweissFE's multi-point constraints, see
:class:`~edelweissmeshfree.constraints.aggregatedmpm.AggregatedMaterialPointMethod`: an exterior degree of freedom is a
slave of the degrees of freedom of the nodes of its interior cell. The condensed system is :math:`E^T K E`, the
condensed residual :math:`E^T R` (eq. 13), and the solution of the exterior nodes is recovered with :math:`E`
(eq. 14).

:func:`reducedStiffness` gives the stiffness matrix reduced to the free degrees of freedom, with and without
aggregation, and its condition number, as studied in Section 4 of the paper (Fig. 7).
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.spatial import cKDTree

#: The numbers of nodes of the Lagrangian cells whose faces this module can recognize, by dimension: two cells share a
#: face if they share at least :data:`_NODES_ON_A_FACE` nodes, and at most :data:`_NODES_ON_A_FACE` - 1 nodes if they
#: share an edge or a corner only.
_LAGRANGIAN_NUMBERS_OF_NODES = {2: (4, 8, 9), 3: (8, 20, 27)}
_NODES_ON_A_FACE = {2: 2, 3: 4}

#: Relative tolerance of the checks that the extrapolated basis of an interior cell is a partition of unity and
#: reproduces the position of the exterior node.
_EXTRAPOLATION_TOLERANCE = 1e-8

#: Weights below this fraction of the largest weight of an exterior node are round-off and left out.
_WEIGHT_AT_ROUND_OFF = 1e-12


def _checkCellIsLagrangian(cell, dimension: int):
    """Raise NotImplementedError for a cell this module cannot aggregate.

    Parameters
    ----------
    cell
        The cell.
    dimension
        The spatial dimension.
    """

    if "bspline" in type(cell).__name__.lower() or "bspline" in str(getattr(cell, "cellType", "")).lower():
        raise NotImplementedError(
            "Aggregation (AgMPM) supports Lagrangian cells only; B-spline cell {:} ({:}) shares its control points "
            "beyond its faces, so the boundary and the exterior nodes are not defined by face connectivity.".format(
                cell.cellNumber, getattr(cell, "cellType", type(cell).__name__)
            )
        )
    if dimension not in _LAGRANGIAN_NUMBERS_OF_NODES or cell.nNodes not in _LAGRANGIAN_NUMBERS_OF_NODES[dimension]:
        raise NotImplementedError(
            "Aggregation (AgMPM) supports Lagrangian quadrilaterals and hexahedra only, not cell {:} with {:} nodes "
            "in {:}D.".format(cell.cellNumber, cell.nNodes, dimension)
        )
    fieldsOfNodes = getattr(cell, "fields", None)
    if fieldsOfNodes is not None and any(sorted(f) != sorted(fieldsOfNodes[0]) for f in fieldsOfNodes):
        raise NotImplementedError(
            "Aggregation (AgMPM) needs every node of a cell to carry the same fields (equal-order interpolation); "
            "cell {:} ({:}) has fields {:} on its nodes.".format(
                cell.cellNumber, getattr(cell, "cellType", type(cell).__name__), list(fieldsOfNodes)
            )
        )


def _centreOf(cell) -> np.ndarray:
    """The centre of a cell: the mean of its node coordinates (the centre of a parallelogram or parallelepiped)."""

    return np.mean([node.coordinates for node in cell.nodes], axis=0)


class GridFaceAdjacency:
    """The face neighbours of the cells of a background grid, found by the nodes they share.

    Parameters
    ----------
    cells
        The cells of the grid.
    """

    def __init__(self, cells):
        cells = sorted(cells, key=lambda cell: cell.cellNumber)
        if not cells:
            raise ValueError("Aggregation (AgMPM): the background grid has no cells.")

        self.dimension = len(cells[0].nodes[0].coordinates)
        for cell in cells:
            _checkCellIsLagrangian(cell, self.dimension)

        #: The cells, by number.
        self.cells = {cell.cellNumber: cell for cell in cells}
        #: The centres of the cells, by number.
        self.centres = {cell.cellNumber: _centreOf(cell) for cell in cells}

        cellsOfNode = {}
        for cell in cells:
            for node in cell.nodes:
                cellsOfNode.setdefault(node.label, []).append(cell.cellNumber)

        nodesOnAFace = _NODES_ON_A_FACE[self.dimension]
        #: The numbers of the face neighbours of every cell, by number.
        self.faceNeighbours = {}
        for cell in cells:
            sharedNodes = {}
            for node in cell.nodes:
                for other in cellsOfNode[node.label]:
                    if other != cell.cellNumber:
                        sharedNodes[other] = sharedNodes.get(other, 0) + 1
            self.faceNeighbours[cell.cellNumber] = sorted(
                other for other, nShared in sharedNodes.items() if nShared >= nodesOnAFace
            )


class CellAggregation:
    """The aggregation of the active cells of a background grid: the classification of the cells and nodes, and the
    extension of the interior cells to the exterior nodes (Section 3.1 of the paper).

    Parameters
    ----------
    adjacency
        The face neighbours of the cells of the grid.
    activeCells
        The active cells, those that hold material points; cells of the grid.
    excludedNodes
        Nodes that are never aggregated, even if exterior: e.g. the nodes that finite elements share with the grid,
        which carry their own stiffness.
    """

    def __init__(self, adjacency: GridFaceAdjacency, activeCells, excludedNodes=()):
        self.adjacency = adjacency

        active = {cell.cellNumber for cell in activeCells}
        unknown = active.difference(adjacency.cells)
        if unknown:
            raise ValueError(
                "Aggregation (AgMPM): active cell(s) {:} are not cells of the background grid.".format(sorted(unknown))
            )

        neighbours = adjacency.faceNeighbours
        #: The numbers of the active cells, sorted.
        self.activeCells = sorted(active)
        #: The numbers of the boundary cells: active cells with an inactive face neighbour, sorted.
        self.boundaryCells = [c for c in self.activeCells if any(n not in active for n in neighbours[c])]
        boundary = set(self.boundaryCells)
        #: The numbers of the interior cells: the active cells that are not boundary cells, sorted.
        self.interiorCells = [c for c in self.activeCells if c not in boundary]

        cells = adjacency.cells
        nodesOfInteriorCells = {node.label for c in self.interiorCells for node in cells[c].nodes}
        excluded = {node.label for node in excludedNodes}

        nodesOfActiveCells = {}
        boundaryCellsOfNode = {}
        for c in self.activeCells:
            for node in cells[c].nodes:
                nodesOfActiveCells[node.label] = node
                if c in boundary:
                    boundaryCellsOfNode.setdefault(node.label, []).append(c)

        #: The exterior nodes: the nodes of active cells that no interior cell holds, sorted by label.
        self.exteriorNodes = [
            nodesOfActiveCells[label]
            for label in sorted(nodesOfActiveCells)
            if label not in nodesOfInteriorCells and label not in excluded
        ]

        #: The interior cell connected to every boundary cell, by number; empty if there is no interior cell.
        self.connectedInteriorCell = self._connectInteriorCells() if self.interiorCells else {}

        #: For every exterior node (by label), the interior cell it is extended from, and the weights of the nodes of
        #: that cell, ``(cellNumber, [(node, weight), ...])``. Empty if there is no interior cell.
        self.extension = {}
        if self.interiorCells:
            for node in self.exteriorNodes:
                self.extension[node.label] = self._extensionTo(node, boundaryCellsOfNode[node.label])

    def _connectInteriorCells(self) -> dict:
        """Connect every boundary cell to an interior cell: the interior face neighbour with the closest centre, else
        (the corner case) the interior cell with the closest centre; ties go to the lower cell number.

        Returns
        -------
        dict
            The number of the connected interior cell, by the number of the boundary cell.
        """

        centres = self.adjacency.centres
        interior = set(self.interiorCells)
        connected = {}
        cornerCases = []

        for c in self.boundaryCells:
            candidates = [n for n in self.adjacency.faceNeighbours[c] if n in interior]
            if candidates:
                connected[c] = self._closest(centres[c], candidates)
            else:
                cornerCases.append(c)

        if cornerCases:
            interiorCentres = np.array([centres[c] for c in self.interiorCells])
            tree = cKDTree(interiorCentres)
            k = min(len(self.interiorCells), 2 ** (self.adjacency.dimension + 1) + 1)
            for c in cornerCases:
                _, idcs = tree.query(centres[c], k=k)
                candidates = [self.interiorCells[i] for i in np.atleast_1d(idcs)]
                connected[c] = self._closest(centres[c], candidates)

        return connected

    def _closest(self, x: np.ndarray, candidates: list) -> int:
        """The candidate cell with the centre closest to a point; of equally close ones (to round-off), the one with
        the lowest number."""

        centres = self.adjacency.centres
        distances = np.array([np.linalg.norm(centres[c] - x) for c in candidates])
        closest = distances <= distances.min() * (1.0 + 1e-9) + 1e-14 * (1.0 + np.linalg.norm(x))
        return min(c for c, isClosest in zip(candidates, closest) if isClosest)

    def _extensionTo(self, node, boundaryCellsOfNode: list) -> tuple:
        """The extension of an interior cell to an exterior node (eqs. 9 and 12 of the paper).

        Parameters
        ----------
        node
            The exterior node.
        boundaryCellsOfNode
            The numbers of the boundary cells that hold the node.

        Returns
        -------
        tuple
            The number of the interior cell, and the weights of its nodes, ``[(node, weight), ...]``.
        """

        x = np.ascontiguousarray(node.coordinates, dtype=float)
        candidates = sorted({self.connectedInteriorCell[c] for c in boundaryCellsOfNode})
        failures = []

        # the closest candidate first; the next one if the inverse map of a (distorted) cell fails for the node
        while candidates:
            interiorCell = self._closest(x, candidates)
            candidates.remove(interiorCell)
            cell = self.adjacency.cells[interiorCell]

            try:
                N = np.asarray(cell.getInterpolationVector(x), dtype=float)
            except RuntimeError as error:
                failures.append("cell {:}: {:}".format(interiorCell, error))
                continue

            nodeCoordinates = np.array([n.coordinates for n in cell.nodes])
            scale = np.max(np.abs(N))
            size = np.linalg.norm(nodeCoordinates.max(axis=0) - nodeCoordinates.min(axis=0))
            if (
                not np.all(np.isfinite(N))
                or abs(N.sum() - 1.0) > _EXTRAPOLATION_TOLERANCE * scale
                or np.linalg.norm(N @ nodeCoordinates - x) > _EXTRAPOLATION_TOLERANCE * scale * size
            ):
                failures.append(
                    "cell {:}: not an extrapolation of its basis (sum {:}, reproduces {:})".format(
                        interiorCell, N.sum(), N @ nodeCoordinates
                    )
                )
                continue

            weights = [(n, w) for n, w in zip(cell.nodes, N) if abs(w) > _WEIGHT_AT_ROUND_OFF * scale]
            return interiorCell, weights

        raise RuntimeError(
            "Aggregation (AgMPM): no interior cell extends its basis to exterior node {:} at {:}; the cell's "
            "getInterpolationVector must extrapolate to points outside of it. {:}".format(
                node.label, x, "; ".join(failures)
            )
        )

    def recordsIn(self, numbering) -> list:
        """The linear dependencies of the exterior degrees of freedom on the interior ones, in a numbering of the
        degrees of freedom: the same scalar weights (the extrapolated basis of the interior cell) for every field of an
        exterior node that the numbering holds, whatever its physics (displacement, rotation, nonlocal field, pressure,
        ...), component by component.

        Raises NotImplementedError if a field of an exterior node is missing on a node of its interior cell (fields of
        mixed interpolation order, which a single scalar basis cannot extend).

        Parameters
        ----------
        numbering
            The numbering of the degrees of freedom, :class:`~edelweissfe.numerics.dofnumbering.DofNumbering`.

        Returns
        -------
        list
            ``(slaveDofIndex, [(masterDofIndex, weight), ...])``, one per exterior degree of freedom.
        """

        indicesOf = numbering.idcsOfFieldVariablesInDofVector
        records = []

        for node in self.exteriorNodes:
            if node.label not in self.extension:
                continue
            _, weights = self.extension[node.label]
            for field, fieldVariable in node.fields.items():
                slaveDofs = indicesOf.get(fieldVariable)
                if slaveDofs is None:
                    continue
                masterDofs = [indicesOf.get(master.fields.get(field)) for master, _ in weights]
                if any(dofs is None or len(dofs) != len(slaveDofs) for dofs in masterDofs):
                    raise NotImplementedError(
                        "Aggregation (AgMPM): field '{:}' of exterior node {:} is not carried, with the same number "
                        "of components, by every node of its interior cell {:}; fields of mixed interpolation order "
                        "are not supported.".format(field, node.label, self.extension[node.label][0])
                    )
                for component, slaveDof in enumerate(slaveDofs):
                    records.append(
                        (
                            int(slaveDof),
                            [(int(dofs[component]), float(w)) for dofs, (_, w) in zip(masterDofs, weights)],
                        )
                    )

        return records


def extensionMatrix(records: list, nDof: int, keptSlaveDofs=()) -> tuple[csr_matrix, np.ndarray]:
    """The extension matrix :math:`E` of eq. 11 of the paper, from the records of
    :meth:`CellAggregation.recordsIn`: the identity on the independent degrees of freedom, and the weights of the
    masters on the rows of the slaves; the columns of the independent degrees of freedom only.

    Parameters
    ----------
    records
        ``(slaveDofIndex, [(masterDofIndex, weight), ...])``.
    nDof
        The number of degrees of freedom.
    keptSlaveDofs
        Slave degrees of freedom whose records are dropped, so that they stay independent (e.g. prescribed ones).

    Returns
    -------
    tuple[csr_matrix, np.ndarray]
        :math:`E`, of shape ``(nDof, nIndependent)``, and the indices of the independent degrees of freedom, i.e. of
        the columns of :math:`E`.
    """

    kept = set(int(i) for i in keptSlaveDofs)
    records = [record for record in records if record[0] not in kept]
    slaves = np.array(sorted(record[0] for record in records), dtype=int)
    independent = np.setdiff1d(np.arange(nDof), slaves)
    column = np.full(nDof, -1, dtype=int)
    column[independent] = np.arange(len(independent))

    rows = list(independent)
    cols = list(range(len(independent)))
    vals = [1.0] * len(independent)
    for slave, masters in records:
        for master, weight in masters:
            if column[master] < 0:
                raise ValueError("Aggregation (AgMPM): master DOF {:} is itself a slave.".format(master))
            rows.append(slave)
            cols.append(column[master])
            vals.append(weight)

    return csr_matrix((vals, (rows, cols)), shape=(nDof, len(independent))), independent


@dataclass
class ReducedStiffness:
    """The stiffness matrix reduced to the free degrees of freedom (see :func:`reducedStiffness`).

    Attributes
    ----------
    matrix
        The reduced stiffness matrix, ``E^T K E`` restricted to the free degrees of freedom if aggregated.
    freeDofs
        The indices of the free degrees of freedom in the numbering of the active domain.
    conditionNumber
        The 2-norm condition number of :attr:`matrix` (``inf`` if singular); None if not computed.
    nExteriorDofs
        The number of aggregated (slave) degrees of freedom, 0 without aggregation.
    aggregation
        The aggregation of the active cells; None without aggregation.
    """

    matrix: csr_matrix
    freeDofs: np.ndarray
    conditionNumber: float | None
    nExteriorDofs: int
    aggregation: CellAggregation | None = None


def _assembleCellStiffness(system, time: float, dT: float) -> csr_matrix:
    """The tangent stiffness of the active cells of a discrete system, at a zero increment from the converged state.

    Parameters
    ----------
    system
        The discrete system, :class:`~edelweissmeshfree.models.mpmdiscretesystem.MPMDiscreteSystem`.
    time
        The time at the end of the (zero) increment.
    dT
        The time increment.

    Returns
    -------
    csr_matrix
        The stiffness matrix, in the numbering of the active domain.
    """

    from edelweissfe.numerics.systemmatrixassembly import VIJSystemMatrixAssembly
    from edelweissfe.timesteppers.timestep import TimeStep

    locations = system.locations
    U = locations.constructDofVector()
    dU = locations.constructDofVector()
    U[:] = 0.0
    dU[:] = 0.0

    system.updateMaterialPoints(dU, TimeStep(0, 0.0, 0.0, dT, time, time), lambda entities, work: work(entities))

    K = VIJSystemMatrixAssembly(locations, 1)
    K.beginAssembly()
    for cell in system.activeCells:
        Ke = K[cell]
        Pe = np.zeros(cell.nDof)
        cell.computeKernels(Ke, Pe, U[cell], dU[cell], time, dT)
    K.flush()

    return K.assembledMatrix().copy()


def reducedStiffness(
    model, dirichlets: list = (), aggregate: bool = False, conditionNumber: bool = True, time=None, dT: float = 1.0
) -> ReducedStiffness:
    """The tangent stiffness of the active cells of an MPM model, reduced to its free degrees of freedom: without the
    degrees of freedom the Dirichlet boundary conditions prescribe, and with aggregation also without the exterior
    (aggregated) ones, ``(E^T K E)`` on the free interior degrees of freedom (eq. 13 of the paper). The inactive
    degrees of freedom are not part of the discrete system of the model in the first place. With its condition number,
    as in Section 4 (Fig. 7) of the paper.

    The connectivity of the model (the material points in the cells) is brought up to date first, as at the start of
    an increment; the stiffness is the tangent at the converged state. Only the active cells
    contribute, no particles, constraints or loads.

    Parameters
    ----------
    model
        The model, :class:`~edelweissmeshfree.models.mpmmodel.MPMModel`, with its MPM managers.
    dirichlets
        The Dirichlet boundary conditions, :class:`~edelweissfe.stepactions.dirichlet.StepAction`; their degrees of
        freedom are removed. A prescribed exterior degree of freedom stays independent (it is not aggregated), as in
        the solver.
    aggregate
        Whether to aggregate the exterior degrees of freedom.
    conditionNumber
        Whether to compute the condition number (dense, :func:`numpy.linalg.cond`; for small systems only).
    time
        The time; default the time of the model.
    dT
        The time increment.

    Returns
    -------
    ReducedStiffness
        The reduced stiffness matrix, its free degrees of freedom and its condition number.
    """

    if time is None:
        time = model.time

    model.updateConnectivity()
    system = model.buildDiscreteSystem()
    numbering = system.numbering
    nDof = numbering.nDof

    K = _assembleCellStiffness(system, time, dT)

    prescribed = set()
    for dirichlet in dirichlets:
        dofs, _ = numbering.locateFieldOnNodeSet(dirichlet.field, dirichlet.nSet)
        prescribed.update(int(i) for i in dofs.reshape((-1, dirichlet.fieldSize))[:, dirichlet.components].flatten())

    nExteriorDofs = 0
    aggregation = None
    if aggregate:
        aggregation = aggregationOfModel(model, system.activeCells)
        records = aggregation.recordsIn(numbering)
        E, independent = extensionMatrix(records, nDof, keptSlaveDofs=prescribed)
        nExteriorDofs = nDof - len(independent)
        K = (E.T @ K @ E).tocsr()
    else:
        E, independent = identity(nDof, format="csr"), np.arange(nDof)

    free = np.array([i for i in range(len(independent)) if int(independent[i]) not in prescribed], dtype=int)
    reduced = K[free][:, free].tocsr()

    cond = None
    if conditionNumber:
        cond = float(np.linalg.cond(reduced.toarray())) if reduced.shape[0] else 0.0

    return ReducedStiffness(reduced, independent[free], cond, nExteriorDofs, aggregation)


def aggregationOfModel(model, activeCells, adjacency: GridFaceAdjacency = None) -> CellAggregation:
    """The aggregation of the active cells of an MPM model, on the grid of all its cells; the nodes of its finite
    elements and cell elements are not aggregated.

    Parameters
    ----------
    model
        The model, :class:`~edelweissmeshfree.models.mpmmodel.MPMModel`.
    activeCells
        The active cells.
    adjacency
        The face neighbours of the cells of the grid; default those of all cells of the model.

    Returns
    -------
    CellAggregation
        The aggregation.
    """

    if adjacency is None:
        adjacency = GridFaceAdjacency(model.cells.values())

    return CellAggregation(adjacency, activeCells, nodesOfFiniteElements(model))


def nodesOfFiniteElements(model) -> list:
    """The nodes of the finite elements and cell elements of a model, which carry their own stiffness and are never
    aggregated."""

    return [node for element in (*model.elements.values(), *model.cellElements.values()) for node in element.nodes]

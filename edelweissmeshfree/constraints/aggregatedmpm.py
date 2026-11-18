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
"""The aggregated material point method (AgMPM, Coombs, Bird and Pretti, CMAME 442 (2025) 118012) for the implicit
solution of an :class:`~edelweissmeshfree.models.mpmmodel.MPMModel`: the degrees of freedom of the exterior nodes of
the active domain are eliminated by extending the basis of interior cells to them, see
:mod:`~edelweissmeshfree.numerics.aggregation`.

The aggregation is a linear multi-point constraint, which EdelweissFE's implicit solvers
(:class:`~edelweissfe.solvers.nonlinearimplicitstatic.NIST`,
:class:`~edelweissfe.solvers.nonlinearimplicitstaticparallel.NISTParallel`) condense out of the system of equations:
with the extension matrix :math:`E`, they solve :math:`E^T K E\\, \\delta U^{int} = E^T R`, recover
:math:`\\delta U = E\\, \\delta U^{int}`, and test the convergence on the condensed residual :math:`E^T R`. It is
opt-in: a model aggregates only if it holds the constraint,

.. code-block:: python

    from edelweissmeshfree.constraints.aggregatedmpm import AggregatedMaterialPointMethod

    mpmModel.multiPointConstraints["agmpm"] = AggregatedMaterialPointMethod("agmpm", mpmModel, journal)

The solvers collect the records of the constraint whenever they build a new discrete system, i.e. whenever the active
cells change; for the same numbering, the records are computed once.

A degree of freedom a Dirichlet boundary condition prescribes is never aggregated: the solver drops its record, and
it stays an independent, prescribed degree of freedom (the constraint sets
:attr:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase.slavesYieldToDirichlet`, so
that this is reported quietly). The masters are the nodes of interior cells, never exterior nodes, so a prescribed
exterior degree of freedom is the master of no other one.
"""

import numpy as np
from edelweissfe.constraints.base.multipointconstraintbase import (
    MultiPointConstraintBase,
)
from edelweissfe.journal.journal import Journal

from edelweissmeshfree.numerics.aggregation import (
    CellAggregation,
    GridFaceAdjacency,
    nodesOfFiniteElements,
)


class AggregatedMaterialPointMethod(MultiPointConstraintBase):
    """Aggregate the exterior degrees of freedom of the active domain of an MPM model with those of its interior cells
    (AgMPM). Parameter free: every exterior degree of freedom is aggregated.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model, :class:`~edelweissmeshfree.models.mpmmodel.MPMModel`; its active cells are those of its MPM
        managers.
    journal
        The journal, for the summary of every new aggregation and the warning if there is no interior cell.
    cells
        The cells of the background grid; default all cells of the model.
    commonProjectionPoint
        Whether the cells of an aggregate evaluate their cell-level projection (F-bar) at one common point, see
        :meth:`updateConnectivity`.
    """

    slavesYieldToDirichlet = True

    def __init__(self, name: str, model, journal: Journal = None, cells=None, commonProjectionPoint: bool = True):
        self._name = name
        self._model = model
        self._journal = journal
        self._cells = list(model.cells.values()) if cells is None else list(cells)
        self._adjacency = None
        self._records = []
        self._numbering = None
        self._activeCellNumbers = None
        self._aggregationCache = None
        #: Whether the cells of an aggregate evaluate their cell-level projection at one common point, see
        #: :meth:`updateConnectivity`.
        self.commonProjectionPoint = commonProjectionPoint
        #: The aggregation of the last numbering, :class:`~edelweissmeshfree.numerics.aggregation.CellAggregation`.
        self.aggregation = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def adjacency(self) -> GridFaceAdjacency:
        """The face neighbours of the cells of the grid, found once."""

        if self._adjacency is None:
            self._adjacency = GridFaceAdjacency(self._cells)
        return self._adjacency

    def _message(self, message: str):
        if self._journal is not None:
            self._journal.message(message, self._name)

    def getMultiPointConstraints(self, numbering) -> list[tuple[int, list[tuple[int, float]]]]:
        """The records of the exterior degrees of freedom of the active cells of the model in a numbering; see
        :meth:`~edelweissfe.constraints.base.multipointconstraintbase.MultiPointConstraintBase.getMultiPointConstraints`.

        Without an interior cell there is nothing to extend: no record, and a warning in the journal.

        Parameters
        ----------
        numbering
            The numbering of the degrees of freedom of the active domain.

        Returns
        -------
        list[tuple[int, list[tuple[int, float]]]]
            ``(slaveDofIndex, [(masterDofIndex, weight), ...])``, one per exterior degree of freedom.
        """

        activeCells = self._model.activeCellsOf(self._model.mpmManagers)
        activeCellNumbers = tuple(cell.cellNumber for cell in activeCells)

        if numbering is self._numbering and activeCellNumbers == self._activeCellNumbers:
            return self._records

        aggregation = CellAggregation(self.adjacency, activeCells, nodesOfFiniteElements(self._model))

        if activeCells and not aggregation.interiorCells:
            self._message(
                "WARNING: no interior cell among {:} active cells, nothing to aggregate with; the exterior "
                "degrees of freedom are not aggregated".format(len(activeCells))
            )
            records = []
        else:
            records = aggregation.recordsIn(numbering)
            self._message(
                "{:} active cells: {:} interior, {:} boundary; {:} exterior nodes, {:} aggregated dof".format(
                    len(aggregation.activeCells),
                    len(aggregation.interiorCells),
                    len(aggregation.boundaryCells),
                    len(aggregation.exteriorNodes),
                    len(records),
                )
            )

        self.aggregation = aggregation
        self._records = records
        self._numbering = numbering
        self._activeCellNumbers = activeCellNumbers

        return records

    def _aggregationOf(self, activeCells: list, activeCellNumbers: tuple) -> CellAggregation:
        """The aggregation of the active cells, computed once per set of active cells."""

        if self._aggregationCache is None or self._aggregationCache[0] != activeCellNumbers:
            aggregation = CellAggregation(self.adjacency, activeCells, nodesOfFiniteElements(self._model))
            self._aggregationCache = (activeCellNumbers, aggregation)
        return self._aggregationCache[1]

    def updateConnectivity(self):
        """Follow the material points the managers have just assigned to the cells (called by
        :meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.updateConnectivity` at the start of every increment):
        every aggregate, i.e. an interior cell and the boundary cells connected to it, gets one common projection
        point, the centroid of the material points of its cells, and every cell of the aggregate evaluates its cell-level
        projection there (:meth:`setProjectionPoint`, e.g. the volume change of an F-bar cell). The boundary cells
        carry the extension of the field of their interior cell, so this imposes one projection constraint per
        aggregate instead of one per cell. Cells without such a projection ignore the point, so this is
        physics-agnostic; with ``commonProjectionPoint=False``, every cell keeps its own choice."""

        if not self.commonProjectionPoint:
            return
        activeCells = self._model.activeCellsOf(self._model.mpmManagers)
        aggregation = self._aggregationOf(activeCells, tuple(cell.cellNumber for cell in activeCells))
        if not aggregation.connectedInteriorCell:
            return
        cellByNumber = {cell.cellNumber: cell for cell in activeCells}
        members = {}
        for boundaryCell, interiorCell in aggregation.connectedInteriorCell.items():
            members.setdefault(interiorCell, [interiorCell]).append(boundaryCell)
        for cellNumbers in members.values():
            cells = [cellByNumber[c] for c in cellNumbers]
            if not all(hasattr(cell, "setProjectionPoint") for cell in cells):
                continue
            x = np.array(
                [
                    np.asarray(mp.getCenterCoordinates(), dtype=float).ravel()
                    for cell in cells
                    for mp in cell.assignedMaterialPoints
                ]
            )
            point = np.ascontiguousarray(x.mean(axis=0)[: self.adjacency.dimension])
            for cell in cells:
                cell.setProjectionPoint(point)

    def claimedSlaveNodes(self) -> set:
        """The exterior nodes of the last aggregation."""

        return set(self.aggregation.exteriorNodes) if self.aggregation is not None else set()

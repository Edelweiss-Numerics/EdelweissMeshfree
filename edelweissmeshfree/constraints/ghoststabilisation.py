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
"""Ghost stabilisation of the implicit material point method (Coombs, IJNME 124 (2023) 4841-4875, Section 3.4): a
penalty on the jumps of the normal derivatives of the basis over the faces of the boundary cells of the active domain,
see :mod:`~edelweissmeshfree.numerics.ghostpenalty`.

It modifies the discrete equilibrium by the force :math:`\\gamma J \\Delta U` (eq. 24 with eq. 25) and the stiffness by
:math:`\\gamma J` (eq. 26), field by field with the field's parameter :math:`\\gamma_f`,

.. math::
    P_G = \\sum_f \\gamma_f\\, (J \\otimes I)\\, \\Delta U_f , \\qquad K_G = \\sum_f \\gamma_f\\, J \\otimes I .

:math:`\\Delta U` is the increment of the nodal values since the last converged state: the displacement of the
background grid in the current step (Section 4.3 of the paper), so a converged state carries no ghost force into the
next step, and for a field with total nodal values (e.g. a nonlocal field) only its change in the step is penalised.
:math:`J` depends on the grid and on the active cells only, not on the motion within an increment (it is constant over
a step, as in the paper), so the residual is linear in :math:`\\Delta U` and the tangent :math:`K_G` is exact. It is
rebuilt whenever the active cells change (:meth:`GhostStabilisation.updateConnectivity`).

It is opt-in (a model is stabilised only if it holds the constraint), and physics-agnostic: the same matrix :math:`J` of
the basis for every field, with a parameter per field that the user gives (the paper gives no parameter-free rule:
Coombs takes :math:`\\gamma_k = E` for the stiffness, :math:`\\gamma_M = \\rho / 4` for the mass)::

    from edelweissmeshfree.constraints.ghoststabilisation import GhostStabilisation

    mpmModel.constraints["ghost"] = GhostStabilisation("ghost", mpmModel, {"displacement": E}, journal)

It combines with the aggregation
(:class:`~edelweissmeshfree.constraints.aggregatedmpm.AggregatedMaterialPointMethod`): the solver condenses the
stabilised system, :math:`E^T (K + K_G) E`. A face between two cells that carry the same extended polynomial (an
interior cell and a boundary cell whose other nodes are all extended from it) has no jump any more and adds nothing.
"""

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable
from scipy.sparse import coo_matrix

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.numerics.aggregation import GridFaceAdjacency
from edelweissmeshfree.numerics.ghostpenalty import (
    FACE_SCALINGS,
    boundaryCellsAndGhostFaces,
    fieldsOfGhostNodes,
    ghostPenaltyMatrix,
)


class GhostStabilisation(MPMConstraintBase):
    """Ghost penalty stabilisation of the boundary cells of the active domain of an MPM model.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model, :class:`~edelweissmeshfree.models.mpmmodel.MPMModel`; its active cells are those of its MPM
        managers.
    penaltyParameters
        The penalty parameter :math:`\\gamma_f` of every field the cells carry, ``{field: gamma}``; 0 leaves a field
        unpenalised. A field of the cells without a parameter raises ValueError.
    journal
        The journal, for the summary of every new set of ghost faces.
    cells
        The cells of the background grid; default all cells of the model.
    faceScaling
        ``"neumann"`` (default; :math:`h^3/3`, Coombs after Hansbo et al.) or ``"dirichlet"`` (:math:`h/3`, Sticko
        et al.), see :mod:`~edelweissmeshfree.numerics.ghostpenalty`.
    """

    #: Carries nothing from one increment to the next.
    checkpointedState = {}

    def __init__(
        self,
        name: str,
        model,
        penaltyParameters: dict,
        journal: Journal = None,
        cells=None,
        faceScaling: str = "neumann",
    ):
        if faceScaling not in FACE_SCALINGS:
            raise ValueError(
                "Ghost penalty: faceScaling must be one of {:}, not {:}.".format(list(FACE_SCALINGS), faceScaling)
            )
        if any(gamma < 0 for gamma in penaltyParameters.values()):
            raise ValueError(
                "Ghost penalty: the penalty parameters must not be negative, {:}.".format(penaltyParameters)
            )
        self._name = name
        self._model = model
        self._journal = journal
        self._cells = list(model.cells.values()) if cells is None else list(cells)
        self._penaltyParameters = dict(penaltyParameters)
        self._faceScaling = faceScaling
        self._adjacency = None
        self._activeCellNumbers = None
        self._nodes = []
        self._fields = []
        self._nDof = 0
        self._K = None
        self._entries = (np.zeros(0, dtype=np.intc), np.zeros(0, dtype=np.intc))
        #: The boundary cells of the last set of active cells.
        self.boundaryCells = []
        #: The ghost faces of the last set of active cells, pairs of cell numbers.
        self.faces = []
        #: The scalar penalty matrix J on :attr:`nodes` (without the parameters).
        self.J = None
        self.withinReach = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def adjacency(self) -> GridFaceAdjacency:
        """The face neighbours of the cells of the grid, found once."""

        if self._adjacency is None:
            self._adjacency = GridFaceAdjacency(self._cells)
        return self._adjacency

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return self._fields

    @property
    def nDof(self) -> int:
        return self._nDof

    @property
    def scalarVariables(self) -> list:
        return []

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    @property
    def stiffness(self):
        """The penalty stiffness :math:`K_G` on the degrees of freedom of the constraint (csr), None before the first
        :meth:`updateConnectivity`."""

        return self._K

    def localEntriesOfBlock(self) -> tuple[np.ndarray, np.ndarray]:
        """The entries of :math:`K_G`: the nodes of the two cells of every ghost face couple, field by field and
        component by component."""

        return self._entries

    def updateConnectivity(self, model) -> bool:
        """Find the ghost faces of the active cells (the cells of the MPM managers, which have just assigned the
        material points) and build the penalty, if the active cells changed.

        Returns
        -------
        bool
            Whether the nodes or fields the constraint couples changed.
        """

        activeCells = self._model.activeCellsOf(self._model.mpmManagers)
        activeCellNumbers = tuple(sorted(cell.cellNumber for cell in activeCells))
        if activeCellNumbers == self._activeCellNumbers:
            return False
        self._activeCellNumbers = activeCellNumbers

        layoutBefore = [(node.label, tuple(fields)) for node, fields in zip(self._nodes, self._fields)]

        boundary, faces = boundaryCellsAndGhostFaces(self.adjacency, activeCellNumbers)
        J, nodes = ghostPenaltyMatrix(self.adjacency, faces, self._faceScaling)
        fieldsOfNode = fieldsOfGhostNodes(self.adjacency, faces, self._penaltyParameters)

        domainSize = self._model.domainSize
        penalised = {f for f, gamma in self._penaltyParameters.items() if gamma != 0.0}
        self._nodes = nodes
        self._fields = [[f for f in fieldsOfNode[node.label] if f in penalised] for node in nodes]

        # the dof of every (node, field): node after node, field after field, component after component
        offset = 0
        dofsOf = {}
        for i, fields in enumerate(self._fields):
            for f in fields:
                size = getFieldSize(f, domainSize)
                dofsOf[(i, f)] = np.arange(offset, offset + size)
                offset += size
        self._nDof = offset

        Jc = J.tocoo()
        rows, cols, vals = [], [], []
        for f in penalised:
            gamma = self._penaltyParameters[f]
            for a, b, v in zip(Jc.row, Jc.col, Jc.data):
                da, db = dofsOf.get((a, f)), dofsOf.get((b, f))
                if da is None or db is None:
                    continue
                rows.append(da)
                cols.append(db)
                vals.append(np.full(len(da), gamma * v))
        if rows:
            K = coo_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(self._nDof, self._nDof)
            ).tocsr()
        else:
            K = coo_matrix((self._nDof, self._nDof)).tocsr()
        K.sum_duplicates()
        K.sort_indices()
        self._K = K
        self._entries = (
            np.repeat(np.arange(self._nDof), np.diff(K.indptr)).astype(np.intc),
            K.indices.astype(np.intc),
        )
        self.J = J
        self.boundaryCells = boundary
        self.faces = faces
        self.withinReach = self._nDof > 0 and K.nnz > 0

        if self._journal is not None:
            self._journal.message(
                "{:} active cells, {:} boundary cells: {:} ghost faces, {:} nodes, {:} dof".format(
                    len(activeCellNumbers), len(boundary), len(faces), len(nodes), self._nDof
                ),
                self._name,
            )

        layout = [(node.label, tuple(fields)) for node, fields in zip(self._nodes, self._fields)]
        return layout != layoutBefore

    def computeKernels(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, K: np.ndarray, timeStep: TimeStep):
        """Add the ghost force :math:`K_G \\Delta U` and the stiffness :math:`K_G` (constant over the step)."""

        P += self._K @ dU
        K += self._K.data

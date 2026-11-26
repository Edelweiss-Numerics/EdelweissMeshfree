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
"""A projection over a patch of background cells, e.g. the F-bar-patch formulation (de Souza Neto, Pires and Owen,
IJNME 62 (2005) 353-383) with the aggregates of the aggregated MPM as patches.

The cells of a patch share one scalar, the weighted mean of the contributions of all their material points,

.. math::

    \\bar{v} = \\frac{\\sum_c v_c(\\boldsymbol{Q}_c)}{\\sum_c w_c},

e.g. for the F-bar patch :math:`v_c = \\sum_{p \\in c} V_p \\det \\Delta\\boldsymbol{F}_p`, :math:`w_c = \\sum_{p\\in c}
V_p`. The physics is that of the Marmot cells (``hasPatchProjection``); this module only drives the two passes and
assembles the coupling across the cells of the patch, so it is physics-agnostic:

1. before the cells interpolate to their material points (:meth:`CellPatchProjection.projectToCells`, called by
   :meth:`~edelweissmeshfree.models.mpmdiscretesystem.MPMDiscreteSystem.updateMaterialPoints`), every cell reports
   :math:`v_c, w_c, \\partial v_c/\\partial \\boldsymbol{Q}_c`, and every cell gets :math:`\\bar{v}`;
2. the cells return their residual and the tangent at fixed :math:`\\bar{v}` as usual; the patch is an entity of the
   discrete system over the union of the dofs of its cells, which adds the coupling
   :math:`\\sum_{c,d} \\partial \\boldsymbol{R}_c/\\partial \\bar{v} \\otimes \\partial v_d/\\partial \\boldsymbol{Q}_d /
   \\sum w` to the tangent (:meth:`CellPatchProjection.computeKernels`; no residual of its own).

The tangent of the patch is therefore exact, also across the cells. The entities are constraints of the discrete system
in the sense of EdelweissFE (an entity with nodes and a dense block), so that they are assembled before the multi-point
constraints condense the system (aggregated MPM: :math:`E^T K E`).
"""

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.timesteppers.timestep import TimeStep


class CellPatchProjection(ConstraintBase):
    """The patch projection over a set of cells, an entity of the discrete system over the union of their nodes.

    Parameters
    ----------
    name
        The name.
    cells
        The cells of the patch; all must support it (``hasPatchProjection``).
    dimension
        The spatial dimension, for the sizes of the fields.
    """

    #: The patch is evaluated before the cells interpolate, see :meth:`projectToCells`.
    projectsOverCells = True

    def __init__(self, name: str, cells: list, dimension: int):
        self._name = name
        self._cells = list(cells)
        self.scalarVariables = []

        nodeFields = {}
        for cell in self._cells:
            for node, fields in zip(cell.nodes, cell.fields):
                nodeFields.setdefault(node, list(fields))
        self._nodes = sorted(nodeFields, key=lambda n: n.label)
        self._fields = [nodeFields[n] for n in self._nodes]

        offset = {}
        n = 0
        for node, fields in zip(self._nodes, self._fields):
            offset[node] = n
            n += sum(getFieldSize(f, dimension) for f in fields)
        self._nDof = n

        # the position in the patch of every cell dof (cell dof i = node-wise dof perm[i] of the cell)
        self._cellDofs = []
        for cell in self._cells:
            nodeWise = []
            for node, fields in zip(cell.nodes, cell.fields):
                assert fields == nodeFields[node], "the cells of a patch must carry the same fields on a node"
                nodeWise.extend(offset[node] + np.arange(sum(getFieldSize(f, dimension) for f in fields)))
            perm = cell.dofIndicesPermutation
            nodeWise = np.asarray(nodeWise, dtype=int)
            self._cellDofs.append(nodeWise[np.asarray(perm, dtype=int)] if perm is not None else nodeWise)

        self._dValue = np.zeros(self._nDof)
        #: The patch value of the last evaluation.
        self.value = 1.0
        #: The weight of the patch (sum of the weights of its material points) of the last evaluation.
        self.weight = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def cells(self) -> list:
        """The cells of the patch."""
        return self._cells

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return self._fields

    @property
    def nDof(self) -> int:
        return self._nDof

    def projectToCells(self, dU):
        """Pass 1 and 2: the patch value at the trial increment, passed to every cell of the patch. Before the cells
        interpolate to their material points.

        Parameters
        ----------
        dU
            The trial solution increment of the discrete system (indexed by the cells).
        """

        value = weight = 0.0
        self._dValue[:] = 0.0
        for cell, dofs in zip(self._cells, self._cellDofs):
            dv = np.zeros(cell.nDof)
            v, w = cell.computeProjectionContribution(np.ascontiguousarray(dU[cell]), dv)
            value += v
            weight += w
            np.add.at(self._dValue, dofs, dv)
        if weight <= 0.0:
            raise RuntimeError("patch projection {:}: no material point with a positive weight".format(self._name))
        self.value = value / weight
        self.weight = weight
        self._dValue /= weight
        for cell in self._cells:
            cell.setProjectedValue(self.value)

    def computeKernels(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, K: np.ndarray, timeStep: TimeStep):
        """The coupling of the cells of the patch through the patch value, after the cells were computed: no
        residual, the tangent :math:`\\partial \\boldsymbol{R}/\\partial \\bar{v} \\otimes \\partial \\bar{v} /
        \\partial \\boldsymbol{Q}` over the dofs of the patch."""

        dR = np.zeros(self._nDof)
        for cell, dofs in zip(self._cells, self._cellDofs):
            a = np.zeros(cell.nDof)
            cell.getProjectionSensitivity(a)
            np.add.at(dR, dofs, a)
        K += np.outer(dR, self._dValue)


def patchesOfCells(cells: list, groups: list[list[int]], dimension: int, name: str = "patch") -> list:
    """The patch entities of groups of cells: a group whose cells all support the patch projection is a patch; in a
    group with cells that do not, every cell that supports it is a patch of its own.

    Parameters
    ----------
    cells
        The cells, by number: ``{cellNumber: cell}``.
    groups
        The groups of cell numbers.
    dimension
        The spatial dimension.
    name
        The prefix of the names of the patches.

    Returns
    -------
    list[CellPatchProjection]
        The patches.
    """

    patches = []
    for group in groups:
        members = [cells[c] for c in group]
        if all(getattr(cell, "hasPatchProjection", False) for cell in members):
            patches.append(CellPatchProjection("{:} {:}".format(name, group[0]), members, dimension))
        else:
            for cell in members:
                if getattr(cell, "hasPatchProjection", False):
                    patches.append(CellPatchProjection("{:} {:}".format(name, cell.cellNumber), [cell], dimension))
    return patches

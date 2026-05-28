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
"""
A pure Python implementation of a bilinear Quad4 cell for 2D displacement problems.

This cell uses standard bilinear shape functions on a quadrilateral to interpolate
displacements from grid nodes to material points, and to assemble residual vectors
and stiffness matrices from material point contributions.
"""

import numpy as np
from edelweissfe.points.node import Node

from edelweissmeshfree.cells.base.cell import CellBase
from edelweissmeshfree.materialpoints.base.mp import MaterialPointBase


def _compute_shape_functions_and_gradients(xi, eta, node_coords):
    """Compute shape functions and their spatial gradients at a given parametric point.

    Parameters
    ----------
    xi
        Parametric coordinate xi in [-1, 1].
    eta
        Parametric coordinate eta in [-1, 1].
    node_coords
        (4, 2) array of node coordinates.

    Returns
    -------
    N
        Shape function values (4,).
    dNdx
        Shape function gradients w.r.t. physical coordinates (4, 2).
    detJ
        Determinant of the Jacobian.
    """
    # Bilinear shape functions in parametric space
    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )

    # Derivatives w.r.t. parametric coordinates
    dNdxi = 0.25 * np.array(
        [
            [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
            [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
        ]
    )

    # Jacobian: J = dNdxi @ node_coords -> (2, 2)
    J = dNdxi @ node_coords

    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

    # Inverse Jacobian
    invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ

    # Derivatives w.r.t. physical coordinates
    dNdx = invJ @ dNdxi  # (2, 4)

    return N, dNdx.T, detJ  # dNdx returned as (4, 2)


def _global_to_parametric(coordinate, node_coords):
    """Map a global coordinate to parametric coordinates using Newton iteration.

    Parameters
    ----------
    coordinate
        Global (x, y) coordinate.
    node_coords
        (4, 2) array of node coordinates.

    Returns
    -------
    xi, eta
        Parametric coordinates.
    """
    xi, eta = 0.0, 0.0

    for _ in range(20):
        N = 0.25 * np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ]
        )

        x_current = N @ node_coords
        residual = coordinate - x_current

        if np.linalg.norm(residual) < 1e-12:
            break

        dNdxi = 0.25 * np.array(
            [
                [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
                [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)],
            ]
        )

        J = dNdxi @ node_coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ

        dxi = invJ @ residual
        xi += dxi[0]
        eta += dxi[1]

    return xi, eta


class PythonCell(CellBase):
    """A pure Python bilinear Quad4 cell for 2D displacement problems.

    This cell computes shape functions and their gradients to:
    - Interpolate displacement increments from nodes to material points
    - Assemble internal force vectors and stiffness matrices from material point stress/tangent

    Parameters
    ----------
    cellType
        A string identifying the cell formulation (unused for this implementation).
    cellNumber
        A unique integer label.
    nodes
        The list of 4 Nodes assigned to this cell.
    """

    def __init__(self, cellType: str, cellNumber: int, nodes: list[Node]):
        """Initialize the Python cell.

        Parameters
        ----------
        cellType
            The identifier of the cell formulation.
        cellNumber
            The unique number of the cell.
        nodes
            The nodes attached to the cell.
        """
        self._cellType = cellType
        self._cellNumber = cellNumber
        self._assignedMaterialPoints = []

        self.nodes = nodes
        self._node_coords = np.array([n.coordinates for n in nodes])

        self._bb_min = np.min(self._node_coords, axis=0)
        self._bb_max = np.max(self._node_coords, axis=0)

    @property
    def cellNumber(self) -> int:
        """The unique number of this cell."""
        return self._cellNumber

    @property
    def nNodes(self) -> int:
        """The number of nodes of this cell."""
        return 4

    @property
    def nDof(self) -> int:
        """The number of degrees of freedom of this cell."""
        return 8

    @property
    def fields(self) -> list[list[str]]:
        """The fields defined on the nodes of this cell."""
        return [["displacement"]] * 4

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        """The permutation mapping local cell dofs to node-major ordering."""
        return np.arange(8, dtype=int)

    @property
    def ensightType(self) -> str:
        """The EnSight geometry type of this cell."""
        return "quad4"

    @property
    def assignedMaterialPoints(self) -> list:
        """The material points currently assigned to this cell."""
        return self._assignedMaterialPoints

    def assignMaterialPoints(self, materialPoints: list):
        """Assign material points to this cell.

        Parameters
        ----------
        materialPoints
            The list of material points to assign.
        """
        self._assignedMaterialPoints = materialPoints

    def _get_B_matrix(self, dNdx):
        """Compute the strain-displacement matrix B for plane strain/stress.

        Parameters
        ----------
        dNdx
            Shape function gradients (4, 2).

        Returns
        -------
        B
            (4, 8) strain-displacement matrix for Voigt notation [eps_xx, eps_yy, eps_zz, gamma_xy].
        """
        B = np.zeros((4, 8))
        for i in range(4):
            B[0, 2 * i] = dNdx[i, 0]  # eps_xx
            B[1, 2 * i + 1] = dNdx[i, 1]  # eps_yy
            # B[2, :] = 0  # eps_zz (plane strain)
            B[3, 2 * i] = dNdx[i, 1]  # gamma_xy
            B[3, 2 * i + 1] = dNdx[i, 0]  # gamma_xy
        return B

    def interpolateFieldsToMaterialPoints(self, dU: np.ndarray):
        """Interpolate nodal displacement increments to material points.

        Parameters
        ----------
        dU
            The nodal displacement increment vector (8,) = [u1x, u1y, u2x, u2y, u3x, u3y, u4x, u4y].
        """
        for mp in self._assignedMaterialPoints:
            mp_coords = mp.getCenterCoordinates()
            xi, eta = _global_to_parametric(mp_coords, self._node_coords)
            N, dNdx, _ = _compute_shape_functions_and_gradients(xi, eta, self._node_coords)

            # Interpolate displacement increment
            dU_mp = np.zeros(2)
            for i in range(4):
                dU_mp[0] += N[i] * dU[2 * i]
                dU_mp[1] += N[i] * dU[2 * i + 1]

            # Compute strain increment from B * dU
            B = self._get_B_matrix(dNdx)
            dStrain = B @ dU

            mp.interpolateFieldsFromCell(dU_mp, dStrain)

    def interpolateSolutionContributionToMaterialPoints(self, dU: np.ndarray):
        """Alias for interpolateFieldsToMaterialPoints (base class interface)."""
        self.interpolateFieldsToMaterialPoints(dU)

    def computeMaterialPointKernels(
        self,
        dU: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute residual and stiffness contributions from all assigned material points.

        Parameters
        ----------
        dU
            The current solution increment (8,).
        P
            The internal load vector to be filled (8,).
        K
            The stiffness matrix to be filled (64,) in row-major order.
        timeTotal
            The current total time.
        dTime
            The time increment.
        """
        K_mat = K.reshape(8, 8)

        for mp in self._assignedMaterialPoints:
            mp_coords = mp.getCenterCoordinates()
            xi, eta = _global_to_parametric(mp_coords, self._node_coords)
            N, dNdx, detJ = _compute_shape_functions_and_gradients(xi, eta, self._node_coords)

            B = self._get_B_matrix(dNdx)
            volume = mp.getVolume()

            # Get stress from material point (Voigt: [s_xx, s_yy, s_zz, s_xy])
            stress = mp.getResultArray("stress")

            # Internal force: P += B^T * sigma * volume
            P += B.T @ stress * volume

            # Stiffness: K += B^T * C * B * volume
            C = mp.getAlgorithmicTangent()
            K_mat += B.T @ C @ B * volume

    def computeBodyLoad(
        self,
        loadType: str,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute body load contribution (e.g., gravity)."""
        for mp in self._assignedMaterialPoints:
            mp_coords = mp.getCenterCoordinates()
            xi, eta = _global_to_parametric(mp_coords, self._node_coords)
            N, dNdx, detJ = _compute_shape_functions_and_gradients(xi, eta, self._node_coords)
            volume = mp.getVolume()

            for i in range(4):
                P[2 * i] -= N[i] * load[0] * volume
                P[2 * i + 1] -= N[i] * load[1] * volume

    def computeDistributedLoad(
        self,
        loadType: str,
        surfaceID: int,
        materialPoint: MaterialPointBase,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute distributed (surface) load for a specific material point."""
        mp_coords = materialPoint.getCenterCoordinates()
        xi, eta = _global_to_parametric(mp_coords, self._node_coords)
        N, dNdx, detJ = _compute_shape_functions_and_gradients(xi, eta, self._node_coords)
        volume = materialPoint.getVolume()

        for i in range(4):
            P[2 * i] -= N[i] * load[0] * volume
            P[2 * i + 1] -= N[i] * load[1] * volume

    def getCoordinatesAtCenter(self) -> np.ndarray:
        """Return the coordinates at the centroid of this cell."""
        return np.mean(self._node_coords, axis=0)

    def isCoordinateInCell(self, coordinate: np.ndarray) -> bool:
        """Return True if the given coordinate lies inside the cell bounding box.

        Parameters
        ----------
        coordinate
            The spatial coordinate to test.
        """
        x, y = coordinate[0], coordinate[1]
        if x >= self._bb_min[0] and x <= self._bb_max[0]:
            if y >= self._bb_min[1] and y <= self._bb_max[1]:
                return True
        return False

    def getBoundingBox(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the axis-aligned bounding box of this cell as (min, max) coordinate arrays."""
        return (self._bb_min.copy(), self._bb_max.copy())

    def getInterpolationVector(self, coordinate: np.ndarray) -> np.ndarray:
        """Return the bilinear shape function values at the given coordinate.

        Parameters
        ----------
        coordinate
            The spatial coordinate at which to evaluate the shape functions.

        Returns
        -------
        np.ndarray
            Shape function values at the requested coordinate (4,).
        """
        xi, eta = _global_to_parametric(coordinate, self._node_coords)
        N = 0.25 * np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ]
        )
        return N

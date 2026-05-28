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
"""
Implementing your own cells can be done easily by subclassing from
the abstract base class :class:`~CellBase`.
"""

import numpy as np
from edelweissfe.points.node import Node

from edelweissmeshfree.cells.base.cell import CellBase
from edelweissmeshfree.materialpoints.base.mp import MaterialPointBase


class Cell(CellBase):
    """A minimal concrete implementation of CellBase for use in unit tests."""

    def __init__(self, cellType: str, cellNumber: int, nodes: list[Node]):
        """Initialize the test cell."""
        self._cellType = cellType
        self._cellNumber = cellNumber
        self._assignedMaterialPoints = list()

        self.nodes = nodes
        self.coordinates = np.array([n.coordinates for n in nodes])

        self.x_min = np.min(self.coordinates[:, 0])
        self.x_max = np.max(self.coordinates[:, 0])
        self.y_min = np.min(self.coordinates[:, 1])
        self.y_max = np.max(self.coordinates[:, 1])

    @property
    def cellNumber(self) -> int:
        """Return the cell number."""
        return self._cellNumber

    @property
    def nNodes(self) -> int:
        """Return the number of nodes."""
        return 4

    @property
    def nDof(self) -> int:
        """Return the number of degrees of freedom."""
        return 4 * 2

    @property
    def fields(self) -> list[list[str]]:
        """Return the nodal fields."""
        return [
            [
                "displacement",
            ]
        ] * 4

    @property
    def dofIndicesPermutation(self) -> np.ndarray:
        """Return the local-to-global dof permutation."""
        return np.arange(0, 8, dtype=int)

    @property
    def ensightType(self) -> str:
        """Return the EnSight cell type."""
        return "quad4"

    @property
    def assignedMaterialPoints(self) -> list:
        """Return the assigned material points."""
        return self._assignedMaterialPoints

    def assignMaterialPoints(self, materialPoints):
        """Assign the material points to the cell."""
        self.materialPoints = materialPoints

    def interpolateSolutionContributionToMaterialPoints(
        self,
        materialPoints: list[MaterialPointBase],
        dU: np.ndarray,
    ):
        """Delegate interpolation of the solution contribution to the material points."""
        pass

    def computeMaterialPointKernels(
        self,
        materialPoints: list[MaterialPointBase],
        P: np.ndarray,
        K: np.ndarray,
        timeStep: float,
        timeTotal: float,
        dTime: float,
    ):
        """Compute material point kernel contributions for the test cell."""
        pass

    def computeBodyLoad(
        self,
        loadType: str,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute the body-load contribution for the test cell."""
        pass

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
        """Compute the distributed-load contribution for the test cell."""
        pass

    def getCoordinatesAtCenter(self) -> np.ndarray:
        """Return the coordinates at the cell center."""
        pass

    def getBoundingBox(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the cell bounding box."""
        return (np.array([self.x_min, self.y_min]), np.array([self.x_max, self.y_max]))

    def isCoordinateInCell(self, coordinate: np.ndarray) -> bool:
        """Return whether the coordinate lies inside the cell."""
        x, y = coordinate

        if x >= self.x_min and x <= self.x_max:
            if y >= self.y_min and y <= self.y_max:
                return True

        return False

    def getInterpolationVector(self, coordinate) -> np.ndarray:
        """Return the interpolation vector at the given coordinate."""
        raise Exception("not implemented!")

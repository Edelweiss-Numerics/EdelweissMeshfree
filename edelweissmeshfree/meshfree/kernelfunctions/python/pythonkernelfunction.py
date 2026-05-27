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
A pure Python implementation of meshfree kernel functions (B-Spline) for teaching purposes.

This module provides a cubic B-Spline kernel function with compact (boxed) support,
suitable for use in RKPM (Reproducing Kernel Particle Method) approximations.
"""

import numpy as np
from edelweissfe.points.node import Node

from edelweissmeshfree.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)


def _cubic_bspline_1d(r):
    """Evaluate the normalized cubic B-spline in 1D.

    Parameters
    ----------
    r
        The normalized distance |x - x_I| / supportRadius, must be >= 0.

    Returns
    -------
    float
        The kernel function value.
    """
    if r < 0.5:
        return 2.0 / 3.0 - 4.0 * r**2 + 4.0 * r**3
    elif r < 1.0:
        return 4.0 / 3.0 - 4.0 * r + 4.0 * r**2 - 4.0 / 3.0 * r**3
    else:
        return 0.0


def _cubic_bspline_1d_derivative(r):
    """Evaluate the derivative of the normalized cubic B-spline in 1D.

    Parameters
    ----------
    r
        The normalized distance |x - x_I| / supportRadius, must be >= 0.

    Returns
    -------
    float
        The derivative of the kernel function value w.r.t. r.
    """
    if r < 0.5:
        return -8.0 * r + 12.0 * r**2
    elif r < 1.0:
        return -4.0 + 8.0 * r - 4.0 * r**2
    else:
        return 0.0


class PythonKernelFunction(BaseMeshfreeKernelFunction):
    """A pure Python implementation of a cubic B-spline kernel function with boxed support.

    The kernel function is defined as a product of 1D cubic B-splines in each dimension
    (tensor product structure), providing compact support within a box of side length
    2 * supportRadius centered at the kernel function center.

    Parameters
    ----------
    node
        The node associated with this kernel function.
    supportRadius
        The support radius of the kernel function.
    """

    def __init__(self, node: Node, supportRadius: float = 1.0):
        self._node = node
        self._center = np.copy(node.coordinates)
        self._supportRadius = supportRadius
        self._dimension = len(self._center)

    @property
    def node(self) -> Node:
        return self._node

    @property
    def center(self) -> np.ndarray:
        return self._center

    def updateCenter(self, center: np.ndarray):
        self._center[:] = center

    def moveTo(self, coordinates: np.ndarray):
        """Move the kernel function center to new coordinates.

        Parameters
        ----------
        coordinates
            The new center coordinates.
        """
        self._center[:] = coordinates

    def getCurrentBoundingBox(self) -> tuple[np.ndarray, np.ndarray]:
        return self.getBoundingBox()

    def getBoundingBox(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the kernel function support.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (min_corner, max_corner) of the bounding box.
        """
        bbMin = self._center - self._supportRadius
        bbMax = self._center + self._supportRadius
        return bbMin, bbMax

    def isCoordinateInCurrentSupport(self, coords: np.ndarray) -> bool:
        """Check if a single coordinate is within the kernel support.

        Parameters
        ----------
        coords
            The coordinate to check (1D array of length dimension).

        Returns
        -------
        bool
            True if the coordinate is within the support.
        """
        diff = np.abs(coords - self._center)
        return bool(np.all(diff < self._supportRadius))

    def isAnyCoordinateInSupport(self, coords: np.ndarray) -> bool:
        """Check if any of the given coordinates are within the kernel support.

        Parameters
        ----------
        coords
            The coordinates to check (2D array of shape (n_points, dimension)).

        Returns
        -------
        bool
            True if any coordinate is within the support.
        """
        for i in range(coords.shape[0]):
            if self.isCoordinateInCurrentSupport(coords[i]):
                return True
        return False

    def computeKernelFunction(self, coords: np.ndarray) -> float:
        """Evaluate the kernel function at the given coordinates.

        The kernel function is computed as a tensor product of 1D cubic B-splines.

        Parameters
        ----------
        coords
            The coordinates at which to evaluate (1D array of length dimension).

        Returns
        -------
        float
            The kernel function value.
        """
        result = 1.0
        for d in range(self._dimension):
            r = abs(coords[d] - self._center[d]) / self._supportRadius
            result *= _cubic_bspline_1d(r) / self._supportRadius
        return result

    def computeKernelFunctionAndGradient(self, coords: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate the kernel function and its gradient at the given coordinates.

        Parameters
        ----------
        coords
            The coordinates at which to evaluate (1D array of length dimension).

        Returns
        -------
        tuple[float, np.ndarray]
            The kernel function value and its gradient (1D array of length dimension).
        """
        values_1d = np.zeros(self._dimension)
        derivs_1d = np.zeros(self._dimension)

        for d in range(self._dimension):
            diff = coords[d] - self._center[d]
            r = abs(diff) / self._supportRadius
            values_1d[d] = _cubic_bspline_1d(r) / self._supportRadius
            sign = 1.0 if diff >= 0 else -1.0
            derivs_1d[d] = sign * _cubic_bspline_1d_derivative(r) / (self._supportRadius**2)

        # Tensor product: value = prod(values_1d)
        value = np.prod(values_1d)

        # Gradient via product rule
        gradient = np.zeros(self._dimension)
        for d in range(self._dimension):
            gradient[d] = derivs_1d[d]
            for d2 in range(self._dimension):
                if d2 != d:
                    gradient[d] *= values_1d[d2]

        return value, gradient

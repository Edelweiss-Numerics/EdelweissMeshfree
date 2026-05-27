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
A pure Python implementation of the Reproducing Kernel Particle Method (RKPM) approximation
for teaching purposes.

RKPM constructs corrected shape functions from kernel functions such that polynomial
completeness of a specified order is achieved. The corrected shape function for node I
evaluated at point x is:

    N_I(x) = C(x) · P(x_I - x) · phi_I(x)

where:
    - phi_I(x) is the kernel function centered at node I
    - P(x_I - x) is the polynomial basis vector evaluated at (x_I - x)
    - C(x) is the correction vector ensuring polynomial completeness

The correction vector C(x) is determined by the reproducing condition:

    sum_I N_I(x) · p(x_I) = p(x)  for all polynomials p of the given order

This leads to the moment matrix system:

    M(x) · C(x) = P(0)

where M(x) = sum_I P(x_I - x) · P(x_I - x)^T · phi_I(x)

References
----------
- Liu, W.K., Jun, S., Zhang, Y.F. (1995). Reproducing kernel particle methods.
  International Journal for Numerical Methods in Fluids, 20(8-9), 1081-1106.
- Chen, J.S., Hillman, M., Ruter, M. (2013). An arbitrary order variationally consistent
  integration for Galerkin meshfree methods.
"""

import numpy as np

from edelweissmeshfree.meshfree.approximations.base.basemeshfreeapproximation import (
    BaseMeshfreeApproximation,
)
from edelweissmeshfree.meshfree.kernelfunctions.python.pythonkernelfunction import (
    PythonKernelFunction,
)


def _build_polynomial_basis_2d(dx, dy, order):
    """Build the polynomial basis vector for 2D RKPM.

    Parameters
    ----------
    dx
        x - x_I (or x_I - x depending on convention).
    dy
        y - y_I.
    order
        Completeness order (0, 1, or 2).

    Returns
    -------
    np.ndarray
        The polynomial basis vector.
    """
    if order == 0:
        return np.array([1.0])
    elif order == 1:
        return np.array([1.0, dx, dy])
    elif order == 2:
        return np.array([1.0, dx, dy, dx * dx, dx * dy, dy * dy])
    else:
        raise ValueError(f"Completeness order {order} not supported (max 2).")


def _build_polynomial_basis_gradient_2d(dx, dy, order):
    """Build the gradient of the polynomial basis vector w.r.t. coordinates for 2D.

    Returns dP/dx and dP/dy as two arrays.

    Parameters
    ----------
    dx, dy
        Differences (x_I - x).
    order
        Completeness order.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (dP/d(dx), dP/d(dy)) - gradients of the basis w.r.t. dx, dy.
    """
    if order == 0:
        return np.array([0.0]), np.array([0.0])
    elif order == 1:
        return np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    elif order == 2:
        return (
            np.array([0.0, 1.0, 0.0, 2.0 * dx, dy, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, dx, 2.0 * dy]),
        )
    else:
        raise ValueError(f"Completeness order {order} not supported (max 2).")


class PythonRKPMApproximation(BaseMeshfreeApproximation):
    """A pure Python RKPM (Reproducing Kernel) approximation.

    Computes corrected meshfree shape functions that achieve polynomial
    completeness of the specified order.

    Parameters
    ----------
    dimension
        The spatial dimension (currently only 2 is supported).
    completenessOrder
        The polynomial completeness order (0, 1, or 2).
    """

    def __init__(self, dimension: int, completenessOrder: int = 1):
        if dimension != 2:
            raise NotImplementedError("Only 2D is currently supported for PythonRKPMApproximation.")
        self._dimension = dimension
        self._completenessOrder = completenessOrder

    @property
    def completenessOrder(self) -> int:
        """The polynomial completeness order."""
        return self._completenessOrder

    def computeShapeFunctionValues(self, coordinates: np.ndarray, kernelfunctions: list) -> np.ndarray:
        """Compute the RKPM shape function values at the given coordinates.

        Parameters
        ----------
        coordinates
            The evaluation point (1D array of length dimension).
        kernelfunctions
            The list of kernel functions contributing to the approximation.

        Returns
        -------
        np.ndarray
            The shape function values at the given coordinates (one per kernel function).
        """
        N, _ = self.computeShapeFunctionsAndGradients(coordinates, kernelfunctions)
        return N

    def computeShapeFunctionsAndGradients(
        self, coordinates: np.ndarray, kernelfunctions: list
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the RKPM shape functions and their spatial gradients.

        The corrected shape function for node I at point x:
            N_I(x) = C(x)^T · P(x_I - x) · phi_I(x)

        where C(x) = M(x)^{-1} · P(0)  and  M(x) = sum_J P(x_J - x) P(x_J - x)^T phi_J(x)

        Parameters
        ----------
        coordinates
            The evaluation point (1D array of length dimension).
        kernelfunctions
            The list of kernel functions.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            N: shape function values (nKernels,)
            dNdx: shape function gradients (nKernels, dimension)
        """
        x = coordinates
        nKernels = len(kernelfunctions)
        order = self._completenessOrder

        # Evaluate kernel functions and polynomial bases
        phi = np.zeros(nKernels)
        dphi = np.zeros((nKernels, self._dimension))
        P = []

        for I, kf in enumerate(kernelfunctions):
            xI = kf.center
            phi_val, phi_grad = kf.computeKernelFunctionAndGradient(x)
            phi[I] = phi_val
            dphi[I, :] = phi_grad

            dx = xI[0] - x[0]
            dy = xI[1] - x[1]
            P.append(_build_polynomial_basis_2d(dx, dy, order))

        P = np.array(P)  # (nKernels, nBasis)
        nBasis = P.shape[1]

        # Compute moment matrix M(x) = sum_I P_I P_I^T phi_I
        M = np.zeros((nBasis, nBasis))
        for I in range(nKernels):
            M += np.outer(P[I], P[I]) * phi[I]

        # P(0) = [1, 0, 0, ...] (polynomial basis evaluated at zero)
        P0 = np.zeros(nBasis)
        P0[0] = 1.0

        # Correction vector: C(x) = M(x)^{-1} P(0)
        C = np.linalg.solve(M, P0)

        # Shape functions: N_I(x) = C^T P_I phi_I
        N = np.zeros(nKernels)
        for I in range(nKernels):
            N[I] = C @ P[I] * phi[I]

        # Now compute gradients dN_I/dx_j
        # We need dM/dx_j, dP_I/dx_j, and dphi_I/dx_j
        # Note: dx = x_I - x, so d(dx)/d(x_j) = -delta_j (the sign matters!)

        dNdx = np.zeros((nKernels, self._dimension))

        for j in range(self._dimension):
            # Compute dM/dx_j = sum_I [dP_I/dx_j P_I^T + P_I dP_I/dx_j^T] phi_I + P_I P_I^T dphi_I/dx_j
            dM_dxj = np.zeros((nBasis, nBasis))
            for I in range(nKernels):
                xI = kernelfunctions[I].center
                dx = xI[0] - x[0]
                dy = xI[1] - x[1]
                dPdx, dPdy = _build_polynomial_basis_gradient_2d(dx, dy, order)
                # d(x_I - x)/dx_j = -1 if j matches
                dP_dxj = -(dPdx if j == 0 else dPdy)

                dM_dxj += (np.outer(dP_dxj, P[I]) + np.outer(P[I], dP_dxj)) * phi[I]
                dM_dxj += np.outer(P[I], P[I]) * dphi[I, j]

            # dC/dx_j = -M^{-1} dM/dx_j M^{-1} P0 = -M^{-1} dM/dx_j C
            dC_dxj = -np.linalg.solve(M, dM_dxj @ C)

            for I in range(nKernels):
                xI = kernelfunctions[I].center
                dx = xI[0] - x[0]
                dy = xI[1] - x[1]
                dPdx, dPdy = _build_polynomial_basis_gradient_2d(dx, dy, order)
                dP_dxj_I = -(dPdx if j == 0 else dPdy)

                # dN_I/dx_j = dC^T P_I phi_I + C^T dP_I/dx_j phi_I + C^T P_I dphi_I/dx_j
                dNdx[I, j] = dC_dxj @ P[I] * phi[I] + C @ dP_dxj_I * phi[I] + C @ P[I] * dphi[I, j]

        return N, dNdx

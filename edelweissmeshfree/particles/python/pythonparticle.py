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
A pure Python implementation of an RKPM (Reproducing Kernel Particle Method) particle
for 2D plane strain problems with linear elastic material behavior.

This particle uses RKPM shape functions (computed from kernel functions and a meshfree
approximation) to:
- Interpolate nodal displacement to the particle location
- Compute strain from shape function gradients
- Evaluate stress using linear elasticity
- Assemble residual and stiffness contributions

The particle also supports Variationally Consistent Integration (VCI) correction terms
to ensure Galerkin exactness.

This implementation is intended for teaching and reference purposes.
"""

import numpy as np

from edelweissmeshfree.meshfree.approximations.python.pythonrkpmapproximation import (
    PythonRKPMApproximation,
    _build_polynomial_basis_2d,
    _build_polynomial_basis_gradient_2d,
)
from edelweissmeshfree.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


def _compute_plane_strain_tangent(E, nu):
    """Compute the 2D plane strain elastic tangent matrix (4x4 Voigt).

    Voigt ordering: [eps_xx, eps_yy, eps_zz, gamma_xy]

    Parameters
    ----------
    E
        Young's modulus.
    nu
        Poisson's ratio.

    Returns
    -------
    C
        (4, 4) elastic tangent matrix.
    """
    factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    C = np.zeros((4, 4))
    C[0, 0] = factor * (1.0 - nu)
    C[0, 1] = factor * nu
    C[0, 2] = factor * nu
    C[1, 0] = factor * nu
    C[1, 1] = factor * (1.0 - nu)
    C[1, 2] = factor * nu
    C[2, 0] = factor * nu
    C[2, 1] = factor * nu
    C[2, 2] = factor * (1.0 - nu)
    C[3, 3] = factor * (1.0 - 2.0 * nu) / 2.0
    return C


class PythonParticle(BaseParticle):
    """A pure Python RKPM particle for 2D plane strain linear elasticity.

    This particle implements the full RKPM workflow:
    1. Kernel functions are assigned by the particle manager
    2. Shape functions are computed using the RKPM approximation
    3. Strain is computed from shape function gradients and nodal displacements
    4. Stress is computed from linear elastic constitutive law
    5. Residual (internal force) and stiffness are assembled

    Parameters
    ----------
    particleType
        A formulation string (e.g., "Displacement/PlaneStrain/Point").
    number
        Unique particle ID.
    vertexCoordinates
        The coordinates of the particle (shape (1, 2) or (2,)).
    volume
        The associated volume (area × thickness for 2D).
    approximation
        The RKPM approximation object.
    material
        A dict with keys "material" (str) and "properties" (array-like: [E, nu]).
    """

    def __init__(
        self,
        particleType: str,
        number: int,
        vertexCoordinates: np.ndarray,
        volume: float,
        approximation: PythonRKPMApproximation,
        material: dict,
    ):
        self._particleType = particleType
        self._number = number
        self._vertexCoordinates = np.asarray(vertexCoordinates, dtype=float).reshape(-1, 2)
        self._centerCoordinates = self._vertexCoordinates.mean(axis=0)
        self._volume = volume
        self._approximation = approximation
        self._nDim = 2

        # Fields
        self._baseFields = ["displacement"]
        self._fields = []

        # Kernel functions and nodes
        self._assignedKernelFunctions = []
        self._nodes = []
        self._nAssignedKernelFunctions = 0

        # State variables
        self._displacement = np.zeros(2)
        self._displacement_temp = np.zeros(2)
        self._strain = np.zeros(4)  # [eps_xx, eps_yy, eps_zz, gamma_xy]
        self._strain_temp = np.zeros(4)
        self._stress = np.zeros(4)
        self._stress_temp = np.zeros(4)

        # Material tangent
        self._C = np.zeros((4, 4))

        # VCI correction terms: eta_AjC shape (nKernels, nDim, nVCIConstraints)
        self._eta = None
        self._nVCIConstraints = 0

        # Set material
        materialName = material.get("material", "LinearElastic")
        materialProperties = np.asarray(material.get("properties", []))
        self._setMaterial(materialName, materialProperties)

    def _setMaterial(self, materialName: str, materialProperties: np.ndarray):
        """Set the material model."""
        if materialName.lower().replace(" ", "") == "linearelastic":
            E = materialProperties[0]
            nu = materialProperties[1]
            self._C = _compute_plane_strain_tangent(E, nu)
        else:
            raise ValueError(f"Unknown material '{materialName}' for PythonParticle.")

    # -------- Properties required by BaseParticle --------

    @property
    def dimension(self) -> int:
        return self._nDim

    @property
    def baseFields(self) -> list[str]:
        return self._baseFields

    @property
    def fields(self):
        return self._fields

    @property
    def propertyNames(self) -> list[str]:
        return []

    @property
    def nodes(self):
        return self._nodes

    @property
    def nDof(self) -> int:
        return self._nAssignedKernelFunctions * self._nDim

    @property
    def nNodes(self) -> int:
        return self._nAssignedKernelFunctions

    @property
    def visualizationNodes(self):
        return self._nodes

    @property
    def ensightType(self) -> str:
        return "point"

    @property
    def number(self) -> int:
        return self._number

    @property
    def kernelFunctions(self) -> list[BaseMeshfreeKernelFunction]:
        return self._assignedKernelFunctions

    @property
    def dofIndicesPermutation(self):
        return None

    # -------- Kernel function management --------

    def assignKernelFunctions(self, kernelFunctions: list[BaseMeshfreeKernelFunction]):
        """Assign kernel functions to this particle (called by particle manager).

        Parameters
        ----------
        kernelFunctions
            The list of kernel functions in the particle's support.
        """
        self._assignedKernelFunctions = kernelFunctions
        self._nAssignedKernelFunctions = len(kernelFunctions)
        self._nodes = [kf.node for kf in kernelFunctions]
        self._fields = [self._baseFields for _ in self._nodes]
        # Reset VCI corrections
        self._eta = None

    def assignMeshfreeKernelFunctions(self, kernelFunctions: list[BaseMeshfreeKernelFunction]):
        """Assign meshfree kernel functions (base class interface)."""
        self.assignKernelFunctions(kernelFunctions)

    # -------- Geometry --------

    def setProperty(self, propertyName: str, propertyValue: np.ndarray):
        pass

    def setProperties(self, properties: np.ndarray):
        pass

    def getBoundingBox(self) -> np.ndarray:
        return self._vertexCoordinates

    def getVertexCoordinates(self) -> np.ndarray:
        return self._vertexCoordinates

    def getFaceCoordinates(self, faceID: int) -> np.ndarray:
        return self._centerCoordinates

    def getEvaluationCoordinates(self) -> np.ndarray:
        return self._centerCoordinates.reshape(1, -1)

    def getCenterCoordinates(self) -> np.ndarray:
        return self._centerCoordinates

    def getVolumeUndeformed(self) -> float:
        return self._volume

    # -------- State management --------

    def initializeYourself(self):
        pass

    def prepareYourself(self, timeTotal: float, dTime: float):
        self._strain_temp[:] = self._strain
        self._stress_temp[:] = self._stress
        self._displacement_temp[:] = self._displacement

    def acceptStateAndPosition(self):
        self._displacement[:] = self._displacement_temp
        self._strain[:] = self._strain_temp
        self._stress[:] = self._stress_temp

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        if stateType == "stress":
            self._stress[:] = values
            self._stress_temp[:] = values
        elif stateType == "strain":
            self._strain[:] = values
            self._strain_temp[:] = values

    def getResultArray(self, result: str, getPersistentView: bool = True, qp: int = 0) -> np.ndarray:
        if result == "displacement":
            return self._displacement
        elif result == "stress":
            return self._stress
        elif result == "strain":
            return self._strain
        else:
            raise ValueError(f"Unknown result '{result}' requested from PythonParticle.")

    def getRestartData(self) -> np.ndarray:
        return np.concatenate([self._displacement, self._strain, self._stress])

    def readRestartData(self, restartData: np.ndarray):
        self._displacement[:] = restartData[0:2]
        self._strain[:] = restartData[2:6]
        self._stress[:] = restartData[6:10]
        self._displacement_temp[:] = self._displacement
        self._strain_temp[:] = self._strain
        self._stress_temp[:] = self._stress

    # -------- Shape function computation --------

    def _computeShapeFunctionsAndGradients(self):
        """Compute RKPM shape functions and gradients at the particle center.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            N: shape function values (nKernels,)
            dNdx: shape function gradients (nKernels, 2)
        """
        N, dNdx = self._approximation.computeShapeFunctionsAndGradients(
            self._centerCoordinates, self._assignedKernelFunctions
        )

        # Apply VCI correction if available
        if self._eta is not None:
            # eta shape: (nKernels, nDim, nVCIConstraints)
            # For completeness order 0: nVCIConstraints = 0, eta corrections are for gradients only
            # The VCI correction modifies the gradient of the test function
            # dN_I^corrected / dx_j += eta_I_j_C (summed contribution)
            # For the simplest case (nVCIConstraints matches basis size):
            for I in range(self._nAssignedKernelFunctions):
                for j in range(self._nDim):
                    dNdx[I, j] += np.sum(self._eta[I, j, :])

        return N, dNdx

    def _getBMatrix(self, dNdx: np.ndarray) -> np.ndarray:
        """Compute the strain-displacement matrix B.

        Parameters
        ----------
        dNdx
            Shape function gradients (nKernels, 2).

        Returns
        -------
        np.ndarray
            B matrix of shape (4, nDof) for Voigt notation [eps_xx, eps_yy, eps_zz, gamma_xy].
        """
        nK = self._nAssignedKernelFunctions
        nDof = nK * self._nDim
        B = np.zeros((4, nDof))
        for I in range(nK):
            B[0, 2 * I] = dNdx[I, 0]  # eps_xx
            B[1, 2 * I + 1] = dNdx[I, 1]  # eps_yy
            # B[2, :] = 0  # eps_zz (plane strain)
            B[3, 2 * I] = dNdx[I, 1]  # gamma_xy
            B[3, 2 * I + 1] = dNdx[I, 0]  # gamma_xy
        return B

    # -------- Physics kernels --------

    def computePhysicsKernels(
        self,
        dU: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Evaluate residual and stiffness for given displacement increment.

        This method:
        1. Computes RKPM shape functions and gradients
        2. Interpolates displacement increment and computes strain increment
        3. Updates stress using linear elasticity
        4. Assembles internal force P += B^T sigma V and stiffness K += B^T C B V

        Parameters
        ----------
        dU
            The current solution increment (nDof,).
        P
            The internal force vector to be filled (nDof,).
        K
            The stiffness matrix to be filled (nDof*nDof,) in row-major.
        timeTotal
            The current total time.
        dTime
            The time increment.
        """
        nDof = self.nDof
        N, dNdx = self._computeShapeFunctionsAndGradients()
        B = self._getBMatrix(dNdx)

        # Interpolate displacement increment at particle
        dU_particle = np.zeros(2)
        for I in range(self._nAssignedKernelFunctions):
            dU_particle[0] += N[I] * dU[2 * I]
            dU_particle[1] += N[I] * dU[2 * I + 1]

        # Compute strain increment
        dStrain = B @ dU

        # Update state
        self._displacement_temp[:] = self._displacement + dU_particle
        self._strain_temp[:] = self._strain + dStrain
        self._stress_temp[:] = self._C @ self._strain_temp

        # Assemble internal force: P += B^T * sigma * volume
        P += B.T @ self._stress_temp * self._volume

        # Assemble stiffness: K += B^T * C * B * volume
        K_mat = K.reshape(nDof, nDof)
        K_mat += B.T @ self._C @ B * self._volume

    def computeBodyLoad(
        self,
        loadType: str,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute body load contribution (e.g., gravity).

        Parameters
        ----------
        loadType
            The type of load.
        load
            The load vector.
        P
            The external load vector.
        K
            The stiffness matrix (unused for body loads).
        timeTotal
            Current total time.
        dTime
            Time increment.
        """
        N, _ = self._computeShapeFunctionsAndGradients()
        for I in range(self._nAssignedKernelFunctions):
            P[2 * I] -= N[I] * load[0] * self._volume
            P[2 * I + 1] -= N[I] * load[1] * self._volume

    def computeDistributedLoad(
        self,
        loadType: str,
        surfaceID: int,
        load: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute distributed load contribution.

        Parameters
        ----------
        loadType
            The type of load.
        surfaceID
            The surface ID (ignored for point particles).
        load
            The load vector.
        P
            The external load vector.
        K
            The stiffness matrix (unused).
        timeTotal
            Current total time.
        dTime
            Time increment.
        """
        N, _ = self._computeShapeFunctionsAndGradients()
        for I in range(self._nAssignedKernelFunctions):
            P[2 * I] -= N[I] * load[0] * self._volume
            P[2 * I + 1] -= N[I] * load[1] * self._volume

    def computePhysicsKernelsExplicit(
        self,
        dU: np.ndarray,
        P: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Evaluate residual for explicit time integration."""
        N, dNdx = self._computeShapeFunctionsAndGradients()
        B = self._getBMatrix(dNdx)
        P += B.T @ self._stress_temp * self._volume

    def computeBodyLoadExplicit(
        self,
        loadType: str,
        load: np.ndarray,
        P: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute body load for explicit time integration."""
        N, _ = self._computeShapeFunctionsAndGradients()
        for I in range(self._nAssignedKernelFunctions):
            P[2 * I] -= N[I] * load[0] * self._volume
            P[2 * I + 1] -= N[I] * load[1] * self._volume

    def computeDistributedLoadExplicit(
        self,
        loadType: str,
        surfaceID: int,
        load: np.ndarray,
        P: np.ndarray,
        timeTotal: float,
        dTime: float,
    ):
        """Compute distributed load for explicit time integration."""
        self.computeBodyLoadExplicit(loadType, load, P, timeTotal, dTime)

    def computeLumpedInertia(self, M: np.ndarray):
        """Compute lumped mass (not implemented for this teaching example)."""
        pass

    # -------- Interpolation --------

    def getInterpolationVector(self, coordinate: np.ndarray) -> np.ndarray:
        """Get the interpolation vector for a given global coordinate."""
        N = self._approximation.computeShapeFunctionValues(coordinate, self._assignedKernelFunctions)
        return N

    # -------- VCI (Variationally Consistent Integration) --------

    def vci_getNumberOfConstraints(self) -> int:
        """Get the number of VCI constraints.

        For completeness order 0: nVCIConstraints = 1 (just the constant term)
        For completeness order 1: nVCIConstraints = nDim + 1 (constant + linear terms)
        For completeness order 2: nVCIConstraints = (nDim+1)*(nDim+2)/2
        """
        order = self._approximation.completenessOrder
        if order == 0:
            return 1
        elif order == 1:
            return self._nDim + 1
        elif order == 2:
            return (self._nDim + 1) * (self._nDim + 2) // 2
        return 0

    def vci_compute_Test_P_BoundaryIntegral(
        self, R_AiC: np.ndarray, boundarySurfaceVector: np.ndarray, boundaryFaceID: int
    ):
        """Compute the boundary integral contribution for VCI.

        Computes: R_AiC += N_A * P_C(x_p) * n_i * |dGamma|

        For point particles, the boundary surface vector already encodes normal * area.

        Parameters
        ----------
        R_AiC
            The integral array of shape (nKernels, nDim, nVCIConstraints).
        boundarySurfaceVector
            The surface vector (normal * area).
        boundaryFaceID
            The boundary face ID (unused for point particles).
        """
        nVCI = self.vci_getNumberOfConstraints()
        N, _ = self._computeShapeFunctionsAndGradients()

        # Polynomial basis evaluated at the particle's actual coordinates
        xp = self._centerCoordinates
        P = _build_polynomial_basis_2d(xp[0], xp[1], self._approximation.completenessOrder)

        for A in range(self._nAssignedKernelFunctions):
            for i in range(self._nDim):
                for C in range(nVCI):
                    R_AiC[A, i, C] += N[A] * P[C] * boundarySurfaceVector[i]

    def vci_compute_TestGradient_P_Integral(self, R_AiC: np.ndarray):
        """Compute the volume integral of test function gradient times polynomial basis.

        Computes: R_AiC += dN_A/dx_i * P_C(x_p) * V

        Parameters
        ----------
        R_AiC
            The integral array of shape (nKernels, nDim, nVCIConstraints).
        """
        nVCI = self.vci_getNumberOfConstraints()
        _, dNdx = self._approximation.computeShapeFunctionsAndGradients(
            self._centerCoordinates, self._assignedKernelFunctions
        )

        xp = self._centerCoordinates
        P = _build_polynomial_basis_2d(xp[0], xp[1], self._approximation.completenessOrder)

        for A in range(self._nAssignedKernelFunctions):
            for i in range(self._nDim):
                for C in range(nVCI):
                    R_AiC[A, i, C] += dNdx[A, i] * P[C] * self._volume

    def vci_compute_Test_PGradient_Integral(self, R_AiC: np.ndarray):
        """Compute the volume integral of test function times polynomial gradient.

        Computes: R_AiC += N_A * dP_C/dx_i(x_p) * V

        Parameters
        ----------
        R_AiC
            The integral array of shape (nKernels, nDim, nVCIConstraints).
        """
        nVCI = self.vci_getNumberOfConstraints()
        order = self._approximation.completenessOrder

        if order == 0:
            # dP/dx = 0 for order 0 (only constant)
            return

        N, _ = self._computeShapeFunctionsAndGradients()

        # dP/dx at the particle's actual coordinates
        xp = self._centerCoordinates
        dPdx, dPdy = _build_polynomial_basis_gradient_2d(xp[0], xp[1], order)

        for A in range(self._nAssignedKernelFunctions):
            for C in range(nVCI):
                R_AiC[A, 0, C] += N[A] * dPdx[C] * self._volume
                R_AiC[A, 1, C] += N[A] * dPdy[C] * self._volume

    def vci_compute_MMatrix(self, M_ACD: np.ndarray):
        """Compute the M-matrix for VCI.

        Computes: M_ACD += N_A * P_C(x_p) * P_D(x_p) * V

        Parameters
        ----------
        M_ACD
            The M-matrix of shape (nKernels, nVCIConstraints, nVCIConstraints).
        """
        nVCI = self.vci_getNumberOfConstraints()
        N, _ = self._computeShapeFunctionsAndGradients()

        xp = self._centerCoordinates
        P = _build_polynomial_basis_2d(xp[0], xp[1], self._approximation.completenessOrder)

        for A in range(self._nAssignedKernelFunctions):
            for C in range(nVCI):
                for D in range(nVCI):
                    M_ACD[A, C, D] += N[A] * P[C] * P[D] * self._volume

    def vci_assignTestFunctionCorrectionTerms(self, eta_AjC: np.ndarray):
        """Assign the VCI correction terms.

        Parameters
        ----------
        eta_AjC
            The correction terms of shape (nKernels, nDim, nVCIConstraints).
        """
        self._eta = np.copy(eta_AjC)
        self._nVCIConstraints = eta_AjC.shape[2] if eta_AjC.ndim == 3 else 0

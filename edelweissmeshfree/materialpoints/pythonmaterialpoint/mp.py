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
A pure Python material point implementation with linear elastic constitutive behavior.

This material point stores displacement, strain, and stress in Voigt notation
for 2D plane strain problems. The constitutive tangent is computed from
Young's modulus and Poisson's ratio.
"""

import numpy as np

from edelweissmeshfree.materialpoints.base.mp import MaterialPointBase


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
    C[3, 3] = factor * (1.0 - 2.0 * nu) / 2.0  # shear modulus G
    return C


class PythonMaterialPoint(MaterialPointBase):
    """A pure Python material point with linear elastic constitutive behavior.

    Supports 2D plane strain problems with Voigt notation:
    [sigma_xx, sigma_yy, sigma_zz, sigma_xy] for stress,
    [eps_xx, eps_yy, eps_zz, gamma_xy] for strain.

    Parameters
    ----------
    formulation
        The formulation string (e.g., "PlaneStrain").
    number
        The unique material point ID.
    coordinates
        The initial coordinates of the material point.
    volume
        The initial volume.
    material
        A dict with keys "material" (str) and "properties" (array-like: [E, nu]).
    """

    shape = "point"

    def __init__(self, formulation: str, number: int, coordinates: np.ndarray, volume: float, material):
        self._number = number
        self._coordinates = np.asarray(coordinates).flatten()
        self._volume = volume
        self._assignedCells = []

        # State arrays
        self._displacement = np.zeros(2)
        self._displacement_temp = np.zeros(2)
        self._strain = np.zeros(4)  # [eps_xx, eps_yy, eps_zz, gamma_xy]
        self._strain_temp = np.zeros(4)
        self._stress = np.zeros(4)  # [sigma_xx, sigma_yy, sigma_zz, sigma_xy]
        self._stress_temp = np.zeros(4)

        # Strain increment from last interpolation
        self._dStrain = np.zeros(4)

        # Material tangent
        self._C = np.zeros((4, 4))

        # Set material if provided
        if material is not None:
            materialName = material.get("material", "LinearElastic")
            materialProperties = np.asarray(material.get("properties", []))
            self.setMaterial(materialName, materialProperties)

    @property
    def number(self) -> int:
        return self._number

    @property
    def assignedCells(self) -> list:
        return self._assignedCells

    def assignCells(self, cells):
        self._assignedCells = cells

    @property
    def ensightType(self) -> str:
        return self.shape

    def getVertexCoordinates(self) -> np.ndarray:
        return (self._coordinates + self._displacement).reshape(1, -1)

    def getCenterCoordinates(self) -> np.ndarray:
        return self._coordinates + self._displacement

    def getVolume(self) -> float:
        return self._volume

    def interpolateFieldsFromCell(self, dU: np.ndarray, dStrain: np.ndarray):
        """Receive interpolated displacement increment and strain increment from the cell.

        Parameters
        ----------
        dU
            Displacement increment at the material point (2,).
        dStrain
            Strain increment at the material point in Voigt notation (4,).
        """
        self._displacement_temp[:] = self._displacement + dU
        self._dStrain[:] = dStrain

    def computeYourself(self, timeTotal: float, dT: float):
        """Compute material response: update stress from strain increment using linear elasticity."""
        self._strain_temp[:] = self._strain + self._dStrain
        self._stress_temp[:] = self._C @ self._strain_temp

    def acceptStateAndPosition(self):
        """Accept the current state after a converged increment."""
        self._displacement[:] = self._displacement_temp
        self._strain[:] = self._strain_temp
        self._stress[:] = self._stress_temp

    def resetToLastValidStateAndPosition(self):
        """Reset to the last converged state."""
        self._displacement_temp[:] = self._displacement
        self._strain_temp[:] = self._strain
        self._stress_temp[:] = self._stress

    def prepareTimestep(self, timeTotal: float, dT: float):
        """Prepare for a new timestep."""
        self._dStrain[:] = 0.0

    def prepareYourself(self, timeTotal: float, dT: float):
        """Prepare for a new timestep (alias used by solvers)."""
        self.prepareTimestep(timeTotal, dT)

    def getResultArray(self, result: str, getPersistentView: bool = True) -> np.ndarray:
        """Get a result array by name.

        Parameters
        ----------
        result
            One of "displacement", "stress", "strain".
        getPersistentView
            If True, return a view that is continuously updated.

        Returns
        -------
        np.ndarray
            The requested result array.
        """
        if result == "displacement":
            return self._displacement
        elif result == "stress":
            return self._stress
        elif result == "strain":
            return self._strain
        else:
            raise ValueError(f"Unknown result '{result}' requested from PythonMaterialPoint.")

    def getAlgorithmicTangent(self) -> np.ndarray:
        """Return the algorithmic (elastic) tangent matrix.

        Returns
        -------
        np.ndarray
            The (4, 4) material tangent matrix.
        """
        return self._C

    def setProperties(self, propertyName: str, elementProperties: np.ndarray):
        """Assign element-level properties (unused for this implementation)."""
        pass

    def initializeYourself(self):
        """Initialize the material point."""
        pass

    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        """Assign a material and material properties.

        Parameters
        ----------
        materialName
            The material name (e.g., "LinearElastic").
        materialProperties
            Array [E, nu] for linear elastic.
        """
        if materialName.lower().replace(" ", "") == "linearelastic":
            E = materialProperties[0]
            nu = materialProperties[1]
            self._C = _compute_plane_strain_tangent(E, nu)
        else:
            raise ValueError(f"Unknown material '{materialName}' for PythonMaterialPoint.")

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        """Assign initial conditions.

        Parameters
        ----------
        stateType
            The type of initial state (e.g., "stress", "strain").
        values
            The initial state values.
        """
        if stateType == "stress":
            self._stress[:] = values
            self._stress_temp[:] = values
        elif stateType == "strain":
            self._strain[:] = values
            self._strain_temp[:] = values

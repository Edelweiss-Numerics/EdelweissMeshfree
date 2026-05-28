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

"""Minimal concrete MaterialPoint implementation for unit testing."""

import numpy as np

from edelweissmeshfree.materialpoints.base.mp import MaterialPointBase


class MaterialPoint(MaterialPointBase):
    """A minimal concrete implementation of MaterialPointBase for use in unit tests."""

    shape = "point"

    def __init__(self, formulation: str, number: int, coordinates: np.ndarray, volume: float, material):
        """Initialize the test material point."""
        self._number = number
        self._coordinates = coordinates
        self._volume = volume
        self._assignedCells = list()

        self._displacement = np.zeros(2)

    @property
    def number(self) -> int:
        """Return the material point number."""
        return self._number

    @property
    def assignedCells(self) -> list:
        """The list of currently assigned cells."""
        return self._assignedCells

    def assignCells(self, cells: list):
        """Assign the list of cells in which the material point is currently residing."""
        self._assignedCells = cells

    @property
    def ensightType(self) -> str:
        """Return the EnSight geometry type."""
        return self.shape

    def getVertexCoordinates(
        self,
    ) -> np.ndarray:
        """Return the material point vertex coordinates."""
        return np.reshape(self._coordinates + self._displacement, (1, 2))

    def getCenterCoordinates(
        self,
    ) -> np.ndarray:
        """Return the material point center coordinates."""
        return self._coordinates + self._displacement

    def getVolume(
        self,
    ) -> float:
        """Return the material point volume."""
        return self._volume

    # def computeMaterialResponse(self, timeStep: float, timeTotal: float, dT: float):
    #     pass

    def acceptStateAndPosition(
        self,
    ):
        """Accept the current state and position."""
        pass

    def resetToLastValidStateAndPosition(
        self,
    ):
        """Reset the material point to the last accepted state and position."""
        pass

    def getResultArray(self, result: str, getPersistentView: bool = True) -> np.ndarray:
        """Return the requested result array."""
        if result == "displacement":
            return self._displacement

    def setProperties(self, propertyName: str, elementProperties: np.ndarray):
        """Set material point properties."""
        pass

    def initializeMaterialPoint(
        self,
    ):
        """Initialize the material point."""
        pass

    def setMaterial(self, materialName: str, materialProperties: np.ndarray):
        """Assign the material definition."""
        pass

    def setInitialCondition(self, stateType: str, values: np.ndarray):
        """Set an initial condition on the material point."""
        pass

    def addDisplacement(self, dU: np.ndarray):
        """Add a displacement increment to the material point."""
        self._displacement += dU

    def initializeYourself(
        self,
    ):
        """Initialize the internal material point state."""
        pass

    def prepareTimestep(self, timeTotal: float, dT: float):
        """Prepare the material point for a new time step."""
        pass

    def computeYourself(self, timeTotal: float, dT: float):
        """Compute the material point response for the current time step."""
        pass

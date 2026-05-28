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

"""Linear predictor for extrapolating solution fields in MPM time stepping."""

from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.numerics.predictors.basepredictor import BasePredictor


class LinearPredictor(BasePredictor):
    """A linear extrapolator which uses the last solution increment to predict the next solution."""

    def __init__(self, journal: Journal = None, arcLength: bool = False):
        """Initialize the linear predictor.

        Parameters
        ----------
        journal
            The journal used for predictor messages.
        arcLength
            Whether arc-length scaling of the load parameter should be predicted.
        """
        self._dU_n = None
        self._deltaT_n = None
        self._arcLength = arcLength
        self._journal = journal
        if self._arcLength:
            self._dLambda_n = None

    def resetHistory(
        self,
    ):
        """Reset the stored increment history used for extrapolation."""
        self._dU_n = None
        self._deltaT_n = None
        if self._arcLength:
            self._dLambda_n = None

    def getPrediction(self, timeStep: TimeStep):
        """Compute a linear prediction for the next increment.

        Parameters
        ----------
        timeStep
            The time step for which the prediction is requested.

        Returns
        -------
        DofVector or tuple[DofVector, float] or None
            The predicted solution increment, optionally with the arc-length increment, or ``None`` when no history is available.
        """
        if self._dU_n is None:
            if self._arcLength:
                return None, None
            else:
                return None
        if timeStep.timeIncrement < 1e-15 or self._deltaT_n < 1e-15:
            if self._arcLength:
                return None, None
            else:
                return None
        else:
            dU = self._dU_n * (timeStep.timeIncrement / self._deltaT_n)
            if self._arcLength:
                dLambda = self._dLambda_n * (timeStep.timeIncrement / self._deltaT_n)
                return dU, dLambda
            else:
                return dU

    def updateHistory(self, dU: DofVector, timeStep: TimeStep, dLambda: float = None):
        """Store the latest converged increment for subsequent predictions.

        Parameters
        ----------
        dU
            The converged solution increment.
        timeStep
            The time step associated with the increment.
        dLambda
            The converged arc-length increment when arc-length control is enabled.
        """
        self._dU_n = dU.copy()
        self._deltaT_n = timeStep.timeIncrement
        if self._arcLength:
            self._dLambda_n = dLambda.copy()

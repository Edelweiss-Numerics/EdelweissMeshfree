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

"""Quadratic predictor for extrapolating solution fields in MPM time stepping."""

from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.numerics.predictors.basepredictor import BasePredictor
from edelweissmeshfree.numerics.predictors.linearpredictor import LinearPredictor


class QuadraticPredictor(BasePredictor):
    """A quadratic extrapolator which uses the last two solution increments to predict the next solution."""

    def __init__(self, journal: Journal = None, arcLength: bool = False):
        """Initialize the quadratic predictor.

        Parameters
        ----------
        journal
            The journal used for predictor messages.
        arcLength
            Whether arc-length scaling of the load parameter should be predicted.
        """
        self._dU_n = None
        self._dU_n_minus_1 = None
        self._deltaT_n = None
        self._deltaT_n_minus_1 = None
        self._arcLength = arcLength
        self._journal = journal
        if self._arcLength:
            self._dLambda_n = None
            self._dLambda_n_minus_1 = None

    def resetHistory(
        self,
    ):
        """Reset the stored increment history used for quadratic extrapolation."""
        self._dU_n = None
        self._dU_n_minus_1 = None
        self._deltaT_n = None
        self._deltaT_n_minus_1 = None
        if self._arcLength:
            self._dLambda_n = None
            self._dLambda_n_minus_1 = None

    def getPrediction(self, timeStep: TimeStep):
        """Compute a quadratic prediction for the next increment.

        Parameters
        ----------
        timeStep
            The time step for which the prediction is requested.

        Returns
        -------
        DofVector or tuple[DofVector, float] or None
            The predicted solution increment, optionally with the arc-length increment, or ``None`` when insufficient history is available.
        """
        if self._dU_n is None:
            if self._arcLength:
                return None, None
            else:
                return None
        if self._deltaT_n < 1e-15:
            if self._arcLength:
                return None, None
            else:
                return None

        if self._dU_n_minus_1 is None or self._deltaT_n_minus_1 < 1e-15:
            if self._journal is not None:
                self._journal.message(
                    "Only one time step in history, falling back to linear predictor.", "QuadraticPredictor", 1
                )
            if self._arcLength:
                linearPredictor = LinearPredictor(arcLength=True)
                linearPredictor._dLambda_n = self._dLambda_n
            else:
                linearPredictor = LinearPredictor()
            linearPredictor._dU_n = self._dU_n
            linearPredictor._deltaT_n = self._deltaT_n
            return linearPredictor.getPrediction(timeStep)

        if self._journal is not None:
            self._journal.message("Computing quadratic prediction", "QuadraticPredictor", 1)
        v_n = self._dU_n / self._deltaT_n
        v_n_minus_1 = self._dU_n_minus_1 / self._deltaT_n_minus_1
        a_n = (v_n - v_n_minus_1) / ((self._deltaT_n + self._deltaT_n_minus_1) / 2)

        dU = v_n * timeStep.timeIncrement + 0.5 * a_n * timeStep.timeIncrement**2

        if self._arcLength:
            vl_n = self._dLambda_n / self._deltaT_n
            vl_n_minus_1 = self._dLambda_n_minus_1 / self._deltaT_n_minus_1
            al_n = (vl_n - vl_n_minus_1) / ((self._deltaT_n + self._deltaT_n_minus_1) / 2)

            dLambda = vl_n * timeStep.timeIncrement + 0.5 * al_n * timeStep.timeIncrement**2

            return dU, dLambda
        else:
            return dU

    def updateHistory(self, dU: DofVector, timeStep: TimeStep, dLambda: float = None):
        """Store the latest converged increments for subsequent predictions.

        Parameters
        ----------
        dU
            The converged solution increment.
        timeStep
            The time step associated with the increment.
        dLambda
            The converged arc-length increment when arc-length control is enabled.
        """
        self._dU_n_minus_1 = self._dU_n
        self._deltaT_n_minus_1 = self._deltaT_n

        self._dU_n = dU.copy()
        self._deltaT_n = timeStep.timeIncrement
        if self._arcLength:
            self._dLambda_n_minus_1 = self._dLambda_n
            self._dLambda_n = dLambda.copy()

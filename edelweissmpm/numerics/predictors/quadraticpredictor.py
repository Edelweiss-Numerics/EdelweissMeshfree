# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  Computational Mechanics of Materials, Institute of Structural Engineering,
#  BOKU University, Vienna
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#  Thomas Mader thomas.mader@boku.ac.at
#
#  This file is part of EdelweissMPM.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMPM.
#  ---------------------------------------------------------------------

from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.dofmanager import DofVector
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmpm.numerics.predictors.basepredictor import BasePredictor
from edelweissmpm.numerics.predictors.linearpredictor import LinearPredictor


class QuadraticPredictor(BasePredictor):
    """A quadratic extrapolator which uses the last two solution increments to predict the next solution."""

    def __init__(self, journal: Journal = None, arcLength: bool = False):
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
        self._dU_n = None
        self._dU_n_minus_1 = None
        self._deltaT_n = None
        self._deltaT_n_minus_1 = None
        if self._arcLength:
            self._dLambda_n = None
            self._dLambda_n_minus_1 = None

    def getPrediction(self, timeStep: TimeStep):

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
        self._dU_n_minus_1 = self._dU_n
        self._deltaT_n_minus_1 = self._deltaT_n

        self._dU_n = dU.copy()
        self._deltaT_n = timeStep.timeIncrement
        if self._arcLength:
            self._dLambda_n_minus_1 = self._dLambda_n
            self._dLambda_n = dLambda.copy()

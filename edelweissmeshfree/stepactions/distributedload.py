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
from collections.abc import Callable

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.sets.materialpointset import MaterialPointSet
from edelweissmeshfree.stepactions.base.mpmdistributedloadbase import (
    MPMDistributedLoadBase,
)


class MaterialPointPointWiseDistributedLoad(MPMDistributedLoadBase):
    """
    This is a simple distributed load for material points which consist of a single vertex.
    Accordingly, the load vector already includes the surface vector defintion.

    """

    #: The load accumulated by the steps before, this step's increment, and whether it is idle.
    checkpointedState = {"_loadAtStepStart": np.ndarray, "_delta": np.ndarray, "_idle": bool}

    def __init__(
        self,
        name: str,
        model: MPMModel,
        journal: Journal,
        materialPoints: MaterialPointSet,
        distributedLoadType: str,
        loadVector: np.ndarray,
        f_t: Callable[[float], float] = None,
    ):
        self.name = name

        self._loadVector = loadVector
        self._loadAtStepStart = np.zeros_like(self._loadVector)
        self._loadType = distributedLoadType
        self._materialPoints = materialPoints

        if len(self._loadVector) < model.domainSize:
            raise Exception("BodyForce {:}: load vector has wrong dimension!".format(self.name))

        self._delta = self._loadVector
        if f_t is not None:
            self._amplitude = f_t
        else:
            self._amplitude = lambda x: x

        self._idle = False

    @property
    def mpSet(self) -> MaterialPointSet:
        return self._materialPoints

    @property
    def loadType(self) -> str:
        return self._loadType

    def applyAtStepEnd(self, model, loadParameter: float | None = None):
        """Keep the value the load reached at the end of the step, from which the next step starts; see
        :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.applyAtStepEnd`.

        Parameters
        ----------
        model
            The current state of the model.
        loadParameter
            The load parameter an arc length step ended at; None for a step that follows the amplitude.
        """
        if not self._idle:
            self._loadAtStepStart += self._delta * self.amplitudeAtStepEnd(self._amplitude, loadParameter)

            self._delta = 0
            self._idle = True

    def getCurrentMaterialPointLoad(self, materialPoint, timeStep: TimeStep) -> tuple[int, np.ndarray]:
        if self._idle:
            t = 1.0
        else:
            t = timeStep.stepProgress

        loadVec = self._loadAtStepStart + self._delta * self._amplitude(t)

        return 0, loadVec

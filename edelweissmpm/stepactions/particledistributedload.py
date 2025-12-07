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
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
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
import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.particles.base.baseparticle import BaseParticle
from edelweissmpm.sets.particleset import ParticleSet


class ParticleDistributedLoad:
    """
    This is a distributed load for particles.

    Parameters
    ----------
    name : str
        Name of the distributed load.
    model : MPMModel
        The MPM model tree.
    journal : Journal
        The journal to write messages to.
    particleSurface : EntityBasedSurface
        The surface defining the particles to which the load is applied.
    distributedLoadType : str
        The type of the distributed load, e.g., "pressure".
    loadVector : np.ndarray
        The load vector to apply to the particles.
    **kwargs
        Additional keyword arguments. The following are supported:
        - f_t : Callable[[float], float]
            The amplitude function of the distributed load.
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        journal: Journal,
        particleSurface: EntityBasedSurface,
        distributedLoadType: str,
        loadVector: np.ndarray,
        **kwargs,
    ):
        self.name = name

        self._loadVector = loadVector
        self._loadAtStepStart = np.zeros_like(self._loadVector)
        self._loadType = distributedLoadType
        self._particleSurface = particleSurface

        self._delta = self._loadVector
        if "f_t" in kwargs:
            self._amplitude = kwargs["f_t"]
        else:
            self._amplitude = lambda x: x

        self._idle = False

    @property
    def particles(self) -> ParticleSet:
        return self._particles

    @property
    def loadType(self) -> str:
        return self._loadType

    def applyAtStepEnd(self, model, stepMagnitude=None):
        if not self._idle:
            if stepMagnitude is None:
                self._loadAtStepStart += self._delta * self._amplitude(1.0)
            else:
                self._loadAtStepStart += self._delta * stepMagnitude

            self._delta = 0
            self._idle = True

    def getCurrentParticleLoads(self, timeStep: TimeStep) -> list[tuple[BaseParticle, int, np.ndarray]]:
        """
        Get the current particle loads for this distributed load.

        Parameters
        ----------
        timeStep : TimeStep
            The current time step.

        Returns
        -------
        list[ tuple[ BaseParticle,int, np.ndarray]]
            A list of tuples containing the particle, surface ID, and load vector.
        """

        if self._idle:
            t = 1.0
        else:
            t = timeStep.stepProgress

        loadVec = self._loadAtStepStart + self._delta * self._amplitude(t)

        particleLoads = list()
        for surfaceID, pSet in self._particleSurface.items():

            particleLoads += [(p, surfaceID, loadVec) for p in pSet]

        return particleLoads

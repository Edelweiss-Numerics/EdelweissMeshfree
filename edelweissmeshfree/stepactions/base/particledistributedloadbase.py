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
"""Distributed loads on the faces of particles."""

from abc import abstractmethod

import numpy as np
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.stepactions.base.loadbase import LoadBase
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticleDistributedLoadBase(LoadBase):
    """Base class for distributed (surface) loads on the faces of particles: each loaded particle integrates the load
    over its face, see :meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.computeDistributedLoad`.

    It is an EdelweissFE load (:class:`~edelweissfe.stepactions.base.loadbase.LoadBase`): it assembles its own residual
    contribution, the negative of the external load, in :meth:`assembleLoad`. A load deriving from it says which kind
    of load it is (:attr:`loadType`), and how large it is on which particle face (:meth:`getCurrentParticleLoads`).
    """

    @property
    @abstractmethod
    def loadType(self) -> str:
        """The type of the load, one the particles support (e.g., pressure).

        Returns
        -------
        str
            The load type.
        """

    @abstractmethod
    def getCurrentParticleLoads(self, timeStep: TimeStep) -> list[tuple[BaseParticle, int, np.ndarray]]:
        """The current load on each loaded particle face.

        Parameters
        ----------
        timeStep
            The current time step.

        Returns
        -------
        list[tuple[BaseParticle, int, np.ndarray]]
            A tuple per loaded particle face: the particle, the face ID, and the load vector.
        """

    def assembleLoad(
        self,
        P: DofVector,
        K,
        U_np: DofVector,
        dU: DofVector,
        timeStep: TimeStep,
        dofManager: DofManager,
    ):
        """Let every loaded particle add the residual contribution of the load on its face, the negative of the
        external load, and its tangent; see :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.assembleLoad`.

        Parameters
        ----------
        P
            The global residual contribution vector to augment.
        K
            The system matrix to augment, which hands out the block of a particle by ``K[particle]``; None for an
            explicit solver, which needs no tangent.
        U_np
            The current global solution.
        dU
            The global solution increment.
        timeStep
            The current time step.
        dofManager
            The DOF manager of the discrete system.
        """

        for p, surfaceID, loadVector in self.getCurrentParticleLoads(timeStep):
            Pc = np.zeros(p.nDof)
            Kc = K[p] if K is not None else np.zeros(p.nDof * p.nDof)
            p.computeDistributedLoad(
                self.loadType,
                surfaceID,
                loadVector,
                Pc,
                Kc,
                timeStep.totalTime,
                timeStep.timeIncrement,
            )

            P[p] += Pc

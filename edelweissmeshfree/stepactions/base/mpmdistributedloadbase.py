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

from abc import abstractmethod

import numpy as np
from edelweissfe.numerics.dofmanager import DofManager, DofVector
from edelweissfe.stepactions.base.loadbase import LoadBase
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.materialpoints.base.mp import MaterialPointBase
from edelweissmeshfree.sets.materialpointset import MaterialPointSet


class MPMDistributedLoadBase(LoadBase):
    """Base class for distributed (surface) loads carried by material points: each cell a loaded material point lies
    in integrates the load. It is an EdelweissFE load (:class:`~edelweissfe.stepactions.base.loadbase.LoadBase`): it
    assembles its own residual contribution, the negative of the external load, in :meth:`assembleLoad`."""

    @property
    @abstractmethod
    def mpSet(self) -> MaterialPointSet:
        """The material points this load is acting on.

        Returns
        -------
        MaterialPointSet
            The list of cells.
        """

    @property
    @abstractmethod
    def loadType(self) -> str:
        """Return the type of load (e.g., pressure, ... )

        Returns
        -------
        str
            The load type.
        """

    @abstractmethod
    def getCurrentMaterialPointLoad(self, materialPoint: MaterialPointBase, timeStep: TimeStep) -> [int, np.ndarray]:
        """Return the current load vector for a given material point.

        Parameters
        ----------
        materialPoint
            The material point for which the load definition is requrested.
        timeStep
            The current time step.

        Returns
        -------
        tuple[int, np.ndarray]
            The tuple consisting of:
                - the surface ID.
                - the load vector.
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
        """Let the cells of every loaded material point add the residual contribution of the load the material point
        carries, the negative of the external load, and its tangent; see
        :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.assembleLoad`.

        Parameters
        ----------
        P
            The global residual contribution vector to augment.
        K
            The system matrix to augment, which hands out the block of a cell by ``K[cell]``; None for an explicit
            solver, which needs no tangent.
        U_np
            The current global solution.
        dU
            The global solution increment.
        timeStep
            The current time step.
        dofManager
            The DOF manager of the discrete system.
        """

        for mp in self.mpSet:
            surfaceID, loadVector = self.getCurrentMaterialPointLoad(mp, timeStep)

            for cl in mp.assignedCells:
                Pc = np.zeros(cl.nDof)
                Kc = K[cl] if K is not None else np.zeros(cl.nDof * cl.nDof)
                cl.computeDistributedLoad(
                    self.loadType,
                    surfaceID,
                    mp,
                    loadVector,
                    Pc,
                    Kc,
                    timeStep.totalTime,
                    timeStep.timeIncrement,
                )
                P[cl] += Pc

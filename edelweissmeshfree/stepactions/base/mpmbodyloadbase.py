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
"""Body loads are loads that act on the entire body of the model."""

from abc import abstractmethod

import numpy as np
from edelweissfe.numerics.doflocations import DofLocations
from edelweissfe.numerics.dofvector import DofVector
from edelweissfe.stepactions.base.loadbase import LoadBase
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.sets.cellset import CellSet


class MPMBodyLoadBase(LoadBase):
    """Base class for body loads on the cells of the background grid, such as gravity.

    It is an EdelweissFE load (:class:`~edelweissfe.stepactions.base.loadbase.LoadBase`): it assembles its own
    residual contribution, the negative of the external load, in :meth:`assembleLoad`. If you want to implement a new
    body load, you should derive from this class.
    """

    @property
    @abstractmethod
    def cellSet(self) -> CellSet:
        """The cells this load is acting on.

        Returns
        -------
        CellSet
            The list of cells.
        """

    @property
    @abstractmethod
    def loadType(self) -> str:
        """Return the type of load (e.g., bodyforce, ... )

        Returns
        -------
        str
            The load type.
        """

    @abstractmethod
    def getCurrentLoad(self, timeStep: TimeStep) -> np.ndarray:
        """Return the current magnitude for this body load.

        Parameters
        ----------
        timeStep
            The current time step.

        Returns
        -------
        np.ndarray
            The magnitude.
        """

    def assembleLoad(
        self,
        P: DofVector,
        K,
        U_np: DofVector,
        dU: DofVector,
        timeStep: TimeStep,
        locations: DofLocations,
    ):
        """Let every active cell of the cell set add the residual contribution of the body load, the negative of the
        external load, and its tangent; see :meth:`~edelweissfe.stepactions.base.loadbase.LoadBase.assembleLoad`.

        The active cells are those of the discrete system, i.e. those the locations hold; a body load acts only
        on them.

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
        locations
            The locations of the entities in the discrete system, from which the load finds where it acts.
        """

        loadVector = self.getCurrentLoad(timeStep)

        # summed in the order of the cell set
        for cl in self.cellSet:
            if cl in locations:
                Pc = np.zeros(cl.nDof)
                Kc = K[cl] if K is not None else np.zeros(cl.nDof * cl.nDof)

                cl.computeBodyLoad(self.loadType, loadVector, Pc, Kc, timeStep.totalTime, timeStep.timeIncrement)

                P[cl] += Pc

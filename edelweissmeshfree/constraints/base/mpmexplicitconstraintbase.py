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
"""Constraints for explicit analyses only: they act with forces, and have no tangent; they give the bound of their
stiffness for the critical time step instead."""

from abc import abstractmethod

import numpy as np
from edelweissfe.solvers.base.criticaltimestep import NextIncrement
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase


class MPMExplicitConstraintBase(MPMConstraintBase):
    """The base class of the constraints that act only in an explicit analysis.

    An explicit analysis solves no linear system, so it needs a constraint's forces, and the bound of their stiffness
    for the critical time step, but not their tangent. A constraint written for it alone -- a penalty contact evaluated
    on the particles' current positions, say -- implements :meth:`computeKernelsExplicit`, EdelweissFE's entry point for
    evaluating a constraint without a tangent, and :meth:`addBoundOfAbsoluteStiffnessTimes`, and has no
    :meth:`computeKernels`: an implicit solver that is handed such a constraint stops with an error rather than solve
    with a tangent that is missing.

    It follows the same sign rule as every other constraint: it adds the *negative* of the force it exerts on its
    nodes to ``P``.
    """

    #: Written for an explicit analysis, it acts there; see
    #: :attr:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.actsInExplicitAnalysis`.
    actsInExplicitAnalysis = True

    def computeKernels(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        P: np.ndarray,
        K: np.ndarray,
        timeStep: TimeStep,
    ):
        """Refuse an implicit analysis: this constraint has no tangent.

        Parameters
        ----------
        U_np
            The current solution, restricted to this constraint's degrees of freedom.
        dU
            The solution increment, likewise restricted.
        P
            The residual contribution.
        K
            The tangent.
        timeStep
            The current time step.

        Raises
        ------
        TypeError
            Always.
        """

        raise TypeError(
            "Constraint '{:}' ({:}) acts only in an explicit analysis: it has no tangent, so an implicit solver "
            "cannot use it.".format(self.name, type(self).__name__)
        )

    @abstractmethod
    def computeKernelsExplicit(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        P: np.ndarray,
        timeStep: TimeStep,
    ):
        """Add this constraint's residual contribution to ``P``: the negative of the force it exerts on its nodes.

        Parameters
        ----------
        U_np
            The current solution, restricted to this constraint's degrees of freedom.
        dU
            The solution increment, likewise restricted.
        P
            The residual contribution to augment.
        timeStep
            The current time step.
        """

    @abstractmethod
    def addBoundOfAbsoluteStiffnessTimes(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        weights: np.ndarray,
        bound: np.ndarray,
        timeStep: TimeStep,
        nextIncrement: NextIncrement | None = None,
    ):
        """Add an upper bound of the row sums of the absolute value of this constraint's stiffness, weighted by column,
        :math:`b_i \\mathrel{+}= \\sum_j |K_{ij}|\\, w_j`, at the state of its force, or over the states the next
        increment can reach if it is given, for the critical time step of ``NED``: see
        :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`. The default
        there bounds the tangent of :meth:`computeKernels`, which this constraint does not have, so it gives the bound
        itself, in one of two ways:

        * the bound of its springs, taken from the same springs as its force: through
          :func:`~edelweissmeshfree.constraints.base.pointsofparticles.addBoundOfAbsoluteStiffnessOfPointsOfParticle`,
          or by deriving from
          :class:`~edelweissmeshfree.constraints.base.penaltyspringsexplicit.PenaltySpringsOnPointsOfOneParticleExplicit`,
          which does both;
        * nothing, saying so in its documentation, if its stiffness is to stay outside the critical time step.

        Parameters
        ----------
        U_np
            The current solution, restricted to this constraint's degrees of freedom.
        dU
            The solution increment, likewise restricted.
        weights
            The weight of each column, likewise restricted.
        bound
            The row sums to augment, one per degree of freedom of this constraint.
        timeStep
            The time step of the state, as for the force.
        nextIncrement
            The next increment, restricted to this constraint's degrees of freedom; None for the state alone.
        """

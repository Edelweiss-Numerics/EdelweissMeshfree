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
"""The penalty contact of points of one particle with an obstacle, in an explicit analysis: the force, the bound of the
stiffness for the critical time step and the gaps over the next increment, written once; a derived class gives the
physics, the gap of a point and its gradient."""

from abc import abstractmethod

import numpy as np
from edelweissfe.constraints.base.constraintbase import NextIncrement
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import rowDot

from edelweissmeshfree.constraints.base.mpmexplicitconstraintbase import (
    MPMExplicitConstraintBase,
)
from edelweissmeshfree.constraints.base.pointsofparticles import (
    SpringsOnPointsOfParticle,
    addBoundOfAbsoluteStiffnessOfPointsOfParticle,
    addForcesOfSpringsOnPointsOfParticle,
)

#: Below this norm, the gradient of a gap has no direction: the point gets no spring.
GRADIENT_WITHOUT_DIRECTION = 1e-14


class PenaltySpringsOnPointsOfOneParticleExplicit(MPMExplicitConstraintBase):
    """The penalty contact of points of one particle -- its center, vertices or face centers -- with an obstacle, in an
    explicit analysis. Every point that penetrates the obstacle gets a spring of the penalty stiffness :math:`k`
    (:class:`~edelweissmeshfree.constraints.base.pointsofparticles.SpringsOnPointsOfParticle`): along the unit direction
    :math:`\\boldsymbol{n}_p` of the gradient of its gap :math:`g_p`, the outward normal of the obstacle, with the force
    :math:`f_p = -k\\, g_p > 0`, which pushes the point out. The force (:meth:`computeKernelsExplicit`) and the bound of
    the stiffness for the critical time step (:meth:`addBoundOfAbsoluteStiffnessTimes`) are taken from the same springs
    (:meth:`_springs`), over the next increment with the smallest gap of each point within it
    (:meth:`_gapsOverTheNextIncrement`).

    A derived class is the physics and a constructor: the gap of a point at a time step (:meth:`_gapsOfPoints`), negative
    where it penetrates, and its gradient (:meth:`_gradientsOfTheGaps`). Its constructor sets the attributes below, and
    its connectivity update keeps the points and the nodes of the particle's kernel functions current.
    """

    #: The particle whose points the springs act on.
    _particle = None
    #: Where the nodes of the particle's kernel functions sit in the block of the constraint.
    _localNodes = np.zeros(0, dtype=np.intp)
    #: The size of the field the springs act on, the displacement.
    _fieldSize = 0
    #: The penalty stiffness :math:`k` of every spring.
    _penaltyParameter = 0.0
    #: The points of the particle the springs act on, where they are now.
    _constrained_points = ()
    #: The sum of the forces of the springs in the last evaluation, :math:`\\sum_p f_p`.
    reactionForce = 0.0

    @abstractmethod
    def _gapsOfPoints(self, points: np.ndarray, timeStep: TimeStep) -> np.ndarray:
        """The gap of each point to the obstacle where it is at the end of a time step: negative where the point
        penetrates.

        Parameters
        ----------
        points
            The points, one row each.
        timeStep
            The time step, for an obstacle that moves.

        Returns
        -------
        np.ndarray
            The gaps.
        """

    @abstractmethod
    def _gradientsOfTheGaps(self, points: np.ndarray) -> np.ndarray:
        """The gradient of the gap of each point with respect to its position, which points out of the obstacle.

        Parameters
        ----------
        points
            The points, one row each.

        Returns
        -------
        np.ndarray
            The gradients, one row each.
        """

    def _gapsOverTheNextIncrement(
        self, points: np.ndarray, timeStep: TimeStep, nextIncrement: NextIncrement
    ) -> np.ndarray:
        """The smallest gap each point may have within the next increment
        (:meth:`~edelweissfe.constraints.base.constraintbase.NextIncrement.smallestGaps`). The argument there needs a gap
        affine in the motion, with the normals held, as within an increment: the gap is linearized about where the
        point is now, :math:`g_p + \\nabla g_p \\cdot \\Delta\\boldsymbol{x}_p`, with the motion of the point
        :math:`\\Delta\\boldsymbol{x}_p = \\mathbf{I}_p \\Delta\\boldsymbol{U}`
        (:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.motionOfPointsOfParticle`), and
        the obstacle where it is at the time step of the state. For a plane it is the gap; for a convex obstacle the
        signed distance is convex, its linearization a lower bound of it, and the smallest gap of the linearization a
        lower bound of the smallest one within the increment, also where a straight motion passes the obstacle and the
        true gap is smallest inside the increment.

        Parameters
        ----------
        points
            The points now, one row each.
        timeStep
            The time step now.
        nextIncrement
            The next increment the solver may take, restricted to the constraint.

        Returns
        -------
        np.ndarray
            The smallest gaps.
        """

        gradients = self._gradientsOfTheGaps(points)

        def gapsAt(U, dU, timeStep):
            motion = self.motionOfPointsOfParticle(dU, self._particle, points, self._localNodes, self._fieldSize)
            return self._gapsOfPoints(points, timeStep) + rowDot(gradients, motion)

        zero = np.zeros(self.nDof)
        return nextIncrement.smallestGaps(gapsAt, zero, zero, timeStep)

    def _springsAt(self, points: np.ndarray, gaps: np.ndarray) -> SpringsOnPointsOfParticle:
        """The springs of the points at the gaps given: one of the penalty stiffness at every point with a negative gap,
        along the unit direction of the gradient of its gap (none where the gradient has no direction, below
        :data:`GRADIENT_WITHOUT_DIRECTION`), with the force :math:`f_p = -k\\, g_p` at that gap. With the gaps now, the
        forces on the points; with the smallest gaps over the next increment, those they could reach there -- the bound
        reads the stiffness only.

        Parameters
        ----------
        points
            The points, one row each.
        gaps
            Their gaps.

        Returns
        -------
        SpringsOnPointsOfParticle
            The springs; none if no point penetrates.
        """

        closed = gaps < 0.0
        directions = self._gradientsOfTheGaps(points[closed]).reshape((-1, self._fieldSize))
        norms = np.array([np.linalg.norm(direction) for direction in directions])
        hasDirection = norms > GRADIENT_WITHOUT_DIRECTION
        directions[hasDirection] /= norms[hasDirection, None]
        directions[~hasDirection] = 0.0
        return SpringsOnPointsOfParticle(
            # the vector the force is distributed with (see interpolationVectorAtPointOf)
            np.array([self.interpolationVectorAtPointOf(self._particle, p).flatten() for p in points[closed]]).reshape(
                (len(directions), len(self._localNodes))
            ),
            directions,
            -self._penaltyParameter * gaps[closed],
            np.full(len(directions), self._penaltyParameter),
        )

    def _springs(self, timeStep: TimeStep, nextIncrement: NextIncrement | None = None) -> SpringsOnPointsOfParticle:
        """The penalty springs at the particle's current position (:meth:`_springsAt`), at the gaps now, or, given the
        next increment, at the smallest gaps within it (:meth:`_gapsOverTheNextIncrement`): what the force and the bound
        of the stiffness are made from.

        Parameters
        ----------
        timeStep
            The current time step, for an obstacle that moves.
        nextIncrement
            The next increment, for the bound over it; None for the state now.

        Returns
        -------
        SpringsOnPointsOfParticle
            The springs.
        """

        points = np.array(self._constrained_points, dtype=float).reshape((len(self._constrained_points), -1))
        if nextIncrement is None:
            gaps = self._gapsOfPoints(points, timeStep)
        else:
            gaps = self._gapsOverTheNextIncrement(points, timeStep, nextIncrement)
        return self._springsAt(points, gaps)

    def computeKernelsExplicit(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, timeStep: TimeStep):
        """Add the residual contribution of the penalty forces: the negative of the forces with which the obstacle pushes
        the points of the particle out, at its current position (:meth:`_springs`); :attr:`reactionForce` is their sum.

        Parameters
        ----------
        U_np
            The current solution; unused, the points are where the particle is now.
        dU
            The solution increment; unused.
        P
            The residual contribution to augment.
        timeStep
            The current time step.
        """

        springs = self._springs(timeStep)
        addForcesOfSpringsOnPointsOfParticle(P.reshape(-1, self._fieldSize), self._localNodes, springs)
        self.reactionForce = springs.forces.sum()

    def addBoundOfAbsoluteStiffnessTimes(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        weights: np.ndarray,
        bound: np.ndarray,
        timeStep: TimeStep,
        nextIncrement: NextIncrement | None = None,
    ):
        """Add the row sums of the absolute value of the stiffness of the penalty springs (:meth:`_springs`), weighted
        by column (see
        :func:`~edelweissmeshfree.constraints.base.pointsofparticles.addBoundOfAbsoluteStiffnessOfPointsOfParticle`),
        over the next increment if it is given: a point that could close within it has its spring.

        Parameters
        ----------
        U_np
            The current solution; unused.
        dU
            The solution increment; unused.
        weights
            The weight of each column.
        bound
            The row sums to augment.
        timeStep
            The current time step.
        nextIncrement
            The next increment the solver may take; None to bound the state now alone.
        """

        addBoundOfAbsoluteStiffnessOfPointsOfParticle(
            bound, weights, self._localNodes, self._springs(timeStep, nextIncrement)
        )

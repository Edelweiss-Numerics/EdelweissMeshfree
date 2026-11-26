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
"""The penalty contact of points of one particle with an obstacle, in an explicit analysis: the force and the bound of
the stiffness for the critical time step, the points and the connectivity, written once; a derived class gives the
physics, the gap of a point, its gradient and the reach of the obstacle."""

from abc import abstractmethod

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmexplicitconstraintbase import (
    MPMExplicitConstraintBase,
)
from edelweissmeshfree.constraints.base.pointsofparticles import (
    SpringsOnPointsOfParticle,
    addBoundOfAbsoluteStiffnessOfPointsOfParticle,
    addForcesOfSpringsOnPointsOfParticle,
    locateTheNodesOfParticles,
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
    (:meth:`_springs`).

    The points, the nodes of the particle's kernel functions and whether the obstacle is within reach are kept current
    here (:meth:`updateConnectivity`). A derived class is the physics and a constructor: the gap of a point at a time
    step (:meth:`_gapsOfPoints`), negative where it penetrates, its gradient (:meth:`_gradientsOfTheGaps`), and whether
    the points are close enough to the obstacle to take part (:meth:`_isWithinReach`); its constructor sets what the
    obstacle needs and calls this one.

    Parameters
    ----------
    name
        The name of the constraint.
    particle
        The particle whose points the springs act on.
    model
        The model.
    location
        Where on the particle the points are: ``"center"``, ``"face"`` (the centers of the faces ``faceIDs``) or
        ``"vertex"`` (the vertices ``vertexIDs``).
    faceIDs
        The faces, one or a list, if ``location`` is ``"face"``.
    vertexIDs
        The vertices, one or a list, if ``location`` is ``"vertex"``.
    penaltyParameter
        The penalty stiffness :math:`k` of every spring.
    doProximityCheck
        Whether the constraint takes part only while the obstacle is within reach (:meth:`_isWithinReach`).
    proximityFactor
        The reach, in multiples of the edge of a cube of the particle's undeformed volume.
    interpolationLinearizedAboutCenter
        The vector of a point: the shape functions at it, N (False), or the particle's interpolation linearized about
        its center, G (True); see
        :meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`.

    Raises
    ------
    ValueError
        If ``location`` is none of the three, if the ids its points need are missing, or if it gives no point.
    """

    #: Carries nothing from one increment to the next.
    checkpointedState = {}

    def __init__(
        self,
        name: str,
        particle,
        model,
        location: str,
        faceIDs: list[int] | int | None,
        vertexIDs: list[int] | int | None,
        penaltyParameter: float,
        doProximityCheck: bool,
        proximityFactor: float,
        interpolationLinearizedAboutCenter: bool,
    ):
        self._name = name
        self.interpolationLinearizedAboutCenter = interpolationLinearizedAboutCenter
        self._field = "displacement"
        #: The size of the field the springs act on, the displacement.
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        #: The particle whose points the springs act on.
        self._particle = particle
        #: The nodes of the particle's kernel functions, and where they sit in the block of the constraint (see
        #: :meth:`updateConnectivity`).
        self._nodes = []
        self._localNodes = np.zeros(0, dtype=int)
        #: The penalty stiffness :math:`k` of every spring.
        self._penaltyParameter = penaltyParameter
        #: The sum of the forces of the springs in the last evaluation, :math:`\\sum_p f_p`.
        self.reactionForce = 0.0

        if location not in ("center", "face", "vertex"):
            raise ValueError(
                "Constraint '{:}': location must be 'center', 'face' or 'vertex', not '{:}'.".format(name, location)
            )
        if location == "face" and faceIDs is None:
            raise ValueError("faceID must be specified when location is 'face'.")
        if location == "vertex" and vertexIDs is None:
            raise ValueError("vertexID must be specified when location is 'vertex'.")
        self._location = location
        self._faceIDs = [faceIDs] if isinstance(faceIDs, int) else faceIDs
        self._vertexIDs = [vertexIDs] if isinstance(vertexIDs, int) else vertexIDs
        #: The points of the particle the springs act on, where they are now.
        self._constrained_points = self._pointsWhereTheyAreNow()
        if not self._constrained_points:
            raise ValueError("Constraint '{:}': no point of the particle at the location '{:}'.".format(name, location))

        self._doProximityCheck = doProximityCheck
        if self._doProximityCheck:
            #: How close the obstacle must be for the constraint to take part.
            self._proximityDist = particle.getVolumeUndeformed() ** (1.0 / model.domainSize) * proximityFactor

    def _pointsWhereTheyAreNow(self) -> list:
        """The points of the particle at its location, where the particle has moved them.

        Returns
        -------
        list
            The coordinates of each point.
        """

        particle = self._particle
        if self._location == "center":
            return [particle.getCenterCoordinates()]
        if self._location == "face":
            return [particle.getFaceCoordinates(idx) for idx in self._faceIDs]
        vertices = particle.getVertexCoordinates()
        return [vertices[idx] for idx in self._vertexIDs]

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return [[self._field]] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(self) -> list:
        return []

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def updateConnectivity(self, model) -> bool:
        """Take the points where the particle has moved them, locate the nodes of its kernel functions in the block of
        the constraint, and decide whether the obstacle is within reach (:meth:`_isWithinReach`, always without a
        proximity check).

        Parameters
        ----------
        model
            The model.

        Returns
        -------
        bool
            True if the nodes of the particle's kernel functions changed.
        """

        self._constrained_points = self._pointsWhereTheyAreNow()

        # the nodes of the particle's kernel functions, and where they sit in the constraint's block
        localIndexOfNode = {}
        [self._localNodes] = locateTheNodesOfParticles([self._particle.kernelFunctions], localIndexOfNode)
        nodes = list(localIndexOfNode)
        hasChanged = nodes != self._nodes
        self._nodes = nodes

        self.withinReach = not self._doProximityCheck or self._isWithinReach(self._constrained_points)

        if not self.active:
            self.reactionForce = 0.0

        return hasChanged

    @abstractmethod
    def _isWithinReach(self, points: list) -> bool:
        """Whether one of the points is close enough to the obstacle to take part, within the reach the constructor
        took from its ``proximityFactor``.

        Parameters
        ----------
        points
            The coordinates of each point, where it is now.

        Returns
        -------
        bool
            True if the obstacle is within reach.
        """

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

    def _springsAt(self, points: np.ndarray, gaps: np.ndarray) -> SpringsOnPointsOfParticle:
        """The springs of the points at the gaps given: one of the penalty stiffness at every point with a negative gap,
        along the unit direction of the gradient of its gap, with the force :math:`f_p = -k\\, g_p` at that gap. A point
        whose gradient has no direction (its norm below :data:`GRADIENT_WITHOUT_DIRECTION`) gets no spring: no force,
        no reaction and no stiffness.

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

        closed = np.flatnonzero(gaps < 0.0)
        gradients = self._gradientsOfTheGaps(points[closed]).reshape((-1, self._fieldSize))
        norms = np.array([np.linalg.norm(gradient) for gradient in gradients])
        hasDirection = norms > GRADIENT_WITHOUT_DIRECTION
        withSpring = closed[hasDirection]
        return SpringsOnPointsOfParticle(
            # the vector the force is distributed with (see interpolationVectorAtPointOf)
            np.array(
                [self.interpolationVectorAtPointOf(self._particle, p).flatten() for p in points[withSpring]]
            ).reshape((len(withSpring), len(self._localNodes))),
            gradients[hasDirection] / norms[hasDirection, None],
            -self._penaltyParameter * gaps[withSpring],
            np.full(len(withSpring), self._penaltyParameter),
        )

    def _springs(self, timeStep: TimeStep) -> SpringsOnPointsOfParticle:
        """The penalty springs at the particle's current position and the gaps now (:meth:`_springsAt`): what the force
        and the bound of the stiffness are made from.

        Parameters
        ----------
        timeStep
            The current time step, for an obstacle that moves.

        Returns
        -------
        SpringsOnPointsOfParticle
            The springs.
        """

        points = np.array(self._constrained_points, dtype=float).reshape((-1, self._fieldSize))
        return self._springsAt(points, self._gapsOfPoints(points, timeStep))

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
    ):
        """Add the row sums of the absolute value of the stiffness of the penalty springs (:meth:`_springs`), weighted
        by column (see
        :func:`~edelweissmeshfree.constraints.base.pointsofparticles.addBoundOfAbsoluteStiffnessOfPointsOfParticle`):
        the springs of the points that penetrate now, those of the force.

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
        """

        addBoundOfAbsoluteStiffnessOfPointsOfParticle(bound, weights, self._localNodes, self._springs(timeStep))

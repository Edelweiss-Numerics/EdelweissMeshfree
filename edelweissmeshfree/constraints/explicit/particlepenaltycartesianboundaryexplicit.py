from collections.abc import Iterable

import numpy as np
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.constraints.base.penaltyspringsexplicit import (
    PenaltySpringsOnPointsOfOneParticleExplicit,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyCartesianBoundaryConstraintExplicit(PenaltySpringsOnPointsOfOneParticleExplicit):
    r"""
    This class implements a Penalty contact boundary constraint for EXPLICIT simulations.
    It prevents a particle from penetrating a defined boundary surface by applying
    penalty forces directly to the grid nodes.

    The constraint is applied based on the current position of the particle relative to the boundary.
    If penetration is detected, the penalty force is

    .. math::

        \boldsymbol{F}_{\mathrm{penalty}} = -k\, g\, \boldsymbol{n},

    where :math:`g < 0` is the gap of the point to the boundary, :math:`k` the penalty parameter and
    :math:`\boldsymbol{n}` the unit normal pointing away from the boundary, the gradient of the gap. The force and the
    bound of its stiffness for the critical time step are those of
    :class:`~edelweissmeshfree.constraints.base.penaltyspringsexplicit.PenaltySpringsOnPointsOfOneParticleExplicit`; the
    boundary gives the gap (:meth:`_gapsOfPoints`), its gradient (:meth:`_gradientsOfTheGaps`) and its reach
    (:meth:`_isWithinReach`).

    Parameters
    ----------
    name : str
        Name of the constraint.
    particle : BaseParticle
        The particle to which the constraint is applied.
    component : int
        The component (0 for x, 1 for y, 2 for z) of the displacement field.
    boundaryPosition : float
        The position of the boundary along the specified component.
    model : MPMModel
        The simulation model.
    location : str
        The location on the particle to apply the constraint ("center", "face", "vertex").
    faceIDs : list[int]
        The face IDs if location is "face".
    vertexIDs : list[int]
        The vertex IDs if location is "vertex".
    velocity : float
        The velocity of the boundary along the constrained component.
    penaltyParameter : float
        The penalty stiffness.
    doProximityCheck : bool
        Whether to perform a proximity check to deactivate distant constraints.
    proximityFactor : float
        The factor for proximity checking distance.
    interpolationLinearizedAboutCenter : bool
        False (default): measure the field at the point and distribute the force there with the shape functions at
        the point, N. True: with the particle's interpolation linearized about its center, G, with which the particle
        moves its own points (symmetric and conservative). See
        :meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`.
    """

    def __init__(
        self,
        name: str,
        particle: BaseParticle,
        component: int,
        boundaryPosition: float,
        model,
        location: str = "center",
        faceIDs: list[int] = None,
        vertexIDs: list[int] = None,
        velocity: float = 0.0,
        penaltyParameter: float = 1e5,
        doProximityCheck: bool = True,
        proximityFactor: float = 2.0,
        interpolationLinearizedAboutCenter: bool = False,
    ):
        self._component = component
        self._boundaryPosition = boundaryPosition
        self._velocity = velocity
        super().__init__(
            name,
            particle,
            model,
            location,
            faceIDs,
            vertexIDs,
            penaltyParameter,
            doProximityCheck,
            proximityFactor,
            interpolationLinearizedAboutCenter,
        )

        domainSize = model.domainSize
        if component < 0 or component >= domainSize:
            raise ValueError(f"Component {component} is out of bounds for domain size {domainSize}.")

        # Sanity check: all points must be on the same side initially
        first_pos = self._constrained_points[0][self._component]
        for p in self._constrained_points:
            if (p[self._component] - boundaryPosition) * (first_pos - boundaryPosition) < -1e-15:
                raise ValueError("All constrained points must be on the same side of the boundary.")

        # Determine orientation: 1.0 if boundary is "above" particle, -1.0 if "below"
        self._orientation = 1.0 if boundaryPosition > first_pos else -1.0

    def _isWithinReach(self, points: list) -> bool:
        """Whether one of the points is within reach of the boundary, at its position at the start, on either side.

        Parameters
        ----------
        points
            The coordinates of each point.

        Returns
        -------
        bool
            True if the boundary is within reach.
        """

        return any(abs(p[self._component] - self._boundaryPosition) <= self._proximityDist for p in points)

    def _gapsOfPoints(self, points: np.ndarray, timeStep: TimeStep) -> np.ndarray:
        """The gaps of points to the boundary where it is at the end of a time step, negative beyond it: the boundary
        above the particle (orientation 1.0) or below it (-1.0).

        Parameters
        ----------
        points
            The points, one row each.
        timeStep
            The time step, for the position of the moving boundary.

        Returns
        -------
        np.ndarray
            The gaps.
        """

        wallPosition = self._boundaryPosition + self._velocity * timeStep.stepProgress
        return -self._orientation * (points[:, self._component] - wallPosition)

    def _gradientsOfTheGaps(self, points: np.ndarray) -> np.ndarray:
        """The gradient of the gap of each point: the unit vector along the component the boundary is normal to,
        pointing away from the boundary, the same for every point.

        Parameters
        ----------
        points
            The points, one row each.

        Returns
        -------
        np.ndarray
            The gradients, one row each.
        """

        gradients = np.zeros((len(points), self._fieldSize))
        gradients[:, self._component] = -self._orientation
        return gradients


def ParticleExplicitPenaltyCartesianBoundaryConstraintFactory(
    baseName: str,
    boundaryPosition: float,
    component: int,
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    field: str,
    model: MPMModel,
    location: str = "center",
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
    penaltyParameter: float = 1e5,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
    interpolationLinearizedAboutCenter: bool = False,
):
    """
    Factory function to create ParticleExplicitPenaltyContactCartesianBoundaryConstraint; ``interpolationLinearizedAboutCenter``
    is passed to every constraint.
    """
    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraint = ParticlePenaltyCartesianBoundaryConstraintExplicit(
                    name,
                    p,
                    component,
                    boundaryPosition,
                    model,
                    location="face",
                    faceIDs=faceID,
                    penaltyParameter=penaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                    interpolationLinearizedAboutCenter=interpolationLinearizedAboutCenter,
                )
                constraints[name] = constraint
        return constraints

    elif isinstance(particleCollection, Iterable):
        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraint = ParticlePenaltyCartesianBoundaryConstraintExplicit(
                name,
                p,
                component,
                boundaryPosition,
                model,
                location,
                faceIDs,
                vertexIDs,
                penaltyParameter=penaltyParameter,
                doProximityCheck=doProximityCheck,
                proximityFactor=proximityFactor,
                interpolationLinearizedAboutCenter=interpolationLinearizedAboutCenter,
            )
            constraints[name] = constraint
        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")

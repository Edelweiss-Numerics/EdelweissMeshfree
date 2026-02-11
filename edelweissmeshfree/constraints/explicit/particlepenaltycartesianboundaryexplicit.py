from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyCartesianBoundaryConstraintExplicit(MPMConstraintBase):
    """
    This class implements a Penalty contact boundary constraint for EXPLICIT simulations.
    It prevents a particle from penetrating a defined boundary surface by applying
    penalty forces directly to the grid nodes.

    The constraint is applied based on the current position of the particle relative to the boundary.
    If penetration is detected:
        F_penalty = -k * g * n
    where g is penetration depth, k is penalty parameter, n is normal.

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
    ):
        self._name = name
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)

        # These are kept for Base Class compatibility, though unused in Explicit assembly
        self._nodes = dict()

        self._component = component
        self.reactionForce = 0.0

        self._boundaryPosition = boundaryPosition
        self._vertexIDs = vertexIDs
        self._faceIDs = faceIDs

        def _getConstraintLocations():
            if location == "center":
                return [particle.getCenterCoordinates()]
            elif location == "face":
                if self._faceIDs is None:
                    raise ValueError("faceID must be specified when location is 'face'.")
                ids = [self._faceIDs] if isinstance(self._faceIDs, int) else self._faceIDs
                return [particle.getFaceCoordinates(idx) for idx in ids]
            elif location == "vertex":
                if self._vertexIDs is None:
                    raise ValueError("vertexID must be specified when location is 'vertex'.")
                ids = [self._vertexIDs] if isinstance(self._vertexIDs, int) else self._vertexIDs
                vertices = particle.getVertexCoordinates()
                return [vertices[idx] for idx in ids]
            return []

        self._getConstraintLocations = _getConstraintLocations
        domainSize = model.domainSize
        self._constrained_points = self._getConstraintLocations()
        self._particle = particle

        if component < 0 or component >= domainSize:
            raise ValueError(f"Component {component} is out of bounds for domain size {domainSize}.")

        # Sanity check: all points must be on the same side initially
        first_pos = self._constrained_points[0][self._component]
        for p in self._constrained_points:
            if (p[self._component] - boundaryPosition) * (first_pos - boundaryPosition) < -1e-15:
                raise ValueError("All constrained points must be on the same side of the boundary.")

        # Determine orientation: 1.0 if boundary is "above" particle, -1.0 if "below"
        self._orientation = 1.0 if boundaryPosition > first_pos else -1.0

        self._velocity = velocity
        self._penaltyParameter = penaltyParameter
        self._doProximityCheck = doProximityCheck

        if self._doProximityCheck:
            self._proximityDist = particle.getVolumeUndeformed() ** (1.0 / domainSize) * proximityFactor

        self.isActive = True
        self._dof_indices = np.array([], dtype=np.int32)

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return [[self._field]] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(self) -> list:
        return []

    @property
    def active(self) -> bool:
        return self.isActive

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def updateConnectivity(self, model):
        """
        Updates the constraint connectivity.
        In explicit, this caches the global indices of the nodes for fast access.
        """
        # Update current coordinates of constrained points (particle may have moved)
        self._constrained_points = self._getConstraintLocations()

        # Update node set (for compatibility)
        nodes = [kf.node for kf in self._particle.kernelFunctions]
        hasChanged = nodes != self._nodes
        self._nodes = nodes

        wasActive = self.isActive
        self.isActive = True

        if self._doProximityCheck:
            self.isActive = False
            # Check if any point is close enough to boundary
            # Using simple distance check
            for p in self._constrained_points:
                dist = abs(p[self._component] - self._boundaryPosition)
                if dist <= self._proximityDist:
                    self.isActive = True
                    break

            if self.isActive != wasActive:
                hasChanged = True

        if not self.isActive:
            self.reactionForce = 0.0

        return hasChanged

    def applyConstraint(self, PExt: np.ndarray, timeStep: TimeStep):
        """
        Applies the penalty force to the external force vector PExt.
        """
        if not self.isActive:
            return

        # Explicit: calculate wall position at current time
        # Assuming stepProgress handles linear motion within step or total time
        currentWallPos = self._boundaryPosition + self._velocity * timeStep.stepProgress

        penalty = self._penaltyParameter
        orientation = self._orientation
        comp = self._component

        total_penalty_force = 0.0

        for p in self._constrained_points:
            val = p[comp]

            # Calculate penetration g
            # g > 0 implies penetration
            # orientation 1.0 (Wall > Point): Penetration if (Point > Wall) -> (Point - Wall) * 1 > 0
            # orientation -1.0 (Wall < Point): Penetration if (Point < Wall) -> (Point - Wall) * -1 > 0
            g = orientation * (val - currentWallPos)

            if g > 0:
                # Force magnitude: proportional to penetration
                # Force direction: Opposite to penetration direction (pushes particle back)
                # If orientation is 1.0 (Normal points -x), force should be -x.
                # If orientation is -1.0 (Normal points +x), force should be +x.
                # Combined: force_val = -penalty * g * orientation
                force_scalar = -penalty * g * orientation

                # Get shape functions for this specific constrained point
                N = self._particle.getInterpolationVector(p).flatten()

                # Vectorized add to global force vector
                # PExt[indices] += N * scalar
                # Using np.add.at is safer if indices repeat (unlikely here) but direct slicing is faster
                PExt[comp :: self._fieldSize] += N * force_scalar

                total_penalty_force += g * penalty

        self.reactionForce = total_penalty_force


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
):
    """
    Factory function to create ParticleExplicitPenaltyContactCartesianBoundaryConstraint.
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
            )
            constraints[name] = constraint
        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")

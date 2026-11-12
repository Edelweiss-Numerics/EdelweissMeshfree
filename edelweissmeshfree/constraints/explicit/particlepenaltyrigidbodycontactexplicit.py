from collections.abc import Callable, Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmexplicitconstraintbase import (
    MPMExplicitConstraintBase,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyContactImplicitSurfaceConstraintExplicit(MPMExplicitConstraintBase):
    """
    Penalty contact constraint for arbitrary implicit surfaces in EXPLICIT simulations.

    Parameters
    ----------
    particle
        The particle to which the constraint is applied.
    implicit_function
        Returns signed distance. Convention: Negative = Inside (Penetration).
    gradient_function
        Returns gradient vector. Convention: Points OUTWARD (Normal).
    model
        The MPM model instance.
    location
        Where to apply the constraint on the particle: "center", "face", or "vertex".
    faceIDs
        List of face IDs to apply constraint if location is "face".
    vertexIDs
        List of vertex IDs to apply constraint if location is "vertex".
    penaltyParameter
        The penalty stiffness parameter. Higher values reduce penetration but can cause numerical issues.
    doProximityCheck
        If True, constraints are only activated when particles are within a certain proximity to the surface.
    proximityFactor
        Multiplier for the proximity distance threshold, based on particle size. Default is 2.
    """

    #: Carries nothing from one increment to the next.
    checkpointedState = {}

    def __init__(
        self,
        name: str,
        particle: BaseParticle,
        implicit_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        model: MPMModel,
        location: str = "center",
        faceIDs: list[int] = None,
        vertexIDs: list[int] = None,
        penaltyParameter: float = 1e5,
        doProximityCheck: bool = True,
        proximityFactor: float = 2.0,
    ):
        self._name = name
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = []

        self._implicit_function = implicit_function
        self._gradient_function = gradient_function

        self.reactionForce = 0.0

        def _getConstraintLocations():
            if location == "center":
                return [particle.getCenterCoordinates()]
            elif location == "face":
                if faceIDs is None:
                    raise ValueError("faceID must be specified when location is 'face'.")
                ids = [faceIDs] if isinstance(faceIDs, int) else faceIDs
                return [particle.getFaceCoordinates(idx) for idx in ids]
            elif location == "vertex":
                if vertexIDs is None:
                    raise ValueError("vertexID must be specified when location is 'vertex'.")
                ids = [vertexIDs] if isinstance(vertexIDs, int) else vertexIDs
                vertices = particle.getVertexCoordinates()
                return [vertices[idx] for idx in ids]
            return []

        self._getConstraintLocations = _getConstraintLocations
        self._domainSize = model.domainSize
        self._constrained_points = self._getConstraintLocations()
        self._particle = particle

        self._penaltyParameter = penaltyParameter
        self._doProximityCheck = doProximityCheck

        if self._doProximityCheck:
            self._proximityDist = particle.getVolumeUndeformed() ** (1.0 / self._domainSize) * proximityFactor

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

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def updateConnectivity(self, model: MPMModel) -> bool:
        """
        Updates the constraint connectivity by refreshing active nodes and positions.

        Parameters
        ----------
        model
            The MPM model instance, used to access grid information if needed.

        Returns
        -------
        bool
            True if the connectivity has changed (e.g., active nodes or positions), False otherwise.
        """
        self._constrained_points = self._getConstraintLocations()

        # Refresh the grid nodes currently influenced by the particle
        nodes = [kf.node for kf in self._particle.kernelFunctions]
        hasChanged = nodes != self._nodes
        self._nodes = nodes

        # within reach if one of the constrained points is close enough to the surface (signed distance below it)
        self.withinReach = not self._doProximityCheck or any(
            self._implicit_function(p_coord) < self._proximityDist for p_coord in self._constrained_points
        )

        if not self.active:
            self.reactionForce = 0.0

        return hasChanged

    def computeKernelsExplicit(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, timeStep: TimeStep):
        """Add the residual contribution of the penalty forces: the negative of the forces with which the surface
        pushes the particle back, evaluated at the particle's current position.

        Parameters
        ----------
        U_np
            The current solution; unused, the penetration follows from the particle's current position.
        dU
            The solution increment; likewise unused.
        P
            The residual contribution to augment.
        timeStep
            The current time step information (not used in this constraint but included for interface consistency).
        """

        total_reaction = 0.0

        for p_coord in self._constrained_points:
            dist = self._implicit_function(p_coord)

            if dist < 0:
                g = abs(dist)
                grad = self._gradient_function(p_coord)
                grad_norm = np.linalg.norm(grad)

                normal = grad / grad_norm if grad_norm > 1e-14 else np.zeros(self._domainSize)

                force_vector = self._penaltyParameter * g * normal

                N = self._particle.getInterpolationVector(p_coord).flatten()

                # the residual contribution is the negative of the force on the nodes
                for d in range(self._domainSize):
                    P[d :: self._fieldSize] -= N * force_vector[d]

                total_reaction += self._penaltyParameter * g

        self.reactionForce = total_reaction


def ParticlePenaltyContactImplicitSurfaceConstraintExplicitFactory(
    baseName: str,
    implicit_function: Callable[[np.ndarray], float],
    gradient_function: Callable[[np.ndarray], np.ndarray],
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    model: MPMModel,
    location: str = "center",
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
    penaltyParameter: float = 1e5,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
):
    """
    Factory function to create Explicit Penalty Contact constraints for collections.

    Parameters
    ----------
    baseName
        Base name for constraints. Unique IDs will be appended.
    implicit_function
        Function that returns signed distance. Convention: Negative = Inside (Penetration).
    gradient_function
        Function that returns gradient vector. Convention: Points OUTWARD (Normal).
    particleCollection
        Either a list of particles or an EntityBasedSurface. If EntityBasedSurface, constraints are created for each face.
    model
        The MPM model instance.
    location
        Where to apply the constraint on the particle: "center", "face", or "vertex
    faceIDs
        List of face IDs to apply constraint if location is "face". Ignored if particleCollection
        is an EntityBasedSurface (constraints will be created for all faces).
    vertexIDs
        List of vertex IDs to apply constraint if location is "vertex".
    penaltyParameter
        The penalty stiffness parameter. Higher values reduce penetration but can cause numerical issues.
    doProximityCheck
        If True, constraints are only activated when particles are within a certain proximity to the surface.
    proximityFactor
        Multiplier for the proximity distance threshold, based on particle size. Default is 2.
    """
    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraints[name] = ParticlePenaltyContactImplicitSurfaceConstraintExplicit(
                    name,
                    p,
                    implicit_function,
                    gradient_function,
                    model,
                    location="face",
                    faceIDs=faceID,
                    penaltyParameter=penaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                )
        return constraints

    elif isinstance(particleCollection, Iterable):
        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraints[name] = ParticlePenaltyContactImplicitSurfaceConstraintExplicit(
                name,
                p,
                implicit_function,
                gradient_function,
                model,
                location,
                faceIDs,
                vertexIDs,
                penaltyParameter,
                doProximityCheck,
                proximityFactor,
            )
        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")

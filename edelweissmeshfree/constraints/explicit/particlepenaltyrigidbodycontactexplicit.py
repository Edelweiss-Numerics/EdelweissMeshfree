from collections.abc import Callable, Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.penaltyspringsexplicit import (
    PenaltySpringsOnPointsOfOneParticleExplicit,
)
from edelweissmeshfree.constraints.base.pointsofparticles import (
    locateTheNodesOfParticles,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyContactImplicitSurfaceConstraintExplicit(PenaltySpringsOnPointsOfOneParticleExplicit):
    """
    Penalty contact constraint for arbitrary implicit surfaces in EXPLICIT simulations: the surface gives the signed
    distance of a point (:meth:`_gapsOfPoints`) and its gradient (:meth:`_gradientsOfTheGaps`), the force and the bound
    of its stiffness for the critical time step are those of
    :class:`~edelweissmeshfree.constraints.base.penaltyspringsexplicit.PenaltySpringsOnPointsOfOneParticleExplicit`.

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
    interpolationLinearizedAboutCenter
        False (default): measure the field at the point and distribute the force there with the shape functions at
        the point, N. True: with the particle's interpolation linearized about its center, G, with which the particle
        moves its own points (symmetric and conservative). See
        :meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`.
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
        interpolationLinearizedAboutCenter: bool = False,
    ):
        self._name = name
        self.interpolationLinearizedAboutCenter = interpolationLinearizedAboutCenter
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        # the nodes of the particle's kernel functions, and where they sit in the block (see updateConnectivity)
        self._nodes = []
        self._localNodes = np.zeros(0, dtype=int)

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

        # the nodes of the particle's kernel functions, and where they sit in the constraint's block
        localIndexOfNode = {}
        [self._localNodes] = locateTheNodesOfParticles([self._particle.kernelFunctions], localIndexOfNode)
        nodes = list(localIndexOfNode)
        hasChanged = nodes != self._nodes
        self._nodes = nodes

        # within reach if one of the constrained points is close enough to the surface (signed distance below it)
        self.withinReach = not self._doProximityCheck or any(
            self._implicit_function(p_coord) < self._proximityDist for p_coord in self._constrained_points
        )

        if not self.active:
            self.reactionForce = 0.0

        return hasChanged

    def _gapsOfPoints(self, points: np.ndarray, timeStep: TimeStep) -> np.ndarray:
        """The signed distance of each point to the surface, negative inside it.

        Parameters
        ----------
        points
            The points, one row each.
        timeStep
            The time step; unused, the surface does not move.

        Returns
        -------
        np.ndarray
            The signed distances.
        """

        return np.array([self._implicit_function(p) for p in points], dtype=float)

    def _gradientsOfTheGaps(self, points: np.ndarray) -> np.ndarray:
        """The gradient of the signed distance at each point, pointing out of the surface.

        Parameters
        ----------
        points
            The points, one row each.

        Returns
        -------
        np.ndarray
            The gradients, one row each.
        """

        return np.array([self._gradient_function(p) for p in points], dtype=float).reshape(
            (len(points), self._fieldSize)
        )


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
    interpolationLinearizedAboutCenter: bool = False,
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
    interpolationLinearizedAboutCenter
        Passed to every constraint: False (default) the shape functions at the point, N; True the particle's
        interpolation linearized about its center, G.
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
                    interpolationLinearizedAboutCenter=interpolationLinearizedAboutCenter,
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
                interpolationLinearizedAboutCenter=interpolationLinearizedAboutCenter,
            )
        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")

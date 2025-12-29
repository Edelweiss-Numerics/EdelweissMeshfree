from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.particles.base.baseparticle import BaseParticle


class ParticlePenaltyContactCartesianBoundaryConstraint(MPMConstraintBase):
    """
    This class implements a Penalty contact boundary constraint.
    It prevents a particle from penetrating a defined boundary surface.
    The constraint is applied using Lagrange multipliers to enforce the contact condition.
    Only cartesian planes are supported as boundary surfaces.

    The constraint equation is defined as:
        g = n . (x - x_boundary) <= 0
    where n is the orientation of boundary, x is the particle position, and x_boundary is the boundary position.

    The orientation is determined based on the initial position of the particle relative to the boundary.

    Parameters
    ----------
    name : str
        Name of the constraint.
    particle : BaseParticle
        The particle to which the constraint is applied.
    location:
        Either "center", "face" or "vertex. For face or vertex, the IDs need
        to be specified in the particle definition. Note that faceIDs start (Abaqus-convention) at 1.
    component : int
        The component (0 for x, 1 for y, 2 for z) of
        the displacement field that is constrained.
    boundaryPosition : float
        The position of the boundary along the specified component.
    model
        The simulation model.
    location : str
        The location on the particle to apply the constraint. Can be "center", "face", or "vertex".
    faceID : int
        The face ID if location is "face".
    vertexID : int
        The vertex ID if location is "vertex".
    velocity : float
        The velocity of the boundary along the constrained component.
    penaltyParameter : float
        The penalty parameter for the constraint.
    doProximityCheck : bool
        Whether to perform a proximity check before applying the constraint.
    proximityFactor : float
        The factor to multiply the particle characteristic length for proximity checking.
    """

    def __init__(
        self,
        name: str,
        particle: BaseParticle,
        component: int,
        boundaryPosition: float,
        model,
        location: str = "center",
        faceID: int = None,
        vertexID: int = None,
        velocity: float = 0.0,
        penaltyParameter: float = 1e5,
        doProximityCheck: bool = True,
        proximityFactor: float = 2.0,
    ):
        self._name = name
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        self._component = component
        self.reactionForce = 0.0

        # determine normal direction:
        self._boundaryPosition = boundaryPosition

        def _getConstraintLocation():
            if location == "center":
                constraintLocation = particle.getCenterCoordinates()
            elif location == "face":
                if faceID is None:
                    raise ValueError("faceID must be specified when location is 'face'.")
                constraintLocation = particle.getFaceCoordinates(faceID)
            elif location == "vertex":
                if vertexID is None:
                    raise ValueError("vertexID must be specified when location is 'vertex'.")
                constraintLocation = particle.getVertexCoordinates(vertexID)
            return constraintLocation

        self._getConstraintLocation = _getConstraintLocation

        domainSize = model.domainSize

        self._particle_coordinate_j = self._getConstraintLocation()[component]
        self._particle = particle

        if component < 0 or component >= domainSize:
            raise ValueError(f"Component {component} is out of bounds for domain size {domainSize}.")

        self._orientation = 1.0 if boundaryPosition > self._particle_coordinate_j else -1.0

        self._velocity = velocity
        self._penaltyParameter = penaltyParameter
        self._doProximityCheck = doProximityCheck

        if self._doProximityCheck:
            self._particleCharacteristicLength_x_proximityFactor = (
                particle.getVolumeUndeformed() ** (1.0 / domainSize) * proximityFactor
            )

        self.isActive = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes.keys()

    @property
    def fieldsOnNodes(self) -> list:
        return [
            [
                self._field,
            ]
        ] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(
        self,
    ) -> list:
        return list()

    @property
    def active(self) -> bool:
        return self.isActive

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def updateConnectivity(self, model):
        nodes = {n: i for i, n in enumerate(set(kf.node for kf in self._particle.kernelFunctions))}

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        self._particle_coordinate_j = self._getConstraintLocation()[self._component]

        if self._doProximityCheck:
            distanceToBoundary = abs(self._particle_coordinate_j - self._boundaryPosition)
            if distanceToBoundary > self._particleCharacteristicLength_x_proximityFactor:
                hasChanged = self.isActive is not False
                self.isActive = False
            else:
                hasChanged = self.isActive is not True
                self.isActive = True

        if not self.isActive:
            self.reactionForce = 0.0

        return hasChanged

    def applyConstraint(self, dU_: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):

        if not self.isActive:
            return

        dU_U = dU_
        PExt_U = PExt

        K = V.reshape((self.nDof, self.nDof))

        K_UU = K

        i = self._component

        P_U_i = PExt_U[i :: self._fieldSize]
        dU_U_j = dU_U[i :: self._fieldSize]
        K_UU_ij = K_UU[i :: self._fieldSize, i :: self._fieldSize]

        N = self._particle.getInterpolationVector(self._getConstraintLocation())

        nodeIdcs = [self._nodes[kf.node] for kf in self._particle.kernelFunctions]

        deltaU_j = N @ dU_U_j[nodeIdcs]

        g_l = self._orientation * (
            deltaU_j - (-self._particle_coordinate_j + self._boundaryPosition + self._velocity * timeStep.stepProgress)
        )

        if g_l < 0:
            return

        penaltyForce = g_l * self._penaltyParameter
        self.reactionForce = penaltyForce

        dg_dU_j = N * self._orientation

        P_U_i[nodeIdcs] += self._penaltyParameter * dg_dU_j * g_l

        K_UU_ij[np.ix_(nodeIdcs, nodeIdcs)] += self._penaltyParameter * np.outer(dg_dU_j, dg_dU_j)


def ParticlePenaltyContactCartesianBoundaryConstraintFactory(
    baseName: str,
    boundaryPosition: float,
    component: int,
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    field: str,
    model: MPMModel,
    location: str = "center",
    faceID: int = None,
    vertexID: int = None,
    penaltyParameter: float = 1e5,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
):
    """
    Factory function to create ParticlePenaltyWeakDirichlet constraints on a collection of particles.

    Parameters
    ----------
    baseName
        The base name for the constraints.
    boundaryPosition
        The position of the boundary along the specified component.
    component
        The component (0 for x, 1 for y, 2 for z) of the boundary normal.
    particleCollection
        An iterable collection of particles or an EntityBasedSurface.
    field
        The field to be constrained.
    model
        The MPMModel instance.
    location
        The location on the particle to apply the constraint. Can be "center", "face", or "vertex".
    faceID
        The face ID if location is "face" and if particleCollection is not an EntityBasedSurface.
    vertexID
        The vertex ID if location is "vertex".
    penaltyParameter
        The penalty parameter for the constraint.
    doProximityCheck
        Whether to perform a proximity check before applying the constraint.
    proximityFactor
        The factor to multiply the particle characteristic length for proximity checking.

    Returns
    -------
    dict
        A dictionary mapping constraint names to ParticlePenaltyWeakDirichlet instances.
    """

    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraint = ParticlePenaltyContactCartesianBoundaryConstraint(
                    name,
                    p,
                    component,
                    boundaryPosition,
                    model,
                    location="face",
                    faceID=faceID,
                    penaltyParameter=penaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                )
                constraints[name] = constraint
        return constraints

    elif isinstance(particleCollection, Iterable):

        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraint = ParticlePenaltyContactCartesianBoundaryConstraint(
                name,
                p,
                component,
                boundaryPosition,
                model,
                location,
                faceID,
                vertexID,
                penaltyParameter=penaltyParameter,
                doProximityCheck=doProximityCheck,
                proximityFactor=proximityFactor,
            )
            constraints[name] = constraint

        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")

from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmpm.models.mpmmodel import MPMModel
from edelweissmpm.particles.base.baseparticle import BaseParticle


class ParticleLagrangianContactCartesianBoundaryConstraint(MPMConstraintBase):
    """
    This class implements a Lagrangian contact boundary constraint.
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
    ):
        self._name = name
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        self._component = component
        self._nLagrangianMultipliers = 1
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
        self._contactCurrentlyActive = True

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
        return len(self._nodes) * self._fieldSize + self._nLagrangianMultipliers

    @property
    def scalarVariables(
        self,
    ) -> list:
        return self._lagrangianMultipliers

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        return self._nLagrangianMultipliers

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        self._lagrangianMultipliers = scalarVariables

    def updateConnectivity(self, model):
        nodes = {n: i for i, n in enumerate(set(kf.node for kf in self._particle.kernelFunctions))}

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        self._particle_coordinate_j = self._getConstraintLocation()[self._component]

        return hasChanged

    def applyConstraint(self, dU_: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):

        dU_U = dU_[: -self._nLagrangianMultipliers]
        dU_L = dU_[-self._nLagrangianMultipliers :]
        PExt_U = PExt[: -self._nLagrangianMultipliers]
        PExt_L = PExt[-self._nLagrangianMultipliers :]

        K = V.reshape((self.nDof, self.nDof))

        K_UL = K[: -self._nLagrangianMultipliers, -self._nLagrangianMultipliers :]
        K_LU = K[-self._nLagrangianMultipliers :, : -self._nLagrangianMultipliers]
        K_LL = K[-self._nLagrangianMultipliers :, -self._nLagrangianMultipliers :]

        # lag is the index of the lagrangian multiplier
        # i is the component of the field being constrained
        i = self._component
        lag = 0

        P_U_i = PExt_U[i :: self._fieldSize]
        dU_U_j = dU_U[i :: self._fieldSize]

        K_UL_j = K_UL[i :: self._fieldSize, :]
        K_LU_j = K_LU[:, i :: self._fieldSize]

        dL_l = dU_L[lag]
        self.reactionForce = dL_l

        N = self._particle.getInterpolationVector(self._getConstraintLocation())

        nodeIdcs = [self._nodes[kf.node] for kf in self._particle.kernelFunctions]

        deltaU_j = N @ dU_U_j[nodeIdcs]

        g_l = self._orientation * (
            deltaU_j - (-self._particle_coordinate_j + self._boundaryPosition + self._velocity * timeStep.stepProgress)
        )

        gap_tol = 1e-13
        tensile_threshhold = 1e-13

        is_penetrating = g_l >= -gap_tol

        is_in_tension = (dL_l) < -tensile_threshhold

        should_be_active = True

        if not is_penetrating or is_in_tension:
            should_be_active = False

        if not should_be_active:
            # contact not active, but we need to ensure that no force is transferred
            PExt_L[lag] += dL_l
            K_LL[lag, lag] += 1.0

            return

        dg_l_dU_j = N * self._orientation

        P_U_i[nodeIdcs] += dL_l * dg_l_dU_j
        PExt_L[lag] += g_l

        K_UL_j[nodeIdcs, lag] += dg_l_dU_j
        K_LU_j[lag, nodeIdcs] += dg_l_dU_j

        self._contactCurrentlyActive = True


def ParticleLagrangianContactCartesianBoundaryConstraintFactory(
    baseName: str,
    boundaryPosition: float,
    component: int,
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    field: str,
    model: MPMModel,
    location: str = "center",
    faceID: int = None,
    vertexID: int = None,
):
    """
    Factory function to create ParticleLagrangianWeakDirichlet constraints on a collection of particles.

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

    Returns
    -------
    dict
        A dictionary mapping constraint names to ParticleLagrangianWeakDirichlet instances.
    """

    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraint = ParticleLagrangianContactCartesianBoundaryConstraint(
                    name, p, component, boundaryPosition, model, location="face", faceID=faceID
                )
                constraints[name] = constraint
        return constraints

    elif isinstance(particleCollection, Iterable):

        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraint = ParticleLagrangianContactCartesianBoundaryConstraint(
                name, p, component, boundaryPosition, model, location, faceID, vertexID
            )
            constraints[name] = constraint

        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")

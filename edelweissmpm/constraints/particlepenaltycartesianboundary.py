import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmpm.constraints.base.mpmconstraintbase import MPMConstraintBase
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

        # self._particle_coordinate_j = self._particle.getCenterCoordinates()[self._component]
        self._particle_coordinate_j = self._getConstraintLocation()[self._component]

        return hasChanged

    def applyConstraint(self, dU_: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):

        dU_U = dU_
        PExt_U = PExt

        K = V.reshape((self.nDof, self.nDof))

        K_UU = K[:, :]
        # K_LU = K[-self._nPenaltyMultipliers :, : -self._nLagrangianMultipliers]

        # lag is the index of the lagrangian multiplier
        # i is the component of the field being constrained
        i = self._component

        P_U_i = PExt_U[i :: self._fieldSize]
        dU_U_j = dU_U[i :: self._fieldSize]
        K_UU_ij = K_UU[i :: self._fieldSize, i :: self._fieldSize]

        # self.reactionForce = dL_l

        N = self._particle.getInterpolationVector(self._getConstraintLocation())

        nodeIdcs = [self._nodes[kf.node] for kf in self._particle.kernelFunctions]

        deltaU_j = N @ dU_U_j[nodeIdcs]

        g_l = self._orientation * (
            deltaU_j - (-self._particle_coordinate_j + self._boundaryPosition + self._velocity * timeStep.stepProgress)
        )

        if g_l < 0:
            return

        penaltyForce = g_l * self._penaltyParameter
        self.reactionForce = -penaltyForce

        dg_dU_j = N * self._orientation

        P_U_i[nodeIdcs] += self._penaltyParameter * dg_dU_j * g_l

        K_UU_ij[np.ix_(nodeIdcs, nodeIdcs)] += self._penaltyParameter * np.outer(dg_dU_j, dg_dU_j)


def ParticlePenaltyContactCartesianBoundaryConstraintFactory(
    baseName: str,
    particles: list[BaseParticle],
    component: int,
    boundaryPosition: float,
    model,
    location: str = "center",
    faceID: int = None,
    vertexID: int = None,
    velocity: float = 0.0,
    penaltyParameter: float = 1e5,
) -> dict[str, ParticlePenaltyContactCartesianBoundaryConstraint]:
    """
    Factory function to create multiple ParticlePenaltyContactBoundaryConstraint instances.

    Parameters
    ----------
    baseName : str
        Base name for the constraints.
    particles : list[BaseParticle]
        List of particles to which the constraints will be applied.
    component : int
        The component (0 for x, 1 for y, 2 for z) of the boundary normal.
    boundaryPosition : float
        The position of the boundary along the specified component.
    model
        The simulation model.
    location: str
        Either "center", "face" or "vertex. For face or vertex, the IDs need
        to be specified in the particle definition. Note that faceIDs start (Abaqus-convention) at 1.
    faceID: int, optional
        The ID of the face if location is "face".
    vertexID: int, optional
        The ID of the vertex if location is "vertex".

    Returns
    -------
    dict[str, ParticlePenaltyContactBoundaryConstraint]
        A dictionary of created constraints with their names as keys.
    """
    constraints = dict()
    for i, p in enumerate(particles):
        name = f"{baseName}_{i}"
        constraint = ParticlePenaltyContactCartesianBoundaryConstraint(
            name=name,
            particle=p,
            component=component,
            boundaryPosition=boundaryPosition,
            model=model,
            location=location,
            faceID=faceID,
            vertexID=vertexID,
            velocity=velocity,
            penaltyParameter=penaltyParameter,
        )
        constraints[name] = constraint
    return constraints

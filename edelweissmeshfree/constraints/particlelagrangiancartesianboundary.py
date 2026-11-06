from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticleLagrangianContactCartesianBoundaryConstraint(MPMConstraintBase):
    r"""
    This class implements a Lagrangian contact boundary constraint.
    It prevents a particle from penetrating a defined boundary surface.
    The constraint is applied using Lagrange multipliers to enforce the contact condition.
    Only cartesian planes are supported as boundary surfaces.

    The constraint equation is defined as

    .. math::

        g = \boldsymbol{n} \cdot (\boldsymbol{x} - \boldsymbol{x}_{\mathrm{boundary}}) \leq 0,

    where :math:`\boldsymbol{n}` is the orientation of the boundary, :math:`\boldsymbol{x}` the particle position, and
    :math:`\boldsymbol{x}_{\mathrm{boundary}}` the boundary position.

    The orientation is determined based on the initial position of the particle relative to the boundary.

    The particle interpolates its displacement from the nodes of its kernel functions, as an increment on the current
    configuration, so the gap is measured with the increment :math:`\boldsymbol{N} \cdot \Delta\mathbf{U}_j` of the
    constrained component :math:`j` from the position at the start of the increment. While the particle is in contact,
    a multiplier :math:`\lambda` enforces :math:`g = 0`, and the constraint adds

    .. math::

        \mathbf{P}_{U_j} \mathrel{+}= \lambda\, \frac{\partial g}{\partial \Delta\mathbf{U}_j}, \qquad
        P_{\lambda} \mathrel{+}= g, \qquad
        \mathbf{K}_{U_j \lambda} \mathrel{+}= \frac{\partial g}{\partial \Delta\mathbf{U}_j}, \qquad
        \mathbf{K}_{\lambda U_j} \mathrel{+}= \frac{\partial g}{\partial \Delta\mathbf{U}_j}^T,

    with :math:`\partial g / \partial \Delta\mathbf{U}_j = \boldsymbol{n}_j \boldsymbol{N}`. The multiplier is the
    contact force, positive in compression. A particle that does not penetrate the boundary, or whose multiplier would
    pull it towards the boundary, is out of contact: the constraint then adds :math:`P_{\lambda} \mathrel{+}= \lambda`
    and :math:`K_{\lambda\lambda} \mathrel{+}= 1`, which holds the multiplier at zero.

    The gap is one on increments, the multiplier is a total: the contact force that balances the total stress of the
    particle, read from ``U_np``, where it continues from the last converged increment. ``reactionForce`` holds it.

    Parameters
    ----------
    name : str
        Name of the constraint.
    particle : BaseParticle
        The particle to which the constraint is applied.
    component : int
        The component (0 for x, 1 for y, 2 for z) of
        the displacement field that is constrained.
    boundaryPosition : float
        The position of the boundary along the specified component.
    model
        The simulation model.
    location
        Where the constraint acts on the particle: at its "center", at the face given by faceID ("face"), or at the
        vertex given by vertexID ("vertex").
    faceID
        The face if location is "face"; face IDs start (Abaqus convention) at 1.
    vertexID
        The vertex if location is "vertex".
    velocity
        The velocity of the boundary along the constrained component.
    """

    #: The multiplier is a scalar variable of the model, which the model's checkpoint holds.
    checkpointedState = {}

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
                constraintLocation = particle.getVertexCoordinates()[vertexID]
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
        nodes = {
            n: i
            for i, n in enumerate(sorted(set(kf.node for kf in self._particle.kernelFunctions), key=lambda n: n.label))
        }

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        self._particle_coordinate_j = self._getConstraintLocation()[self._component]

        return hasChanged

    def computeKernels(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, K: np.ndarray, timeStep: TimeStep):
        # the nodal field values come first, the multiplier last; the nodal values are increments, the multiplier a
        # total
        nU = len(self._nodes) * self._fieldSize
        dU_U = dU[:nU]
        multiplier = U_np[nU]
        P_U = P[:nU]
        P_L = P[nU:]

        K = K.reshape((self.nDof, self.nDof))
        K_UL = K[:nU, nU:]
        K_LU = K[nU:, :nU]
        K_LL = K[nU:, nU:]

        # j is the constrained component of the field, lag the index of the multiplier
        j = self._component
        lag = 0

        P_U_j = P_U[j :: self._fieldSize]
        dU_U_j = dU_U[j :: self._fieldSize]
        K_UL_j = K_UL[j :: self._fieldSize, :]
        K_LU_j = K_LU[:, j :: self._fieldSize]

        self.reactionForce = multiplier

        N = self._particle.getInterpolationVector(self._getConstraintLocation())

        nodeIdcs = [self._nodes[kf.node] for kf in self._particle.kernelFunctions]

        deltaU_j = N @ dU_U_j[nodeIdcs]

        g = self._orientation * (
            deltaU_j - (-self._particle_coordinate_j + self._boundaryPosition + self._velocity * timeStep.stepProgress)
        )

        gap_tol = 1e-13
        tensile_threshhold = 1e-13

        is_penetrating = g >= -gap_tol

        is_in_tension = multiplier < -tensile_threshhold

        if not is_penetrating or is_in_tension:
            # out of contact: hold the multiplier at zero, so that no force is transferred
            P_L[lag] += multiplier
            K_LL[lag, lag] += 1.0

            return

        dg_dU_j = N * self._orientation

        P_U_j[nodeIdcs] += multiplier * dg_dU_j
        P_L[lag] += g

        K_UL_j[nodeIdcs, lag] += dg_dU_j
        K_LU_j[lag, nodeIdcs] += dg_dU_j

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

from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticleLagrangianWeakDirichlet(MPMConstraintBase):
    r"""Prescribes the increment of a field at one point of one particle -- its center, the center of one of its faces,
    or one of its vertices -- with Lagrange multipliers.

    It is the Lagrange multiplier counterpart of
    :class:`~edelweissmeshfree.constraints.particlepenaltyweakdirichtlet.ParticlePenaltyWeakDirichlet`. A penalty
    spring enforces the prescribed value only approximately, the more closely the stiffer it is; a multiplier enforces
    it exactly, at the cost of one more unknown per prescribed component, and of a zero block on the diagonal of the
    tangent. To constrain a set of particles, create one constraint per particle with
    :func:`ParticleLagrangianWeakDirichletOnParticleSetFactory`: every constraint couples only the nodes of its own
    particle, so its tangent block stays small, where one constraint on all particles would couple all nodes of the set
    in one dense block.

    A particle interpolates the field from the nodes of its kernel functions, as an increment on the current
    configuration: :math:`u(\boldsymbol{x}) = \boldsymbol{N}(\boldsymbol{x}) \cdot \Delta\mathbf{U}`. At the
    constrained point :math:`\boldsymbol{x}_c` and for every prescribed component :math:`i`, a multiplier
    :math:`\lambda_i` enforces

    .. math::

        g_i = \boldsymbol{N}(\boldsymbol{x}_c) \cdot \Delta\mathbf{U}_i - \Delta\bar{u}_i = 0,

    where :math:`\Delta\bar{u}_i` is the share of the step's prescribed increment that falls into the current
    increment. Following the rule of every constraint (see
    :class:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase`), the constraint adds

    .. math::

        \mathbf{P}_{U_i} \mathrel{+}= \lambda_i\, \boldsymbol{N}(\boldsymbol{x}_c), \qquad
        P_{\lambda_i} \mathrel{+}= g_i, \qquad
        \mathbf{K}_{U_i \lambda_i} \mathrel{+}= \boldsymbol{N}(\boldsymbol{x}_c), \qquad
        \mathbf{K}_{\lambda_i U_i} \mathrel{+}= \boldsymbol{N}(\boldsymbol{x}_c)^T,

    so that it acts on the particle with the point force :math:`-\lambda_i` at :math:`\boldsymbol{x}_c`. The tangent is
    symmetric, as the constraint stems from the potential :math:`\sum_i \lambda_i g_i`.

    The condition :math:`g_i` is one on increments, the multiplier :math:`\lambda_i` is a total: the point force that
    balances the total stress of the particle, read from ``U_np``, where it continues from the last converged
    increment. ``reactionForce`` holds, per component of the field, the multiplier: the force with which the particle
    acts on the constraint.

    The multipliers are ordered as the components in ``prescribedStepDelta``.

    Parameters
    ----------
    name
        The name of this constraint.
    constrainedParticle
        The particle whose field is to be constrained.
    field
        The field this constraint is acting on.
    prescribedStepDelta
        The prescribed increment of the field over the step, by component.
    model
        The full MPMModel instance.
    location
        The point of the particle to constrain: ``"center"``, ``"face"`` (the center of the face ``faceID``) or
        ``"vertex"`` (the vertex ``vertexID``).
    faceID
        The face whose center is constrained, if ``location`` is ``"face"``.
    vertexID
        The vertex that is constrained, if ``location`` is ``"vertex"``.
    """

    #: The multipliers are scalar variables of the model, which the model's checkpoint holds.
    checkpointedState = {}

    def __init__(
        self,
        name: str,
        constrainedParticle: BaseParticle,
        field: str,
        prescribedStepDelta: dict,
        model: MPMModel,
        location: str = "center",
        faceID: int = None,
        vertexID: int = None,
    ):
        self._name = name
        self._constrainedParticle = constrainedParticle
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        if location not in ("center", "face", "vertex"):
            raise ValueError("Constraint '{:}': location must be 'center', 'face' or 'vertex'.".format(name))
        if location == "face" and faceID is None:
            raise ValueError("Constraint '{:}': faceID must be specified when location is 'face'.".format(name))
        if location == "vertex" and not isinstance(vertexID, int):
            raise ValueError(
                "Constraint '{:}': vertexID must be one vertex index when location is 'vertex'.".format(name)
            )
        self._location = location
        self._faceID = faceID
        self._vertexID = vertexID

        self._nLagrangianMultipliers = len(self._prescribedStepDelta)
        if self._nLagrangianMultipliers == 0:
            raise ValueError("Constraint '{:}' constrains nothing: it needs prescribed components.".format(name))

        self.reactionForce = np.zeros(self._fieldSize)

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
            for i, n in enumerate(
                sorted(set(kf.node for kf in self._constrainedParticle.kernelFunctions), key=lambda n: n.label)
            )
        }

        hasChanged = nodes != self._nodes
        self._nodes = nodes

        return hasChanged

    def _constrainedPointCoordinates(self) -> np.ndarray:
        """The current coordinates of the constrained point of the particle.

        Returns
        -------
        np.ndarray
            The coordinates of the center, of the center of the face, or of the vertex.
        """

        particle = self._constrainedParticle
        if self._location == "center":
            return particle.getCenterCoordinates()
        if self._location == "face":
            return particle.getFaceCoordinates(self._faceID)
        return particle.getVertexCoordinates()[self._vertexID]

    def computeKernels(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, K: np.ndarray, timeStep: TimeStep):
        # the nodal field values come first, the multipliers last; the nodal values are increments, the multipliers
        # totals
        nU = len(self._nodes) * self._fieldSize
        dU_U = dU[:nU]
        multipliers = U_np[nU:]
        P_U = P[:nU]
        P_L = P[nU:]

        K = K.reshape((self.nDof, self.nDof))
        K_UL = K[:nU, nU:]
        K_LU = K[nU:, :nU]

        self.reactionForce.fill(0.0)

        particle = self._constrainedParticle
        N = particle.getInterpolationVector(self._constrainedPointCoordinates())
        nodeIdcs = [self._nodes[kf.node] for kf in particle.kernelFunctions]

        for lag, (i, prescribedComponent) in enumerate(self._prescribedStepDelta.items()):
            # lag counts the multipliers, i is the component of the field
            dU_U_i = dU_U[i :: self._fieldSize]
            P_U_i = P_U[i :: self._fieldSize]
            K_UL_i = K_UL[i :: self._fieldSize, :]
            K_LU_i = K_LU[:, i :: self._fieldSize]
            multiplier = multipliers[lag]

            g = N @ dU_U_i[nodeIdcs] - prescribedComponent * timeStep.stepProgressIncrement
            dg_dU_i = N

            P_U_i[nodeIdcs] += multiplier * dg_dU_i
            P_L[lag] += g

            K_UL_i[nodeIdcs, lag] += dg_dU_i
            K_LU_i[lag, nodeIdcs] += dg_dU_i

            self.reactionForce[i] += multiplier


def ParticleLagrangianWeakDirichletOnParticleSetFactory(
    baseName: str,
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    field: str,
    prescribedStepDelta: dict,
    model: MPMModel,
    location: str = "center",
    faceID: int = None,
    vertexID: int = None,
) -> dict[str, ParticleLagrangianWeakDirichlet]:
    """Creates one :class:`ParticleLagrangianWeakDirichlet` per particle of a collection, all constraining the same
    point of their particles.

    Parameters
    ----------
    baseName
        The base name of the constraints.
    particleCollection
        The particles to constrain, or an EntityBasedSurface, whose particles are constrained at the faces it names.
    field
        The field to be constrained.
    prescribedStepDelta
        The prescribed increment of the field over the step, by component.
    model
        The MPMModel instance.
    location
        The point of every particle to constrain: ``"center"``, ``"face"`` or ``"vertex"``.
    faceID
        The face whose center is constrained, if ``location`` is ``"face"`` and ``particleCollection`` is not an
        EntityBasedSurface.
    vertexID
        The vertex that is constrained, if ``location`` is ``"vertex"``. To constrain several vertices of every
        particle, call this factory once per vertex.

    Returns
    -------
    dict[str, ParticleLagrangianWeakDirichlet]
        The constraints, by name.
    """

    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraints[name] = ParticleLagrangianWeakDirichlet(
                    name, p, field, prescribedStepDelta, model, location, faceID
                )
        return constraints

    if isinstance(particleCollection, Iterable):
        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraints[name] = ParticleLagrangianWeakDirichlet(
                name, p, field, prescribedStepDelta, model, location, faceID, vertexID
            )
        return constraints

    raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")

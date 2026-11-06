from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticleLagrangianDirichlet(MPMConstraintBase):
    r"""Prescribes the increment of a field at points of a set of particles -- their centers, or some of their
    vertices -- with Lagrange multipliers.

    It is the Lagrange multiplier counterpart of
    :class:`~edelweissmeshfree.constraints.particlepenaltyweakdirichtlet.ParticlePenaltyWeakDirichlet`, and takes the
    same arguments, save the penalty parameter. A penalty spring enforces the prescribed value only approximately, the
    more closely the stiffer it is; a multiplier enforces it exactly, at the cost of one more unknown per point and
    component, and of a zero block on the diagonal of the tangent. Where
    :class:`~edelweissmeshfree.constraints.particlelagrangianweakdirichlet.ParticleLagrangianWeakDirichlet` constrains
    one point of one particle, this constraint constrains all points of all its particles, and sums their multipliers
    into one reaction force.

    A particle interpolates the field from the nodes of its kernel functions, as an increment on the current
    configuration: :math:`u(\boldsymbol{x}) = \boldsymbol{N}(\boldsymbol{x}) \cdot \Delta\mathbf{U}`. At every
    constrained point :math:`\boldsymbol{x}_c` and for every prescribed component :math:`i`, a multiplier
    :math:`\lambda` enforces

    .. math::

        g = \boldsymbol{N}(\boldsymbol{x}_c) \cdot \Delta\mathbf{U}_i - \Delta\bar{u}_i = 0,

    where :math:`\Delta\bar{u}_i` is the share of the step's prescribed increment that falls into the current
    increment. Following the rule of every constraint (see
    :class:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase`), the constraint adds

    .. math::

        \mathbf{P}_{U_i} \mathrel{+}= \lambda\, \boldsymbol{N}(\boldsymbol{x}_c), \qquad
        P_\lambda \mathrel{+}= g, \qquad
        \mathbf{K}_{U_i \lambda} \mathrel{+}= \boldsymbol{N}(\boldsymbol{x}_c), \qquad
        \mathbf{K}_{\lambda U_i} \mathrel{+}= \boldsymbol{N}(\boldsymbol{x}_c)^T,

    so that it acts on the particle with the point force :math:`-\lambda` at :math:`\boldsymbol{x}_c`.
    ``reactionForce`` holds, per component of the field, the sum of the multipliers: the force with which the particles
    act on the constraint.

    The multipliers are ordered point by point -- particle by particle, and within a particle center or vertex by
    vertex --, and within a point component by component.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The full MPMModel instance.
    constrainedParticles
        The particles to be constrained.
    field
        The field this constraint is acting on.
    prescribedStepDelta
        The prescribed increment of the field over the step, by component.
    constrain
        ``"center"`` to constrain the center of every particle, or a list of vertex indices to constrain these vertices
        of every particle.
    """

    #: Carries nothing from one increment to the next: the multipliers are unknowns of every increment.
    checkpointedState = {}

    def __init__(
        self,
        name: str,
        model: MPMModel,
        constrainedParticles: Iterable[BaseParticle],
        field: str,
        prescribedStepDelta: dict,
        constrain: str | list[int] = "center",
    ):
        self._name = name
        self._constrainedParticles = list(constrainedParticles)
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        if isinstance(constrain, str) and constrain == "center":
            self._constrainedVertices = None
            nPointsPerParticle = 1
        elif isinstance(constrain, list) and constrain and all(isinstance(i, int) for i in constrain):
            self._constrainedVertices = constrain
            nPointsPerParticle = len(constrain)
        else:
            raise ValueError(
                "Constraint '{:}': constrain must be 'center' or a non-empty list of vertex indices.".format(name)
            )

        self._nLagrangianMultipliers = (
            len(self._constrainedParticles) * nPointsPerParticle * len(self._prescribedStepDelta)
        )
        if self._nLagrangianMultipliers == 0:
            raise ValueError(
                "Constraint '{:}' constrains nothing: it needs particles and prescribed components.".format(name)
            )

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
                sorted(
                    set(kf.node for p in self._constrainedParticles for kf in p.kernelFunctions), key=lambda n: n.label
                )
            )
        }

        hasChanged = nodes != self._nodes
        self._nodes = nodes

        return hasChanged

    def _shapeFunctionsAtConstrainedPoints(self) -> list[tuple[np.ndarray, list[int]]]:
        """The shape functions of the particles at the constrained points, point by point.

        Returns
        -------
        list[tuple[np.ndarray, list[int]]]
            For every point, the values of the shape functions there, and the indices of their nodes in this
            constraint's node list.
        """

        shapeFunctionsAtPoints = []
        for p in self._constrainedParticles:
            nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

            if self._constrainedVertices is None:
                constrainedCoordinates = [p.getCenterCoordinates()]
            else:
                constrainedCoordinates = p.getVertexCoordinates()[self._constrainedVertices]

            for coordinates in constrainedCoordinates:
                shapeFunctionsAtPoints.append((p.getInterpolationVector(coordinates), nodeIdcs))

        return shapeFunctionsAtPoints

    def computeKernels(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, K: np.ndarray, timeStep: TimeStep):
        # the nodal field values come first, the multipliers last
        nU = len(self._nodes) * self._fieldSize
        dU_U = dU[:nU]
        multipliers = dU[nU:]
        P_U = P[:nU]
        P_L = P[nU:]

        K = K.reshape((self.nDof, self.nDof))
        K_UL = K[:nU, nU:]
        K_LU = K[nU:, :nU]

        self.reactionForce.fill(0.0)

        shapeFunctionsAtPoints = self._shapeFunctionsAtConstrainedPoints()
        nComponents = len(self._prescribedStepDelta)

        for comp, (i, prescribedComponent) in enumerate(self._prescribedStepDelta.items()):
            # comp counts the prescribed components, i is the component of the field
            dU_U_i = dU_U[i :: self._fieldSize]
            P_U_i = P_U[i :: self._fieldSize]
            K_UL_i = K_UL[i :: self._fieldSize, :]
            K_LU_i = K_LU[:, i :: self._fieldSize]

            for point, (N, nodeIdcs) in enumerate(shapeFunctionsAtPoints):
                lag = point * nComponents + comp
                multiplier = multipliers[lag]

                g = N @ dU_U_i[nodeIdcs] - prescribedComponent * timeStep.stepProgressIncrement
                dg_dU_i = N

                P_U_i[nodeIdcs] += multiplier * dg_dU_i
                P_L[lag] += g

                K_UL_i[nodeIdcs, lag] += dg_dU_i
                K_LU_i[lag, nodeIdcs] += dg_dU_i

                self.reactionForce[i] += multiplier

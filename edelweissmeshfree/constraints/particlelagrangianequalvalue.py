import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticleLagrangianEqualValueConstraint(MPMConstraintBase):
    r"""Constrains the field of two particles to change equally, in some components, with Lagrange multipliers.

    Each particle interpolates the field at its center from the nodes of its kernel functions, as an increment on the
    current configuration: :math:`\boldsymbol{N}_m \cdot \Delta\mathbf{U}` for the master particle,
    :math:`\boldsymbol{N}_s \cdot \Delta\mathbf{U}` for the slave particle. For every constrained component :math:`i`, a
    multiplier :math:`\lambda_i` enforces

    .. math::

        g_i = \boldsymbol{N}_m \cdot \Delta\mathbf{U}_i - \boldsymbol{N}_s \cdot \Delta\mathbf{U}_i = 0.

    Following the rule of every constraint (see
    :class:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase`), the constraint adds

    .. math::

        \mathbf{P}_{U_i} \mathrel{+}= \lambda_i\, \frac{\partial g_i}{\partial \Delta\mathbf{U}_i}, \qquad
        P_{\lambda_i} \mathrel{+}= g_i, \qquad
        \mathbf{K}_{U_i \lambda_i} \mathrel{+}= \frac{\partial g_i}{\partial \Delta\mathbf{U}_i}, \qquad
        \mathbf{K}_{\lambda_i U_i} \mathrel{+}= \frac{\partial g_i}{\partial \Delta\mathbf{U}_i}^T,

    with :math:`\partial g_i / \partial \Delta\mathbf{U}_i = \boldsymbol{N}_m - \boldsymbol{N}_s`: the master particle
    receives the point force :math:`-\lambda_i` at its center, the slave particle :math:`+\lambda_i`.

    The condition :math:`g_i` is one on increments -- two particles constrained from the start keep equal values -- the
    multiplier :math:`\lambda_i` is a total: the force that the two particles transmit, read from ``U_np``, where it
    continues from the last converged increment. ``reactionForce`` holds the multiplier per component of the field.

    Parameters
    ----------
    name
        The name of this constraint.
    masterParticle
        The master particle whose field value is to be matched.
    components
        The components of the field to be constrained.
    slaveParticle
        The slave particle whose field value is to be constrained to match the master particle.
    field
        The field this constraint is acting on.
    model
        The full MPMModel instance.
    """

    #: The multipliers are scalar variables of the model, which the model's checkpoint holds.
    checkpointedState = {}

    def __init__(
        self,
        name: str,
        masterParticle: BaseParticle,
        components: list[int] | int,
        slaveParticle: BaseParticle,
        field: str,
        model,
    ):
        self._name = name
        self._field = field
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        self._masterParticle = masterParticle
        self._slaveParticle = slaveParticle
        if isinstance(components, int):
            self._components = [components]
        elif isinstance(components, list):
            self._components = components
        else:
            raise Exception("Invalid components format")

        self._components = np.asarray(self._components)
        self._nLagrangianMultipliers = len(self._components)
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
                    set(kf.node for p in [self._masterParticle, self._slaveParticle] for kf in p.kernelFunctions),
                    key=lambda n: n.label,
                )
            )
        }

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        return hasChanged

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

        N_master = self._masterParticle.getInterpolationVector(self._masterParticle.getCenterCoordinates())
        N_slave = self._slaveParticle.getInterpolationVector(self._slaveParticle.getCenterCoordinates())
        masterNodeIdcs = [self._nodes[kf.node] for kf in self._masterParticle.kernelFunctions]
        slaveNodeIdcs = [self._nodes[kf.node] for kf in self._slaveParticle.kernelFunctions]
        dg_dU_master = N_master
        dg_dU_slave = -N_slave

        for lag, i in enumerate(self._components):
            # lag counts the multipliers, i is the component of the field
            dU_U_i = dU_U[i :: self._fieldSize]
            P_U_i = P_U[i :: self._fieldSize]
            K_UL_i = K_UL[i :: self._fieldSize, :]
            K_LU_i = K_LU[:, i :: self._fieldSize]
            multiplier = multipliers[lag]

            g = N_master @ dU_U_i[masterNodeIdcs] - N_slave @ dU_U_i[slaveNodeIdcs]

            P_U_i[masterNodeIdcs] += multiplier * dg_dU_master
            P_U_i[slaveNodeIdcs] += multiplier * dg_dU_slave
            P_L[lag] += g

            K_UL_i[masterNodeIdcs, lag] += dg_dU_master
            K_LU_i[lag, masterNodeIdcs] += dg_dU_master
            K_UL_i[slaveNodeIdcs, lag] += dg_dU_slave
            K_LU_i[lag, slaveNodeIdcs] += dg_dU_slave

            self.reactionForce[i] += multiplier


def ParticleLagrangianEqualValueConstraintOnParticleSetFactory(
    name: str,
    constrainedParticleSet: set[BaseParticle],
    components: list[int] | int,
    field: str,
    model,
) -> list[ParticleLagrangianEqualValueConstraint]:
    """
    Factory function to create a set of ParticleLagrangianEqualValueConstraint instances
    that constrain all particles in the given set to have equal values for the specified field components.

    Parameters
    ----------
    name
        The base name for the constraints.
    constrainedParticleSet
        The set of particles to be constrained.
    components
        The components of the field to be constrained.
    field
        The field this constraint is acting on.
    model
        The full MPMModel instance.

    Returns
    -------
    dict
        A dictionary of ParticleLagrangianEqualValueConstraint instances, keyed by their names.
    """
    if len(constrainedParticleSet) < 2:
        raise ValueError("At least two particles are required to create equal value constraints.")

    constrainedParticleList = list(constrainedParticleSet)
    masterParticle = constrainedParticleList[0]
    constraints = dict()
    for i, slaveParticle in enumerate(constrainedParticleList[1:]):
        constraintName = f"{name}_eq_{i}"
        constraint = ParticleLagrangianEqualValueConstraint(
            constraintName,
            masterParticle,
            components,
            slaveParticle,
            field,
            model,
        )
        constraints[constraintName] = constraint

    return constraints

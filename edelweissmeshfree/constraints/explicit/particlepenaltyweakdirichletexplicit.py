import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyWeakDirichletExplicit(MPMConstraintBase):
    """
    This is an implementation of weak Dirichlet boundary conditions using a penalty formulation
    for EXPLICIT dynamics. It penalizes the difference between the actual and prescribed field.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The full MPMModel instance.
    constrainedParticles
        The list of particles to be constrained.
    field
        The field this constraint is acting on.
    prescribedStepDelta
        The dictionary containing the prescribed bc components for the field in the present load step.
    penaltyParameter
        The penalty parameter value.
    constrain
        Either constrain the center of the particle or a list of vertex indices for particles with multiple vertices.
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        constrainedParticles: list[BaseParticle],
        field: str,
        prescribedStepDelta: dict,
        penaltyParameter: float,
        constrain: str | list[int] = "center",
        **kwargs,
    ):
        self._name = name
        self._model = model
        self._constrainedParticles = constrainedParticles
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._penaltyParameter = penaltyParameter
        self._nodes = dict()
        if constrain == "center":
            self._constrainVertices = None
        else:
            if not isinstance(constrain, list):
                raise ValueError("Constrain must be 'center' or a list of vertex indices.")
            if len(constrain) > 0 and not all(isinstance(i, int) for i in constrain):
                raise ValueError("Constrain must be 'center' or a list of vertex indices.")
            self._constrainVertices = constrain

        self.penaltyForce = np.zeros(self._fieldSize)

        if "f_t" in kwargs:
            self._f_t = kwargs["f_t"]
        else:
            self._f_t = lambda x: 1.0  # By default, use full value in explicit

        self.isActive = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return list(self._nodes.keys())

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
        nodes = {
            n: i for i, n in enumerate(set(kf.node for p in self._constrainedParticles for kf in p.kernelFunctions))
        }

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes
        return hasChanged

    def applyConstraint(self, PExt: np.ndarray, timeStep: TimeStep):
        nodeField = self._model.nodeFields.get(self._field)
        if nodeField is None or "U" not in nodeField:
            return

        self.penaltyForce.fill(0.0)

        for i, prescribedComponent in self._prescribedStepDelta.items():
            P_i = PExt[i :: self._fieldSize]
            target_value = prescribedComponent * self._f_t(timeStep.stepProgress)

            total_force = 0.0

            for p in self._constrainedParticles:
                if self._constrainVertices:
                    constrainedCoordinates = p.getVertexCoordinates()[self._constrainVertices]
                else:
                    constrainedCoordinates = [p.getCenterCoordinates()]

                for constrainedCoordinate in constrainedCoordinates:
                    N = p.getInterpolationVector(constrainedCoordinate)
                    nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

                    # Compute actual displacement
                    actual_u = 0.0
                    for kf, n_val in zip(p.kernelFunctions, N):
                        actual_u += n_val * nodeField.subset(kf.node)["U"][0][i]

                    deviation = actual_u - target_value
                    force_scalar = -self._penaltyParameter * deviation

                    P_i[nodeIdcs] += N * force_scalar
                    total_force += force_scalar

            self.penaltyForce[i] = total_force

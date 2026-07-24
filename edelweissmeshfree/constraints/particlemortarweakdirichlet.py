from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticleMortarWeakDirichlet(MPMConstraintBase):
    """Mortar-type weak Dirichlet boundary condition on a set of particles.

    Instead of collocating the constraint pointwise at every particle (which produces
    alternating point reactions on nearly incompressible media and, with it, a pressure
    checkerboard along the boundary), the constraint is enforced in the integrated sense

        g_k = sum_q w_q P_k(xi_q) ( u_i(x_q) - ubar_i ) = 0,   k = 0..multiplierOrder,

    with Legendre polynomials P_k on the edge parametrization xi in [-1, 1] and the
    particle centers x_q as quadrature points. The multiplier space is a smooth,
    LOW-dimensional polynomial space (satisfying the Babuska condition for boundary
    multiplier spaces), and the resulting boundary traction

        t_i(s) = sum_k lambda_ik P_k(s)

    is smooth by construction: no alternating point forces, no pressure checkerboard.

    Parameters
    ----------
    name
        The name of this constraint.
    constrainedParticles
        The particles along the boundary (quadrature points at their centers).
    field
        The field this constraint is acting on.
    prescribedStepDelta
        A dictionary mapping field component indices to their prescribed step increments.
    model
        The full MPMModel instance.
    multiplierOrder
        Highest polynomial order of the multiplier field per constrained component.
    """

    def __init__(
        self,
        name: str,
        constrainedParticles: Iterable[BaseParticle],
        field: str,
        prescribedStepDelta: dict,
        model: MPMModel,
        multiplierOrder: int = 4,
    ):
        self._name = name
        self._constrainedParticles = list(constrainedParticles)
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._multiplierOrder = multiplierOrder
        self._nModes = multiplierOrder + 1
        self._nodes = dict()

        if len(self._constrainedParticles) < self._nModes:
            raise ValueError("more multiplier modes than constrained particles")

        self._nLagrangianMultipliers = len(self._prescribedStepDelta) * self._nModes
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
                dict.fromkeys(kf.node for p in self._constrainedParticles for kf in p.kernelFunctions)
            )
        }

        hasChanged = nodes != self._nodes
        self._nodes = nodes

        return hasChanged

    def _edgeParametrizationAndWeights(self):
        """Parametrize the quadrature (particle center) points along the edge by arc length,
        mapped to xi in [-1, 1]; weights are the tributary lengths."""

        coords = np.array([p.getCenterCoordinates() for p in self._constrainedParticles])
        s = np.zeros(len(coords))
        s[1:] = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))

        length = s[-1] if s[-1] > 0 else 1.0
        xi = 2.0 * s / length - 1.0

        w = np.zeros(len(s))
        w[1:-1] = (s[2:] - s[:-2]) / 2.0
        w[0] = (s[1] - s[0]) / 2.0
        w[-1] = (s[-1] - s[-2]) / 2.0
        # normalize: absolute scale is absorbed by the multipliers, but a normalized
        # measure keeps the multiplier magnitudes ~ integrated tractions
        w /= length

        return xi, w

    def applyConstraint(self, dU_: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):
        nL = self._nLagrangianMultipliers
        dU_U = dU_[:-nL]
        dU_L = dU_[-nL:]
        PExt_U = PExt[:-nL]
        PExt_L = PExt[-nL:]

        K = V.reshape((self.nDof, self.nDof))
        K_UL = K[:-nL, -nL:]
        K_LU = K[-nL:, :-nL]

        self.reactionForce.fill(0.0)

        xi, w = self._edgeParametrizationAndWeights()
        # Legendre polynomials evaluated at the quadrature points: (nModes, nQ)
        P_legendre = np.array([np.polynomial.legendre.Legendre.basis(k)(xi) for k in range(self._nModes)])

        # interpolation of the constrained field at the quadrature points: rows of B map the
        # (union) node values to the field values at the particle centers
        B = np.zeros((len(self._constrainedParticles), len(self._nodes)))
        for q, p in enumerate(self._constrainedParticles):
            N = p.getInterpolationVector(p.getCenterCoordinates())
            nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]
            B[q, nodeIdcs] = N

        for comp, (i, prescribedComponent) in enumerate(self._prescribedStepDelta.items()):
            dU_U_j = dU_U[i :: self._fieldSize]
            P_U_i = PExt_U[i :: self._fieldSize]
            K_UL_j = K_UL[i :: self._fieldSize, :]
            K_LU_j = K_LU[:, i :: self._fieldSize]

            u_q = B @ dU_U_j  # field values at the quadrature points
            gap_q = u_q - prescribedComponent * timeStep.stepProgressIncrement

            for k in range(self._nModes):
                lag = comp * self._nModes + k
                dL_l = dU_L[lag]

                wPk = w * P_legendre[k]
                g_l = wPk @ gap_q
                dg_l_dU = wPk @ B

                P_U_i += dL_l * dg_l_dU
                PExt_L[lag] += g_l

                K_UL_j[:, lag] += dg_l_dU
                K_LU_j[lag, :] += dg_l_dU

                if k == 0:
                    # P_0 = 1: the zeroth multiplier is the resultant of the (normalized)
                    # traction distribution
                    self.reactionForce[i] += dL_l


def ParticleMortarWeakDirichletOnParticleSetFactory(
    baseName: str,
    particleCollection: Iterable[BaseParticle],
    field: str,
    prescribedStepDelta: dict,
    model: MPMModel,
    multiplierOrder: int = 4,
):
    """Factory mirroring ParticleLagrangianWeakDirichletOnParticleSetFactory: returns a dict
    with a single mortar constraint over the whole particle collection."""

    return {
        baseName: ParticleMortarWeakDirichlet(
            baseName, particleCollection, field, prescribedStepDelta, model, multiplierOrder
        )
    }

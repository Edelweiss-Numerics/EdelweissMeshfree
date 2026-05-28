"""Penalty constraint for enforcing equal-value conditions between nodes."""

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.materialpoints.base.mp import MaterialPointBase
from edelweissmeshfree.models.mpmmodel import MPMModel


class PenaltyEqualValue(MPMConstraintBase):
    """
    This is an implementation of an equal value constraint using a penalty formulation.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The full MPMModel instance.
    constrainedMaterialPoints
        The list of material points to be constrained.
    field
        The field this constraint is acting on.
    prescribedComponent
        The index of the constrained component.
    penaltyParameter
        The penalty parameter value.
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        constrainedMaterialPoints: list[MaterialPointBase],
        field: str,
        prescribedComponent: int,
        penaltyParameter: float,
    ):
        """Initialize the object.

        Parameters
        ----------
        name
            The unique name of the object.
        model
            The model associated with the object.
        constrainedMaterialPoints
            The list of material points subject to this constraint.
        field
            The field associated with the object.
        prescribedComponent
            The index of the field component to constrain.
        penaltyParameter
            The penalty stiffness used to enforce the constraint.
        """
        self._name = name
        self._model = model
        self._constrainedMPs = constrainedMaterialPoints
        self._field = field
        self._prescribedComponent = prescribedComponent
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._penaltyParameter = penaltyParameter
        self._nodes = dict()

    @property
    def name(self) -> str:
        """The name of this constraint."""
        return self._name

    @property
    def nodes(self) -> list:
        """The nodes involved in this constraint."""
        return self._nodes.keys()

    @property
    def fieldsOnNodes(self) -> list:
        """The fields required on the constrained nodes."""
        return [
            [
                self._field,
            ]
        ] * len(self._nodes)

    @property
    def nDof(self) -> int:
        """The number of degrees of freedom of this constraint."""
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(
        self,
    ) -> list:
        """The scalar variables associated with this constraint."""
        return []

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        """Return the number of additional scalar variables required by this constraint."""
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        """Assign the additional scalar variables required by this constraint.

        Parameters
        ----------
        scalarVariables
            The scalar variables assigned to the constraint.
        """
        pass

    def updateConnectivity(self, model):
        """Update the nodal connectivity used by this constraint.

        Parameters
        ----------
        model
            The model providing the current connectivity information.
        """
        nodes = {
            n: i for i, n in enumerate(set(n for mp in self._constrainedMPs for c in mp.assignedCells for n in c.nodes))
        }

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        return hasChanged

    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):
        """Apply this constraint contribution to the current system vectors and matrices."""
        i = self._prescribedComponent
        P_i = PExt[i :: self._fieldSize]
        dU_j = dU[i :: self._fieldSize]

        K_ij = V.reshape((self.nDof, self.nDof))[i :: self._fieldSize, i :: self._fieldSize]

        # Part 1: compute the mean values over all material points

        meanValue = 0.0
        dMeanValue_dDU_j = np.zeros_like(dU_j)
        for mp in self._constrainedMPs:
            center = mp.getCenterCoordinates()

            for c in mp.assignedCells:
                nodeIdcs = [self._nodes[n] for n in c.nodes]

                N = c.getInterpolationVector(center)

                meanValue += N @ dU_j[nodeIdcs]
                dMeanValue_dDU_j[nodeIdcs] += N

        meanValue /= len(self._constrainedMPs)
        dMeanValue_dDU_j /= len(self._constrainedMPs)

        # Part 2: compute the penalized difference between mean value and mp value:
        for mp in self._constrainedMPs:
            center = mp.getCenterCoordinates()

            for c in mp.assignedCells:
                N = c.getInterpolationVector(center)

                nodeIdcs = [self._nodes[n] for n in c.nodes]

                mpValue = N @ dU_j[nodeIdcs]

                P_i[nodeIdcs] += N * self._penaltyParameter * (mpValue - meanValue)

                K_ij[np.ix_(nodeIdcs, nodeIdcs)] += np.outer(N, N) * self._penaltyParameter
                K_ij[nodeIdcs, :] += np.outer(N, dMeanValue_dDU_j) * -1 * self._penaltyParameter

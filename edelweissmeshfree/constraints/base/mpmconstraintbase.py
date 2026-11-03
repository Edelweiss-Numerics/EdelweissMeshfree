"""
Constraints are used to enforce certain conditions on the system. They are used to enforce boundary conditions, contact conditions, etc.
Technically, constraints are implemented as a set of kernels, which are applied to the system matrix and the external load vector.
In this regard, constraints are similar to elements, but they mey also use additional scalar variables to enforce the conditions.
Unlike elements, they are not subject to external loads.
"""

from abc import ABC, abstractmethod

import numpy as np
from edelweissfe.numerics.vijentitybase import VIJEntityBase
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.models.mpmmodel import MPMModel


class MPMConstraintBase(ABC, VIJEntityBase):
    """The MPMConstraintBase class is an abstract base class for all constraints.
    If you want to implement a new constraint, you have to inherit from this class."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this constraint.

        Returns
        -------
        str
            The name."""

    @property
    def active(self) -> bool:
        """Whether this constraint is active or not.

        Returns
        -------
        bool
            The truth value.
        """
        return True

    @property
    @abstractmethod
    def nodes(self) -> list[Node]:
        """The nodes this constraint is acting on.
        Duplicates are _allowed_.

        Returns
        -------
        list[Node]
            The list of nodes."""

    @property
    @abstractmethod
    def fieldsOnNodes(self) -> list[list[str]]:
        """The fields on the nodes this constraint is acting on.

        Returns
        -------
        list[list[str]]
            The node-wise list of fields."""

    @property
    @abstractmethod
    def nDof(self) -> int:
        """The total number of degrees of freedom this constraint is associated with.

        Returns
        -------
        int
            The total number of degrees of freedom."""

    @property
    @abstractmethod
    def scalarVariables(
        self,
    ) -> list[ScalarVariable]:
        """The list of assigned ScalarVariable.

        Returns
        -------
        list[ScalarVariable]
            The list of scalar variables."""

    @abstractmethod
    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        """Tell the framework how many scalar variables (e.g., Lagrangian multipliers)
        this constraint needs.

        Returns
        -------
        int
            The number of requested ScalarVariable
        """

    @abstractmethod
    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        """This is the list of constraint specific scalar variables which are assigned to this constraint.

        Parameters
        ----------
        scalarVariables
            The list of ScalarVariable to be assigned.
        """

    @abstractmethod
    def updateConnectivity(self, model: MPMModel) -> bool:
        """This method is called before each new timeStep, after material point connectivity was updated, but before the global equation system is created.
        If the contribution to the global system changes, True is returned.

        Parameters
        ----------
        model
            The current model.

        Returns
        -------
        bool
            The truth value if the connectivity has changed.
        """

    def acceptLastState(self):
        """Called by :meth:`~edelweissfe.models.femodel.FEModel.advanceToTime` (via
        :meth:`~edelweissmeshfree.models.mpmmodel.MPMModel.advanceToTime`) when an increment is
        accepted, so a stateful constraint can promote the state of the last (converged) Newton
        iterate to its history.

        The default implementation does nothing, which is correct for every stateless constraint
        (i.e. every constraint that does not override this method)."""

    def getRestartData(self) -> dict[str, np.ndarray] | None:
        """Capture the state of this constraint for inclusion in a restart checkpoint.

        The default implementation returns ``None``, indicating the constraint carries no
        checkpointable internal state. Override this in stateful constraints.

        Returns
        -------
        dict[str, np.ndarray] | None
            A mapping from dataset names to NumPy arrays, or ``None`` if stateless.
        """
        return None

    def setRestartData(self, restartData: dict[str, np.ndarray]) -> None:
        """Restore the state of this constraint from a restart checkpoint.

        Parameters
        ----------
        restartData
            The mapping previously returned by :meth:`getRestartData`.
        """

    @abstractmethod
    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):
        """Apply the constraint, i.e., compute the 'kernels'. Add the contributions to the external load vector and the system matrix.

        Parameters
        ----------
        dU
            The current increment since the last time the constraint was applied.
        PExt
            The external load vector.
        V
            The system (stiffness) matrix.
        timeStep
            The current step and total time.
        """

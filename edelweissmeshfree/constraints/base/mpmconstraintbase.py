"""
Constraints are used to enforce certain conditions on the system. They are used to enforce boundary conditions, contact conditions, etc.
Technically, a constraint is an entity of the equation system like an element: it adds a residual contribution and its
tangent, and it may introduce additional scalar variables (e.g., Lagrange multipliers) to enforce its condition.
Unlike elements, constraints are not subject to external loads.
"""

from abc import abstractmethod

from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.variables.scalarvariable import ScalarVariable


class MPMConstraintBase(ConstraintBase):
    """The base class of the constraints of EdelweissMeshfree; implement a new constraint by deriving from it.

    A meshfree constraint is an EdelweissFE constraint
    (:class:`~edelweissfe.constraints.base.constraintbase.ConstraintBase`), and follows its rules:

    - :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.computeKernels` ``(U_np, dU, P, K, timeStep)``
      adds the constraint's residual contribution to ``P`` -- the *negative* of the force it exerts on its nodes,
      what an element of the same mechanics would add as its internal force -- and its tangent ``K = dP/dU``.
      See the EdelweissFE solver documentation, *The residual and the tangent*.
    - The nodes of a meshfree constraint are those its particles or material points currently interpolate from, so
      they change as the kernel functions and cells move: ``updateConnectivity`` finds them anew at the start of
      every increment, and reports a change.
    - A constraint may switch itself off in ``updateConnectivity``, e.g. a contact constraint whose particle is far
      from the boundary: it sets
      :attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.withinReach`, and reports a change of
      :attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.active`. A constraint switched off from
      outside, through ``active``, stays off whatever it finds, and rests: the model does not locate it anew.
    - Every constraint declares the state it carries from one increment to the next, in ``checkpointedState``
      (empty if it carries none), so that a restart can restore it.

    The nodal values of the particles and of the cells are increments on the current configuration, so the
    constraints here read ``dU``, the increment since the last converged state, and leave ``U_np`` unused.

    A constraint that acts only in an explicit analysis, with forces and without a tangent, derives from
    :class:`~edelweissmeshfree.constraints.base.mpmexplicitconstraintbase.MPMExplicitConstraintBase` instead.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this constraint.

        Returns
        -------
        str
            The name."""

    @property
    @abstractmethod
    def scalarVariables(
        self,
    ) -> list[ScalarVariable]:
        """The scalar variables (e.g., Lagrange multipliers) assigned to this constraint.

        Returns
        -------
        list[ScalarVariable]
            The list of scalar variables."""

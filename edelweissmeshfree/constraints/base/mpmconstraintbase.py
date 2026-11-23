"""
Constraints are used to enforce certain conditions on the system. They are used to enforce boundary conditions, contact conditions, etc.
Technically, a constraint is an entity of the discrete system like an element: it adds a residual contribution and its
tangent, and it may introduce additional scalar variables (e.g., Lagrange multipliers) to enforce its condition.
Unlike elements, constraints are not subject to external loads.
"""

from abc import abstractmethod

import numpy as np
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
      every increment, and reports whether they changed.
    - A constraint may switch itself off in ``updateConnectivity``, e.g. a contact constraint whose particle is far
      from the boundary: it sets
      :attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.withinReach`; the model sees the change of
      :attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.active` itself. A constraint switched off from
      outside, through ``active``, stays off whatever it finds, and rests: the model does not locate it anew.
    - Every constraint declares the state it carries from one increment to the next, in ``checkpointedState``
      (empty if it carries none), so that a restart can restore it.

    The nodal values of the particles and of the cells are increments on the current configuration, so the
    constraints here read them from ``dU``, the increment since the last converged state. A Lagrange multiplier, in
    contrast, is a force or a traction, which balances the total stresses of the particles or elements: it is a
    total value, read from ``U_np``, where the solver continues it from the last converged increment. The model holds
    the multipliers as its scalar variables, and its checkpoint holds them with it.

    A constraint that acts only in an explicit analysis, with forces and without a tangent, derives from
    :class:`~edelweissmeshfree.constraints.base.mpmexplicitconstraintbase.MPMExplicitConstraintBase` instead. Whether a
    constraint acts in an explicit analysis at all, it declares in :attr:`actsInExplicitAnalysis`.
    """

    #: Whether this constraint acts in an explicit analysis (see
    #: :attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.actsInExplicitAnalysis`). The explicit
    #: dynamic solver ``NED`` evaluates the constraints of a meshfree model at the configuration it has just accepted:
    #: the particles have moved, and the nodes of the kernel functions carry neither a solution nor an increment. A
    #: constraint that measures its condition at the particles' current positions acts there, e.g. a contact with a
    #: boundary, or a weak Dirichlet condition that holds the displacement at zero; a constraint on the increment, e.g.
    #: a weak Dirichlet condition that prescribes a change, or an equal value, would act as if nothing had moved, and the
    #: solver refuses it. Not by default, then, unlike a constraint of finite elements.
    actsInExplicitAnalysis: bool = False

    #: The nodes of a meshfree constraint are those its particles or material points currently interpolate from, so they
    #: change as the particles move (see
    #: :attr:`~edelweissfe.nodecouplingentity.base.nodecouplingentity.BaseNodeCouplingEntity.connectivityChangesWithMotion`): a solver that
    #: never updates the connectivity, e.g. ``NEST``, refuses it.
    connectivityChangesWithMotion = True

    #: Whether this constraint uses, at a point of a particle, the particle's interpolation linearized about its center
    #: (True) or the shape functions at the point (False); see :meth:`interpolationVectorAtPointOf`. The constraints
    #: that act on points of particles set it from their constructor option ``interpolationLinearizedAboutCenter``.
    interpolationLinearizedAboutCenter: bool = False

    def interpolationVectorAtPointOf(self, particle, coordinates: np.ndarray) -> np.ndarray:
        """The vector with which this constraint measures a field at a point of a particle (its center, a vertex, a
        face center) and distributes its force there over the nodes of the particle's kernel functions.

        - ``interpolationLinearizedAboutCenter = False`` (the default): the shape functions at the point,
          :meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.getInterpolationVector`, :math:`\\mathbf{N}`.
          A constraint on the increment of a field there is then symmetric. But the particle itself moves the point
          with its interpolation linearized about its center, :math:`\\mathbf{G}`, not with :math:`\\mathbf{N}`: a
          constraint on the position of the point (a support or a contact in an explicit analysis, where the point is
          where the particle has put it) distributes its force with :math:`\\mathbf{N}` while the point moves with
          :math:`\\mathbf{G}`. Its stiffness :math:`k\\,\\mathbf{N}\\mathbf{G}^\\mathsf{T}` is unsymmetric and not
          conservative: it does work over a closed cycle of motion and can feed energy into softened material next to
          a support.
        - ``interpolationLinearizedAboutCenter = True``: the particle's own interpolation at its own points,
          :meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.getInterpolationVectorLinearizedAboutCenter`,
          :math:`\\mathbf{G}`, for measuring and for distributing, for every field. The stiffness
          :math:`k\\,\\mathbf{G}\\mathbf{G}^\\mathsf{T}` is symmetric and the constraint conservative.

        Parameters
        ----------
        particle
            The particle.
        coordinates
            The point, in the current configuration of the particle.

        Returns
        -------
        np.ndarray
            The vector, one value per kernel function of the particle.
        """

        if self.interpolationLinearizedAboutCenter:
            return particle.getInterpolationVectorLinearizedAboutCenter(coordinates)
        return particle.getInterpolationVector(coordinates)

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

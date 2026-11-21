import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyWeakDirichlet(MPMConstraintBase):
    r"""
    This is an implementation of weak Dirichlet boundary conditions using a penalty formulation.
    It constrains a material point field increment.

    **In an implicit analysis**, it penalizes the difference between the increment of the field at the constrained
    points, :math:`\boldsymbol{N} \cdot \Delta\boldsymbol{U}`, and the prescribed increment, the change of
    ``prescribedStepDelta`` times ``f_t`` over the increment. The shape functions :math:`\boldsymbol{N}` at the point
    both measure the increment and distribute the force: the stiffness :math:`k\,\boldsymbol{N}\boldsymbol{N}^\mathsf{T}`
    is symmetric.

    **In an explicit analysis**, the nodes of the kernel functions carry no increment to constrain: the particles have
    moved when the constraints are evaluated. There the constraint holds displacement components at zero, as a
    support, e.g. on a symmetry plane or at a clamped edge, and it measures them where the constrained points are now:
    the force on a constrained point is

    .. math::

        f_i = -k \left( x_i - X_i \right)

    for every constrained component :math:`i`, with :math:`\boldsymbol{X}` the position of the point when the
    constraint was created. The point (the center or a vertex of the particle) is where the particle has put it: it
    moves with the particle, in an increment by :math:`\boldsymbol{G} \cdot \Delta\boldsymbol{U}`, with the particle's
    interpolation vector of an attached point
    (:meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.getAttachedPointInterpolationVector`), which
    away from the center is not the vector of the shape functions at the point. The force is distributed over the
    nodes of the kernel functions with the same :math:`\boldsymbol{G}`, the transpose of the point's motion: the
    penalty stiffness :math:`k\,\boldsymbol{G}\boldsymbol{G}^\mathsf{T}` is symmetric, and the support does no work
    over a closed cycle of motion. (Distributed with the shape functions at the point, the stiffness would be
    :math:`k\,\boldsymbol{N}\boldsymbol{G}^\mathsf{T}`, unsymmetric: it pumps energy into the material next to the
    support once that has softened.) The force acts on both sides of :math:`X_i`. It acts in an explicit analysis
    (:attr:`actsInExplicitAnalysis`) only if it constrains the displacement and every prescribed change is zero: an
    explicit analysis prescribes motion through velocities, not through penalized displacements.

    A support holds only as well as its penalty is stiff against the material next to it: a penalty that is merely
    comparable to the stiffness of a particle lets the support give way and spring back, an error that does not shrink
    with the particle size. Its upper bound is the stable time increment of the explicit integration,
    :math:`\Delta t < 2 \sqrt{m / k}` with :math:`m` the mass a constrained point carries; see example 186.

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
        If "center", the center of the particle is constrained. If a list, the vertices at the given indices are constrained.
    """

    #: Carries nothing from one increment to the next: the positions the points are held at in an explicit analysis
    #: are those of the model the constraint is created with.
    checkpointedState = {}

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

        #: Where the constrained points of each particle are when the constraint is created, the positions an explicit
        #: analysis holds them at.
        self._initialCoordinatesOfConstrainedPoints = [
            np.array(self._constrainedCoordinatesOf(particle)) for particle in self._constrainedParticles
        ]

        if "f_t" in kwargs:
            self._f_t = kwargs["f_t"]
        else:
            self._f_t = lambda x: x

    @property
    def actsInExplicitAnalysis(self) -> bool:
        """Whether this constraint acts in an explicit analysis: if it constrains the displacement, and every prescribed
        change is zero (see the class documentation and
        :attr:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.actsInExplicitAnalysis`).

        Returns
        -------
        bool
            True if it acts in an explicit analysis.
        """

        return self._field == "displacement" and all(delta == 0.0 for delta in self._prescribedStepDelta.values())

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes.keys()

    @property
    def fields(self) -> list:
        return [
            [
                self._field,
            ]
        ] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(
        self,
    ) -> list:
        return []

    def getNumberOfAdditionalNeededScalarVariables(
        self,
    ) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def updateConnectivity(self, model):
        nodes = {
            n: i
            for i, n in enumerate(
                sorted(
                    set(kf.node for p in self._constrainedParticles for kf in p.kernelFunctions), key=lambda n: n.label
                )
            )
        }

        hasChanged = False
        if nodes != self._nodes:
            hasChanged = True

        self._nodes = nodes

        return hasChanged

    def computeKernels(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, K: np.ndarray, timeStep: TimeStep):
        for i, prescribedComponent in self._prescribedStepDelta.items():
            P_i = P[i :: self._fieldSize]
            dU_j = dU[i :: self._fieldSize]

            K_ij = K.reshape((self.nDof, self.nDof))[i :: self._fieldSize, i :: self._fieldSize]

            for p in self._constrainedParticles:
                for constrainedCoordinate in self._constrainedCoordinatesOf(p):

                    N = p.getInterpolationVector(constrainedCoordinate)

                    nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

                    mpValue = N @ dU_j[nodeIdcs]

                    P_i[nodeIdcs] += (
                        N
                        * self._penaltyParameter
                        * (
                            mpValue
                            - prescribedComponent
                            * (
                                self._f_t(timeStep.stepProgress)
                                - self._f_t(timeStep.stepProgress - timeStep.stepProgressIncrement)
                            )
                        )
                    )
                    K_ij[np.ix_(nodeIdcs, nodeIdcs)] += np.outer(N, N) * self._penaltyParameter

            self.penaltyForce[i] = np.sum(P_i)

    def computeKernelsExplicit(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, timeStep: TimeStep):
        """Add the residual contribution of the support in an explicit analysis: the negative of the forces that hold
        the constrained components of the constrained points at their initial positions, measured where the points are
        now (see the class documentation).

        Parameters
        ----------
        U_np
            The current solution; unused, the displacement follows from the current positions.
        dU
            The solution increment; likewise unused.
        P
            The residual contribution to augment.
        timeStep
            The current time step; unused.
        """

        for p, initialCoordinates in zip(self._constrainedParticles, self._initialCoordinatesOfConstrainedPoints):
            nodeIdcs = [self._nodes[kf.node] for kf in p.kernelFunctions]

            for constrainedCoordinate, initialCoordinate in zip(self._constrainedCoordinatesOf(p), initialCoordinates):
                G = p.getAttachedPointInterpolationVector(constrainedCoordinate)

                for i in self._prescribedStepDelta:
                    P[i :: self._fieldSize][nodeIdcs] += (
                        G * self._penaltyParameter * (constrainedCoordinate[i] - initialCoordinate[i])
                    )

        for i in self._prescribedStepDelta:
            self.penaltyForce[i] = np.sum(P[i :: self._fieldSize])

    def _constrainedCoordinatesOf(self, particle: BaseParticle) -> list[np.ndarray]:
        """The current coordinates of the constrained points of a particle: its center, or the constrained vertices.

        Parameters
        ----------
        particle
            The particle.

        Returns
        -------
        list[np.ndarray]
            The coordinates, one array per point.
        """

        if self._constrainVertices:
            return list(particle.getVertexCoordinates()[self._constrainVertices])
        return [particle.getCenterCoordinates().copy()]

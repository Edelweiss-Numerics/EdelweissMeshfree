import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.constraints.base.pointsofparticles import (
    SpringsOnPointsOfParticle,
    addBoundOfAbsoluteStiffnessOfPointsOfParticle,
    addForcesOfSpringsOnPointsOfParticle,
    locateTheNodesOfParticles,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyWeakDirichlet(MPMConstraintBase):
    r"""
    This is an implementation of weak Dirichlet boundary conditions using a penalty formulation.
    It constrains a material point field increment.

    **In an implicit analysis**, it penalizes the difference between the increment of the field at the constrained
    points, :math:`\boldsymbol{N} \cdot \Delta\boldsymbol{U}`, and the prescribed increment, the change of
    ``prescribedStepDelta`` times ``f_t`` over the increment. The same vector measures the increment and distributes the
    force, so the stiffness is symmetric: the shape functions at the point, :math:`\boldsymbol{N}` (the default), or,
    with ``interpolationLinearizedAboutCenter``, the particle's interpolation linearized about its center,
    :math:`\boldsymbol{G}`, its own view of the point.

    **In an explicit analysis**, the nodes of the kernel functions carry no increment to constrain: the particles have
    moved when the constraints are evaluated. There the constraint holds displacement components at zero, as a
    support, e.g. on a symmetry plane or at a clamped edge, and it measures them where the constrained points are now:
    the force on a constrained point is

    .. math::

        f_i = -k \left( x_i - X_i \right)

    for every constrained component :math:`i`, with :math:`\boldsymbol{X}` the position of the point when the
    constraint was created. The point (the center or a vertex of the particle) is where the particle has put it: it
    moves with the particle, in an increment by :math:`\boldsymbol{G} \cdot \Delta\boldsymbol{U}`. By default the
    force is distributed over the nodes of the kernel functions with the shape functions at the point,
    :math:`\boldsymbol{N}`, which away from the center is not :math:`\boldsymbol{G}`: the penalty stiffness
    :math:`k\,\boldsymbol{N}\boldsymbol{G}^\mathsf{T}` is unsymmetric, does work over a closed cycle of motion, and can
    feed energy into the material next to the support once that has softened. With
    ``interpolationLinearizedAboutCenter`` it is distributed with :math:`\boldsymbol{G}`, the transpose of the point's
    motion: the stiffness :math:`k\,\boldsymbol{G}\boldsymbol{G}^\mathsf{T}` is symmetric and the support conservative
    (:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`).
    The force acts on both sides of :math:`X_i`. It acts in an explicit analysis
    (:attr:`actsInExplicitAnalysis`) only if it constrains the displacement and every prescribed change is zero: an
    explicit analysis prescribes motion through velocities, not through penalized displacements.

    A support holds only as well as its penalty is stiff against the material next to it: a penalty that is merely
    comparable to the stiffness of a particle lets the support give way and spring back, an error that does not shrink
    with the particle size. A stiffer one costs time increments: the explicit solver adds its springs to the
    frequencies of the nodes they hold (:meth:`addBoundOfAbsoluteStiffnessTimes`), and a support far stiffer than
    the particles sets the critical time step, about :math:`2 \sqrt{m / k}` with :math:`m` the mass a constrained point
    carries; see example 186.

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
        If "center", the center of the particle is constrained. If a list, the vertices at the given indices are
        constrained; it must name at least one vertex.
    interpolationLinearizedAboutCenter
        False (default): measure the field at the point and distribute the force there with the shape functions at
        the point, N. True: with the particle's interpolation linearized about its center, G, with which the particle
        moves its own points (symmetric and conservative). See
        :meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`.
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
        interpolationLinearizedAboutCenter: bool = False,
        **kwargs,
    ):
        self._name = name
        self.interpolationLinearizedAboutCenter = interpolationLinearizedAboutCenter
        self._model = model
        self._constrainedParticles = list(constrainedParticles)
        self._field = field
        self._prescribedStepDelta = prescribedStepDelta
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._penaltyParameter = penaltyParameter
        self._nodes = dict()
        #: Per constrained particle, where the nodes of its kernel functions sit in the constraint's block (see
        #: :meth:`updateConnectivity`).
        self._localNodesOfParticle = []
        #: Per constrained particle, its kernel functions as of the last :meth:`updateConnectivity`.
        self._kernelFunctionsOfParticle = []
        if constrain == "center":
            self._constrainVertices = None
        else:
            # an empty list would constrain nothing, and is not taken for "center" either
            if not isinstance(constrain, list) or not constrain or not all(isinstance(i, int) for i in constrain):
                raise ValueError("Constrain must be 'center' or a list of vertex indices that is not empty.")
            self._constrainVertices = constrain

        self.penaltyForce = np.zeros(self._fieldSize)

        #: Where the constrained points of each particle are when the constraint is created, the positions an explicit
        #: analysis holds them at.
        self._initialCoordinatesOfConstrainedPoints = [
            self._constrainedCoordinatesOf(particle) for particle in self._constrainedParticles
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
        """Collect the nodes of the kernel functions of the constrained particles, each once, ordered by label, and
        record where each particle's nodes sit in the constraint's block
        (:func:`~edelweissmeshfree.constraints.base.pointsofparticles.locateTheNodesOfParticles`). The nodes stay as
        they are, and nothing is collected, if no constrained particle has other kernel functions than at the last
        update. The constraint
        compares the particles' lists of kernel functions with those of its last update itself, instead of reading the
        particle managers' report (``particlesWithChangedKernelFunctions``), which covers their last search only: the
        comparison holds whether or not the constraint was updated after every search.

        Parameters
        ----------
        model
            The model; unused, the particles hold their kernel functions.

        Returns
        -------
        bool
            True if the nodes of the constraint changed.
        """

        kernelFunctionsOfParticle = [particle.kernelFunctions for particle in self._constrainedParticles]
        if self._nodes and kernelFunctionsOfParticle == self._kernelFunctionsOfParticle:
            return False
        self._kernelFunctionsOfParticle = kernelFunctionsOfParticle

        # the nodes ordered by label, so that the block, and with it the numbering of the model, does not depend on the
        # order of the particles
        nodesByLabel = sorted({kf.node for kfs in kernelFunctionsOfParticle for kf in kfs}, key=lambda node: node.label)
        nodes = {node: index for index, node in enumerate(nodesByLabel)}
        self._localNodesOfParticle = locateTheNodesOfParticles(kernelFunctionsOfParticle, nodes)
        hasChanged = nodes != self._nodes
        self._nodes = nodes

        return hasChanged

    def computeKernels(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, K: np.ndarray, timeStep: TimeStep):
        for i, prescribedComponent in self._prescribedStepDelta.items():
            P_i = P[i :: self._fieldSize]
            dU_j = dU[i :: self._fieldSize]

            K_ij = K.reshape((self.nDof, self.nDof))[i :: self._fieldSize, i :: self._fieldSize]

            for p, nodeIdcs in zip(self._constrainedParticles, self._localNodesOfParticle):
                for constrainedCoordinate in self._constrainedCoordinatesOf(p):

                    interpolation = self.interpolationVectorAtPointOf(p, constrainedCoordinate)

                    mpValue = interpolation @ dU_j[nodeIdcs]

                    P_i[nodeIdcs] += (
                        interpolation
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
                    K_ij[np.ix_(nodeIdcs, nodeIdcs)] += np.outer(interpolation, interpolation) * self._penaltyParameter

            self.penaltyForce[i] = np.sum(P_i)

    def computeKernelsExplicit(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, timeStep: TimeStep):
        """Add the residual contribution of the support in an explicit analysis: the negative of the forces that hold
        the constrained components of the constrained points at their initial positions, measured where the points are
        now (see the class documentation), from its springs (:meth:`_springsOfEachParticle`).

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

        PAtNodes = P.reshape(-1, self._fieldSize)
        for localNodes, springs in self._springsOfEachParticle():
            addForcesOfSpringsOnPointsOfParticle(PAtNodes, localNodes, springs)

        for i in self._prescribedStepDelta:
            self.penaltyForce[i] = np.sum(P[i :: self._fieldSize])

    def addBoundOfAbsoluteStiffnessTimes(
        self,
        U_np: np.ndarray,
        dU: np.ndarray,
        weights: np.ndarray,
        bound: np.ndarray,
        timeStep: TimeStep,
    ):
        """Add the row sums of the absolute value of the stiffness of the support in an explicit analysis, weighted by
        column, from the same springs as its force (:meth:`_springsOfEachParticle`,
        :func:`~edelweissmeshfree.constraints.base.pointsofparticles.addBoundOfAbsoluteStiffnessOfPointsOfParticle`).

        Parameters
        ----------
        U_np
            The current solution; unused.
        dU
            The solution increment; unused.
        weights
            The weight of each column.
        bound
            The row sums to augment.
        timeStep
            The current time step; unused.
        """

        for localNodes, springs in self._springsOfEachParticle():
            addBoundOfAbsoluteStiffnessOfPointsOfParticle(bound, weights, localNodes, springs)

    def _springsOfEachParticle(self) -> list[tuple[np.ndarray, SpringsOnPointsOfParticle]]:
        """The springs of the support in an explicit analysis, per constrained particle and per constrained component
        :math:`i`: one of the penalty stiffness :math:`k` at each constrained point, along the axis :math:`i`, whose gap
        is the displacement of the point from where it is held, :math:`g_p = x_{pi} - X_{pi}`, and whose force
        :math:`f_p = -k\\, g_p` pulls it back; with the vector the force is distributed with
        (:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`).
        What the force (:meth:`computeKernelsExplicit`) and the bound of the stiffness
        (:meth:`addBoundOfAbsoluteStiffnessTimes`) are made from.

        Returns
        -------
        list[tuple[numpy.ndarray, SpringsOnPointsOfParticle]]
            Where the nodes of the particle's kernel functions sit in the block, and the springs.
        """

        springsOfEachParticle = []
        for particle, localNodes, initialCoordinates in zip(
            self._constrainedParticles, self._localNodesOfParticle, self._initialCoordinatesOfConstrainedPoints
        ):
            coordinates = self._constrainedCoordinatesOf(particle)
            interpolations = np.array([self.interpolationVectorAtPointOf(particle, x) for x in coordinates]).reshape(
                (len(coordinates), len(localNodes))
            )
            stiffness = np.full(len(coordinates), self._penaltyParameter)
            for i in self._prescribedStepDelta:
                directions = np.zeros((len(coordinates), self._fieldSize))
                directions[:, i] = 1.0
                gaps = coordinates[:, i] - initialCoordinates[:, i]
                springsOfEachParticle.append(
                    (
                        localNodes,
                        SpringsOnPointsOfParticle(
                            interpolations, directions, -self._penaltyParameter * gaps, stiffness
                        ),
                    )
                )
        return springsOfEachParticle

    def _constrainedCoordinatesOf(self, particle: BaseParticle) -> np.ndarray:
        """The current coordinates of the constrained points of a particle: its center, or the constrained vertices.

        Parameters
        ----------
        particle
            The particle.

        Returns
        -------
        np.ndarray
            The coordinates, one row per point.
        """

        if self._constrainVertices:
            return particle.getVertexCoordinates()[self._constrainVertices]
        return particle.getCenterCoordinates().copy()[np.newaxis, :]

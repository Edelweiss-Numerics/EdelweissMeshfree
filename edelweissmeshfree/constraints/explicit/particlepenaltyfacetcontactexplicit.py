"""Penalty contact between the vertices of particles and a faceted surface, for explicit analyses.

The faceted surface (the *master*) is the boundary of a body made of finite elements, tiled into flat contact facets by
EdelweissFE's surface element generator (:mod:`edelweissfe.generators.surfaceelementgenerator`): Line2 facets in 2D,
Tria3 facets in 3D. The vertices of the particles are the contact points (the *slave* points). The geometry -- the
closest point on a facet, the facet normal, the search for the closest facet -- is EdelweissFE's
(:mod:`edelweissfe.utils.facetcontactgeometry`), the same as that of its node-to-surface contact
(:mod:`edelweissfe.constraints.nodetodeformablesurfacepenalty`).
"""

from enum import IntEnum

import numpy as np
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.facetcontactgeometry import (
    closestFacetCandidates,
    closestFacets,
    facetNormalAndMeasure,
    rowDot,
)
from edelweissfe.utils.meshtools import currentNodeCoordinates

from edelweissmeshfree.constraints.base.mpmexplicitconstraintbase import (
    MPMExplicitConstraintBase,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ContactState(IntEnum):
    """Where a contact point stands with respect to the master surface, as of the last update of the connectivity."""

    #: No facet within the search distance.
    FREE = 0
    #: A facet within the search distance, the point outside of it: the point is seen before it touches.
    APPROACHING = 1
    #: The point has entered the master body through its paired facet, and is pushed back out through it.
    IN_CONTACT = 2
    #: The point lies inside the master body without having been seen outside of it before: it has tunnelled through
    #: the search distance in one increment, or the bodies overlapped from the start. It gets no force (the closest
    #: facet is no indication of the way out), and the constraint reports it.
    FOUND_INSIDE = 3


class ParticlePenaltyFacetContactExplicit(MPMExplicitConstraintBase):
    r"""Penalty contact between the vertices of a set of particles and a faceted master surface, for explicit
    analyses.

    **The gap and the force.** A contact point :math:`\boldsymbol{x}_s` (a vertex of a particle) is paired with a facet
    of the master surface. Its closest point on the facet has the weights :math:`w_a` of the facet's nodes
    :math:`\boldsymbol{x}_a` -- non-negative, and summing to one, because the projection is clamped to the facet --
    and the gap is measured along the facet's outward unit normal :math:`\boldsymbol{n}`,

    .. math::

        g = \boldsymbol{n} \cdot \Big( \boldsymbol{x}_s - \sum_a w_a\, \boldsymbol{x}_a \Big),

    negative where the point has entered the master body. There the penalty spring pushes the point back out, with
    the force :math:`\boldsymbol{f} = -k\, g\, \boldsymbol{n}` on the point, which the particle distributes over the
    nodes of its kernel functions with its shape functions :math:`N_I` at the point. The reaction
    :math:`-\boldsymbol{f}` acts on the facet, distributed over its nodes with the weights :math:`w_a` (transformed by
    the facet's weight transform, if the surface element generator gave it one). Both the shape functions and the
    weights sum to one, so the two forces are equal and opposite, and the contact exchanges momentum without creating
    any. ``penaltyParameter`` :math:`k` is a force per unit penetration of one contact point, as in the other penalty
    contacts of the particles (e.g.
    :class:`~edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit.ParticlePenaltyCartesianBoundaryConstraintExplicit`);
    EdelweissFE's node-to-surface contact takes a penalty per unit area instead, which is :math:`k` divided by the
    area a contact point stands for.

    **The pairing.** In every update of the connectivity, the contact points within the search distance of the
    surface are paired with a facet: a point with its closest facet, and with the weights of its closest point there,
    but only while it is outside the master body. Once it is in contact, it keeps the facet it entered through, and moves
    on only to a facet that shares a node with it (its *patch*), as it slides along the surface. This is what keeps a
    deep contact point from being pushed out of the wrong side: the closest facet of a point deep inside a body
    need not be the one it came in through, and its normal then points the wrong way. A point is admitted to contact
    only if it has been seen outside within the search distance before (:attr:`ContactState.APPROACHING`); a point
    that is first found inside (:attr:`ContactState.FOUND_INSIDE`) gets no force and is reported in the journal. The
    search distance must therefore exceed the distance a point travels towards the surface in one increment; the
    constraint warns when a point enters deeper than ``entryDepthWarningFraction`` of it. A point in contact never
    leaves the search, however deep it is.

    **Where it is evaluated.** The pairing (the facet and the weights) holds from one update of the connectivity to the
    next; the gap is evaluated where the bodies are when the forces are: the contact points where the particles are
    now, the facet where its nodes are now -- with ``twoWayCoupling`` from their total displacement in the solution,
    without it from the model's node field ``displacement`` -- with its normal there. An explicit analysis that
    updates the connectivity in every increment, as one with particles does, evaluates at the configuration of the
    update; one that updates it less often still evaluates the gaps of the current configuration, and a paired point
    that penetrates between two updates is pushed back as well.

    **The nodes.** The constraint takes part in the discrete system while a point is paired
    (:attr:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.withinReach`). Its nodes are those of the
    kernel functions of the particles with a paired point and, with ``twoWayCoupling``, the nodes of the facets they
    are paired with. Without ``twoWayCoupling``, the master surface is moved by something else -- it is rigid, or its
    motion is prescribed -- and the reaction is not applied; its nodes then need no degrees of freedom.

    **What it reports**, after each evaluation: the contact force on the particles and on the master surface
    (:attr:`contactForceOnParticles`, :attr:`contactForceOnMaster`), whose sum is zero to round-off; the energy stored
    in the penalty springs (:attr:`penaltyEnergy`), for an energy balance; the deepest penetration
    (:attr:`deepestPenetration`); and how many points are in contact or were found inside
    (:attr:`numberOfPointsInContact`, :attr:`numberOfPointsFoundInside`).

    The pairing is checked for a master surface that is convex where it is touched, such as the nose of a
    projectile: at a reentrant corner, the sign of the gap of the closest facet does not tell inside from outside.

    Parameters
    ----------
    name
        The name of the constraint.
    model
        The model.
    particles
        The particles whose vertices are the contact points.
    masterFacets
        The contact facet elements of the master surface (Line2 facets in 2D, Tria3 facets in 3D), e.g. the element
        set the surface element generator creates; their node ordering makes their normals point out of the master
        body.
    penaltyParameter
        The penalty stiffness, a force per unit penetration of one contact point. It lowers the stable time
        increment of the explicit integration.
    searchDistance
        The distance from the master surface within which a point is seen, and may come into contact in the next
        increment. It must exceed the distance a point travels towards the surface in one increment.
    journal
        The journal, for the reports.
    twoWayCoupling
        Apply the reaction to the nodes of the master facets. Leave it off only for a master surface that is moved by
        something else, rigid or prescribed.
    entryDepthWarningFraction
        The fraction of the search distance beyond which the depth at which a point enters contact is reported: a
        point that enters that deep has almost outrun the search.
    """

    #: The pairing a point carries from one increment to the next: its state and its paired facet. The weights follow
    #: from the next update of the connectivity, which comes before the next evaluation.
    checkpointedState = {
        "_stateOfPoint": np.ndarray,
        "_pairedFacetOfPoint": np.ndarray,
    }

    def __init__(
        self,
        name: str,
        model: MPMModel,
        particles: list[BaseParticle],
        masterFacets: list,
        penaltyParameter: float,
        searchDistance: float,
        journal,
        twoWayCoupling: bool = True,
        entryDepthWarningFraction: float = 0.5,
    ):
        self._name = name
        self._model = model
        self._journal = journal
        self._domainSize = model.domainSize
        self._particles = list(particles)
        self._facets = list(masterFacets)
        self._penaltyParameter = penaltyParameter
        self._searchDistance = searchDistance
        self._twoWayCoupling = twoWayCoupling
        self._entryDepthWarningFraction = entryDepthWarningFraction

        if penaltyParameter <= 0.0:
            raise ValueError(f"Constraint '{name}': the penalty must be positive.")
        if searchDistance <= 0.0:
            raise ValueError(f"Constraint '{name}': the search distance must be positive.")
        if not self._particles:
            raise ValueError(f"Constraint '{name}': no particles were given.")
        if not self._facets:
            raise ValueError(f"Constraint '{name}': no master facets were given.")
        nodesPerFacet = {len(facet.nodes) for facet in self._facets}
        if nodesPerFacet != {self._domainSize}:
            raise ValueError(
                f"Constraint '{name}': the master facets of a {self._domainSize}D model have {self._domainSize} nodes "
                f"(Line2 facets in 2D, Tria3 facets in 3D), but the given ones have {sorted(nodesPerFacet)}."
            )

        # The master nodes, each once, and the facets as rows of indices into them.
        indexOfMasterNode = {}
        for facet in self._facets:
            for node in facet.nodes:
                indexOfMasterNode.setdefault(node, len(indexOfMasterNode))
        self._masterNodes = list(indexOfMasterNode)
        self._referenceCoordinatesOfMasterNodes = np.array([node.coordinates for node in self._masterNodes])
        self._masterNodesOfFacet = np.array(
            [[indexOfMasterNode[node] for node in facet.nodes] for facet in self._facets]
        )

        # The patch of a facet: the facets that share a node with it, itself included, in ascending order.
        facetsOfMasterNode = [[] for _ in self._masterNodes]
        for facetIndex, nodeIndices in enumerate(self._masterNodesOfFacet):
            for nodeIndex in nodeIndices:
                facetsOfMasterNode[nodeIndex].append(facetIndex)
        self._patchOfFacet = [
            sorted({neighbour for nodeIndex in nodeIndices for neighbour in facetsOfMasterNode[nodeIndex]})
            for nodeIndices in self._masterNodesOfFacet
        ]

        # The contact points are the vertices of the particles, particle after particle.
        verticesOfParticle = [len(particle.getVertexCoordinates()) for particle in self._particles]
        self._particleOfPoint = np.repeat(np.arange(len(self._particles)), verticesOfParticle)
        self._vertexOfPoint = np.concatenate([np.arange(nVertices) for nVertices in verticesOfParticle])
        nPoints = len(self._particleOfPoint)

        self._stateOfPoint = np.full(nPoints, ContactState.FREE, dtype=np.int8)
        self._pairedFacetOfPoint = np.full(nPoints, -1, dtype=np.intp)

        # The paired points of the last update of the connectivity, the weights of their closest points, and where the
        # nodes of their particles and facets are in the constraint's block.
        self._pairedPoints = np.zeros(0, dtype=np.intp)
        self._weightsOfPoint = np.zeros((nPoints, self._domainSize))
        self._localNodesOfParticle = {}
        self._localMasterNodesOfPoint = {}
        self._deepestEntryReported = 0.0

        self._nodes = []

        #: The contact force on the particles, summed over their nodes, at the last evaluation.
        self.contactForceOnParticles = np.zeros(self._domainSize)
        #: The contact force on the master surface, summed over its nodes, at the last evaluation; zero without
        #: ``twoWayCoupling``.
        self.contactForceOnMaster = np.zeros(self._domainSize)
        #: The energy stored in the penalty springs at the last evaluation, :math:`\sum \frac{1}{2} k g^2`.
        self.penaltyEnergy = 0.0
        #: The number of points in contact at the last evaluation.
        self.numberOfPointsInContact = 0
        #: The largest penetration of a point at the last evaluation, a positive length; zero without contact.
        self.deepestPenetration = 0.0

        self.withinReach = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fields(self) -> list:
        return [["displacement"]] * len(self._nodes)

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._domainSize

    @property
    def scalarVariables(self) -> list:
        return []

    @property
    def stateOfPoints(self) -> np.ndarray:
        """The :class:`ContactState` of every contact point, as of the last update of the connectivity, the vertices
        of the particles particle after particle.

        Returns
        -------
        numpy.ndarray
            The states.
        """

        return self._stateOfPoint

    @property
    def numberOfPointsFoundInside(self) -> int:
        """The number of contact points found inside the master body without having been seen outside of it, which
        get no force (:attr:`ContactState.FOUND_INSIDE`).

        Returns
        -------
        int
            The number.
        """

        return int(np.count_nonzero(self._stateOfPoint == ContactState.FOUND_INSIDE))

    def updateConnectivity(self, model: MPMModel) -> bool:
        """Pair the contact points with the facets of the master surface at the current configuration, and find the
        nodes the paired points act on (see the class documentation).

        Parameters
        ----------
        model
            The model, whose node field ``displacement`` moves the master surface.

        Returns
        -------
        bool
            True if the nodes of the constraint changed.
        """

        coordinatesOfPoint = np.concatenate([particle.getVertexCoordinates() for particle in self._particles])
        coordinatesOfMasterNodes = currentNodeCoordinates(
            self._masterNodes, model, self._referenceCoordinatesOfMasterNodes
        )
        coordinatesOfFacet = coordinatesOfMasterNodes[self._masterNodesOfFacet]
        normalOfFacet = np.array([facetNormalAndMeasure(coordinates)[0] for coordinates in coordinatesOfFacet])

        previousState = self._stateOfPoint
        wasInContact = previousState == ContactState.IN_CONTACT
        wasSeen = wasInContact | (previousState == ContactState.APPROACHING)

        # A point in contact looks for its closest facet in the patch of its paired facet, every other point
        # everywhere within the search distance.
        candidatesOfPoint = [[] for _ in coordinatesOfPoint]
        for point in np.flatnonzero(wasInContact):
            candidatesOfPoint[point] = self._patchOfFacet[self._pairedFacetOfPoint[point]]
        searched = np.flatnonzero(~wasInContact)
        if len(searched):
            for point, candidates in zip(
                searched,
                closestFacetCandidates(coordinatesOfPoint[searched], list(coordinatesOfFacet), self._searchDistance),
            ):
                candidatesOfPoint[point] = candidates
        closestFacet, weights, distance = closestFacets(coordinatesOfPoint, list(coordinatesOfFacet), candidatesOfPoint)

        reached = (closestFacet >= 0) & (wasInContact | (distance <= self._searchDistance))

        gap = np.zeros(len(coordinatesOfPoint))
        facetOfReached = closestFacet[reached]
        closestPoint = np.einsum("pa,pad->pd", weights[reached], coordinatesOfFacet[facetOfReached])
        gap[reached] = rowDot(normalOfFacet[facetOfReached], coordinatesOfPoint[reached] - closestPoint)

        penetrating = reached & (gap < 0.0)
        state = np.full(len(coordinatesOfPoint), ContactState.FREE, dtype=np.int8)
        state[reached & ~penetrating] = ContactState.APPROACHING
        state[penetrating & wasSeen] = ContactState.IN_CONTACT
        state[penetrating & ~wasSeen] = ContactState.FOUND_INSIDE

        self._reportEntriesAndPointsFoundInside(previousState, state, gap)

        self._stateOfPoint = state
        paired = (state == ContactState.APPROACHING) | (state == ContactState.IN_CONTACT)
        self._pairedFacetOfPoint = np.where(paired, closestFacet, -1)
        self._weightsOfPoint = weights
        self._pairedPoints = np.flatnonzero(paired)

        self.withinReach = bool(len(self._pairedPoints))
        if not self.withinReach:
            self.contactForceOnParticles = np.zeros(self._domainSize)
            self.contactForceOnMaster = np.zeros(self._domainSize)
            self.penaltyEnergy = 0.0
            self.numberOfPointsInContact = 0
            self.deepestPenetration = 0.0

        nodes = self._collectNodesOfPairedPoints()
        hasChanged = nodes != self._nodes
        self._nodes = nodes

        return hasChanged

    def _reportEntriesAndPointsFoundInside(self, previousState: np.ndarray, state: np.ndarray, gap: np.ndarray):
        """Report in the journal the points newly found inside the master body, and a point that enters contact
        deeper than any reported before, beyond ``entryDepthWarningFraction`` of the search distance.

        Parameters
        ----------
        previousState
            The states before this update.
        state
            The states after it.
        gap
            The gaps after it.
        """

        newlyFoundInside = (state == ContactState.FOUND_INSIDE) & (previousState != ContactState.FOUND_INSIDE)
        if newlyFoundInside.any():
            self._journal.message(
                f"{np.count_nonzero(newlyFoundInside)} contact point(s) found inside the master body without having "
                f"been seen outside of it within the search distance ({self._searchDistance:.3e}); they get no force. "
                "Increase the search distance, or reduce the time increment.",
                self._name,
                level=1,
            )

        entering = (state == ContactState.IN_CONTACT) & (previousState == ContactState.APPROACHING)
        if entering.any():
            entryDepth = float(-gap[entering].min())
            if (
                entryDepth > self._entryDepthWarningFraction * self._searchDistance
                and entryDepth > self._deepestEntryReported
            ):
                self._deepestEntryReported = entryDepth
                self._journal.message(
                    f"a contact point entered {entryDepth:.3e} deep in one increment, more than "
                    f"{self._entryDepthWarningFraction:.0%} of the search distance ({self._searchDistance:.3e}): "
                    "points may soon tunnel through it unseen.",
                    self._name,
                    level=1,
                )

    def _collectNodesOfPairedPoints(self) -> list:
        """The nodes the paired points act on, each once: those of the kernel functions of their particles, then, with
        ``twoWayCoupling``, those of their facets. Records where each particle's and each point's nodes sit in the
        constraint's block.

        Returns
        -------
        list
            The nodes.
        """

        localIndexOfNode = {}
        self._localNodesOfParticle = {}
        for particleIndex in np.unique(self._particleOfPoint[self._pairedPoints]):
            particle = self._particles[particleIndex]
            self._localNodesOfParticle[particleIndex] = np.array(
                [localIndexOfNode.setdefault(kf.node, len(localIndexOfNode)) for kf in particle.kernelFunctions]
            )

        self._localMasterNodesOfPoint = {}
        if self._twoWayCoupling:
            for point in self._pairedPoints:
                facet = self._facets[self._pairedFacetOfPoint[point]]
                self._localMasterNodesOfPoint[point] = np.array(
                    [localIndexOfNode.setdefault(node, len(localIndexOfNode)) for node in facet.nodes]
                )

        return list(localIndexOfNode)

    def computeKernelsExplicit(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, timeStep: TimeStep):
        """Add the residual contribution of the penalty forces of the paired points that penetrate their facets now:
        the negative of the force on the nodes of the particles and, with ``twoWayCoupling``, the negative of the
        reaction on the nodes of the facets.

        Parameters
        ----------
        U_np
            The current solution: with ``twoWayCoupling``, the total displacement of the nodes of the paired facets
            places them; the nodes of the kernel functions carry none, the particles are where they are.
        dU
            The solution increment; unused.
        P
            The residual contribution to augment.
        timeStep
            The current time step; unused.
        """

        pairedPoints = self._pairedPoints
        coordinatesOfPoint, coordinatesOfFacet = self._currentCoordinatesOfPairedPoints(U_np)
        normalOfFacet = np.array([facetNormalAndMeasure(coordinates)[0] for coordinates in coordinatesOfFacet])
        weights = self._weightsOfPoint[pairedPoints]
        closestPoint = np.einsum("pa,pad->pd", weights, coordinatesOfFacet)
        gap = rowDot(normalOfFacet, coordinatesOfPoint - closestPoint)

        PAtNodes = P.reshape(-1, self._domainSize)
        forceOnParticles = np.zeros(self._domainSize)
        forceOnMaster = np.zeros(self._domainSize)

        penetrating = np.flatnonzero(gap < 0.0)
        for i in penetrating:
            point = pairedPoints[i]
            force = -self._penaltyParameter * gap[i] * normalOfFacet[i]

            particleIndex = self._particleOfPoint[point]
            N = self._particles[particleIndex].getInterpolationVector(coordinatesOfPoint[i]).flatten()
            # the kernel functions of a particle are on distinct nodes, so fancy indexing does not drop any
            PAtNodes[self._localNodesOfParticle[particleIndex]] -= np.outer(N, force)
            forceOnParticles += N.sum() * force

            if self._twoWayCoupling:
                facet = self._facets[self._pairedFacetOfPoint[point]]
                weightsOnFacetNodes = (
                    weights[i] if facet.weightTransform is None else facet.weightTransform @ weights[i]
                )
                PAtNodes[self._localMasterNodesOfPoint[point]] += np.outer(weightsOnFacetNodes, force)
                forceOnMaster -= weightsOnFacetNodes.sum() * force

        self.contactForceOnParticles = forceOnParticles
        self.contactForceOnMaster = forceOnMaster
        self.penaltyEnergy = float(0.5 * self._penaltyParameter * np.sum(gap[penetrating] ** 2))
        self.numberOfPointsInContact = len(penetrating)
        self.deepestPenetration = float(-gap[penetrating].min()) if len(penetrating) else 0.0

    def _currentCoordinatesOfPairedPoints(self, U_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Where the paired points and the nodes of their facets are now.

        Parameters
        ----------
        U_np
            The current solution of the constraint's block.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            The coordinates of the paired points, of shape ``(nPaired, domainSize)``, and those of the nodes of their
            facets, of shape ``(nPaired, nNodesPerFacet, domainSize)``.
        """

        pairedPoints = self._pairedPoints
        verticesOfParticle = {
            particleIndex: self._particles[particleIndex].getVertexCoordinates().copy()
            for particleIndex in self._localNodesOfParticle
        }
        coordinatesOfPoint = np.array(
            [verticesOfParticle[self._particleOfPoint[point]][self._vertexOfPoint[point]] for point in pairedPoints]
        ).reshape(-1, self._domainSize)

        masterNodesOfPoint = self._masterNodesOfFacet[self._pairedFacetOfPoint[pairedPoints]]
        if self._twoWayCoupling:
            UAtNodes = U_np.reshape(-1, self._domainSize)
            displacementOfFacetNodes = np.array(
                [UAtNodes[self._localMasterNodesOfPoint[point]] for point in pairedPoints]
            ).reshape(masterNodesOfPoint.shape + (self._domainSize,))
            coordinatesOfFacet = self._referenceCoordinatesOfMasterNodes[masterNodesOfPoint] + displacementOfFacetNodes
        else:
            coordinatesOfMasterNodes = currentNodeCoordinates(
                self._masterNodes, self._model, self._referenceCoordinatesOfMasterNodes
            )
            coordinatesOfFacet = coordinatesOfMasterNodes[masterNodesOfPoint]

        return coordinatesOfPoint, coordinatesOfFacet

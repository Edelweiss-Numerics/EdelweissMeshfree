# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _
# | ____|__| | ___| |_      _____(_)___ ___
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/
# |  \/  | ___  ___| |__  / _|_ __ ___  ___
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \
# | |  | |  __/\__ \ | | |  _| | |  __/  __/
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
#
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------

from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


def quadraticEdgeShapeFunctions(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The shape functions of a quadratic edge -- corner node, midside node, corner node -- and their derivatives, at
    points :math:`\\xi \\in [-1, 1]` of the edge.

    Parameters
    ----------
    xi
        The points on the edge.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``N`` and ``dN_dXi``, each of shape ``(len(xi), 3)``.
    """

    N = np.column_stack((0.5 * xi * (xi - 1.0), 1.0 - xi**2, 0.5 * xi * (xi + 1.0)))
    dN_dXi = np.column_stack((xi - 0.5, -2.0 * xi, xi + 0.5))

    return N, dN_dXi


def isQuadraticEdge(corner: np.ndarray, midside: np.ndarray, otherCorner: np.ndarray) -> bool:
    """Whether three points, in this order, can be the nodes of a quadratic element edge: the midside node lies within
    a quarter of the distance between the corners from their midpoint. A midside node further along the line between
    the corners folds the edge onto itself (the edge mapping is no longer monotonic), one further away from that line
    bends the edge by more than a right angle between its corners.

    Parameters
    ----------
    corner
        The first corner node.
    midside
        The midside node.
    otherCorner
        The other corner node.

    Returns
    -------
    bool
        True if the points can be a quadratic edge in this order.
    """

    return bool(np.linalg.norm(midside - 0.5 * (corner + otherCorner)) < 0.25 * np.linalg.norm(otherCorner - corner))


def turnsBack(edge: np.ndarray, nextEdge: np.ndarray) -> bool:
    """Whether a quadratic edge turns back at its end, where the next edge starts: whether the two directions there
    enclose more than a right angle. An interface may have corners, but it does not go back on itself.

    Parameters
    ----------
    edge
        The corner, midside and corner node of the edge.
    nextEdge
        The corner, midside and corner node of the next edge, which starts at the last node of ``edge``.

    Returns
    -------
    bool
        True if the interface turns back.
    """

    _, dN_dXi = quadraticEdgeShapeFunctions(np.array([1.0, -1.0]))
    dx_dXi_atTheEnd = dN_dXi[0] @ edge
    dx_dXi_atTheStartOfTheNext = dN_dXi[1] @ nextEdge

    # a corner of exactly a right angle is no turn back, whatever the round-off of the coordinates
    return bool(
        dx_dXi_atTheEnd @ dx_dXi_atTheStartOfTheNext
        < -1e-12 * np.linalg.norm(dx_dXi_atTheEnd) * np.linalg.norm(dx_dXi_atTheStartOfTheNext)
    )


def distanceToPolygon(point: np.ndarray, vertices: np.ndarray) -> float:
    """The distance of a point to a convex polygon: zero inside, else the distance to the nearest edge.

    Parameters
    ----------
    point
        The point.
    vertices
        The vertices of the polygon, in order around it.

    Returns
    -------
    float
        The distance.
    """

    edgeVectors = np.roll(vertices, -1, axis=0) - vertices
    toPoint = point - vertices
    crossProducts = edgeVectors[:, 0] * toPoint[:, 1] - edgeVectors[:, 1] * toPoint[:, 0]
    if np.all(crossProducts >= 0.0) or np.all(crossProducts <= 0.0):
        return 0.0

    positionOnEdge = np.clip(np.sum(toPoint * edgeVectors, axis=1) / np.sum(edgeVectors**2, axis=1), 0.0, 1.0)
    closestPoints = vertices + positionOnEdge[:, None] * edgeVectors

    return float(np.min(np.linalg.norm(point - closestPoints, axis=1)))


class ParticleFiniteElementMortarTie(MPMConstraintBase):
    r"""Ties an RKPM domain to a finite element domain along a common interface, in two dimensions: the displacement of
    the particles and the displacement of the finite elements agree on the interface in the weak (mortar) sense.

    **What is tied.** On the interface :math:`\Gamma`, the finite element side interpolates the displacement with the
    shape functions :math:`N_a` of its quadratic element edges, the RKPM side with the reproducing kernel shape
    functions :math:`\Psi_K` of the particles there. The two do not match pointwise: the reproducing kernel shape
    functions are smooth and not interpolating, the element edges are piecewise polynomials. The tie therefore asks
    that their difference be orthogonal to the multiplier space -- the shape functions of the element edges, one
    multiplier per interface node and displacement component:

    .. math::

        g_{ai} = \int_\Gamma N_a \left( \sum_b N_b\, \Delta U_{bi} - \sum_K \Psi_K\, \Delta U_{Ki} \right) d\Gamma = 0.

    **Incremental, on the current configuration.** RKPM is updated-Lagrangian: its nodal values are the coefficients
    of one increment, on the configuration at the start of the increment, where the kernel functions sit with their
    particles. The tie therefore couples increments :math:`\Delta U`, and integrates on that configuration: the element
    edges are placed at their nodes' converged positions, :math:`\boldsymbol{x} = \boldsymbol{X} + \boldsymbol{U}`.
    Both sides start stress-free and tied, so the increments add up to a tied total displacement. During an increment
    the configuration is fixed, so :math:`g` is linear in :math:`\Delta U`, and its derivatives

    .. math::

        \frac{\partial g_{ai}}{\partial \Delta U_{bi}} = \int_\Gamma N_a N_b \, d\Gamma, \qquad
        \frac{\partial g_{ai}}{\partial \Delta U_{Ki}} = -\int_\Gamma N_a \Psi_K \, d\Gamma

    are computed once per increment, in :meth:`updateConnectivity`.

    **Forces.** The multiplier :math:`\lambda_{ai}` is the interface traction, interpolated like the displacement of
    the element edges. Unlike the increments it couples, it is a total quantity: the stresses on both sides are total,
    so is the force that the interface transmits between them. Following the rule of every constraint (see
    :class:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase`), the tie adds

    .. math::

        P_{bi} \mathrel{+}= \sum_a \lambda_{ai} \frac{\partial g_{ai}}{\partial \Delta U_{bi}}, \qquad
        P_{Ki} \mathrel{+}= \sum_a \lambda_{ai} \frac{\partial g_{ai}}{\partial \Delta U_{Ki}}, \qquad
        P_{\lambda_{ai}} \mathrel{+}= g_{ai},

    and the tangent, symmetric as the tie stems from the potential :math:`\sum_{ai} \lambda_{ai} g_{ai}`. The finite
    element side receives the force :math:`-\int_\Gamma N_b \lambda_i \, d\Gamma`, the RKPM side the opposite one.

    **The integral.** Gauss quadrature on every element edge, in the configuration at the start of the increment. The
    reproducing kernel shape functions at a quadrature point are those of the interface particle the point lies on.
    A field that is linear in space is represented exactly on both sides, so it is tied exactly, whatever the
    quadrature (the patch test).

    **Rules for the interface.**

    - The finite element side carries the multipliers. Its nodes must not be prescribed by a Dirichlet boundary
      condition as well, which would constrain them twice.
    - The finite element side is given by its interface nodes, in order along the interface: corner node, midside
      node, corner node, midside node, ..., corner node -- the edges of quadratic elements. The tie checks that the
      order is one along the interface: every midside node lies near the middle of its corners
      (:func:`isQuadraticEdge`), and the interface does not turn back at a corner node (:func:`turnsBack`). A node set
      in the order of its labels, for instance, is refused.
    - The RKPM side is given by its particles along the interface.

    Parameters
    ----------
    name
        The name of this constraint.
    model
        The model.
    interfaceParticles
        The particles of the RKPM domain along the interface.
    finiteElementInterfaceNodes
        The nodes of the finite element domain on the interface, in order along it: corner, midside, corner, ...
    quadraturePointsPerEdge
        The number of Gauss points on every element edge.
    """

    #: The traction is a scalar variable of the model, which the model's checkpoint holds; the integrals are recomputed
    #: at the start of every increment.
    checkpointedState = {}

    def __init__(
        self,
        name: str,
        model: MPMModel,
        interfaceParticles: Iterable[BaseParticle],
        finiteElementInterfaceNodes: Iterable[Node],
        quadraturePointsPerEdge: int = 8,
    ):
        self._name = name
        self._interfaceParticles = list(interfaceParticles)
        self._finiteElementNodes = list(finiteElementInterfaceNodes)

        if model.domainSize != 2:
            raise ValueError("Constraint '{:}': the tie is implemented for two dimensions.".format(name))
        if len(self._finiteElementNodes) < 3 or len(self._finiteElementNodes) % 2 != 1:
            raise ValueError(
                "Constraint '{:}': the finite element interface nodes must form quadratic edges: "
                "corner, midside, corner, ..., corner.".format(name)
            )
        if not self._interfaceParticles:
            raise ValueError("Constraint '{:}': there are no interface particles.".format(name))

        self._fieldSize = getFieldSize("displacement", model.domainSize)
        self._checkThatTheNodesAreInOrderAlongTheInterface()
        self._nLagrangianMultipliers = len(self._finiteElementNodes) * self._fieldSize
        self._xi, self._gaussWeights = np.polynomial.legendre.leggauss(quadraturePointsPerEdge)
        self._N_edge, self._dN_dXi_edge = quadraticEdgeShapeFunctions(self._xi)

        self._kernelNodes = []
        #: The derivative of the gap with respect to the increments at the finite element nodes, per component.
        self.dGap_dFiniteElementIncrement = np.zeros((len(self._finiteElementNodes), len(self._finiteElementNodes)))
        #: The derivative of the gap with respect to the increments at the kernel nodes, per component.
        self.dGap_dKernelIncrement = np.zeros((len(self._finiteElementNodes), 0))

    def _checkThatTheNodesAreInOrderAlongTheInterface(self):
        """Refuse finite element interface nodes that are not in order along the interface: corner, midside, corner,
        ... The tie integrates every triple of them as an element edge, so nodes in another order would tie the wrong
        edges without any other sign of it.

        Raises
        ------
        ValueError
            If a triple is no quadratic edge, or if the interface turns back at a corner node.
        """

        coordinates = np.array([node.coordinates[: self._fieldSize] for node in self._finiteElementNodes])
        labels = [node.label for node in self._finiteElementNodes]

        edges = [slice(first, first + 3) for first in range(0, len(coordinates) - 1, 2)]
        for edge in edges:
            if not isQuadraticEdge(*coordinates[edge]):
                raise ValueError(
                    "Constraint '{:}': the finite element interface nodes {:} are no quadratic element edge: the "
                    "midside node must lie near the middle of the corners. The interface nodes must be given in order "
                    "along the interface: corner, midside, corner, ..., corner; a node set in the order of its "
                    "labels is not in that order in general.".format(self._name, labels[edge])
                )
        for edge, nextEdge in zip(edges, edges[1:]):
            if turnsBack(coordinates[edge], coordinates[nextEdge]):
                raise ValueError(
                    "Constraint '{:}': the finite element interface turns back at node {:}, from the edge {:} to the "
                    "edge {:}. The interface nodes must be given in order along the interface: corner, midside, "
                    "corner, ..., corner.".format(self._name, labels[nextEdge][0], labels[edge], labels[nextEdge])
                )

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        """The finite element interface nodes, in their order along the interface, then the nodes of the kernel
        functions the interface particles interpolate from, by label."""
        return self._finiteElementNodes + self._kernelNodes

    @property
    def fields(self) -> list:
        return [["displacement"]] * len(self.nodes)

    @property
    def nDof(self) -> int:
        return len(self.nodes) * self._fieldSize + self._nLagrangianMultipliers

    @property
    def scalarVariables(self) -> list:
        return self._lagrangianMultipliers

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return self._nLagrangianMultipliers

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        self._lagrangianMultipliers = scalarVariables

    def _particleAt(self, point: np.ndarray, particleVertices: list) -> int:
        """The index of the interface particle a point lies on, or is closest to.

        Parameters
        ----------
        point
            The point.
        particleVertices
            The vertices of every interface particle.

        Returns
        -------
        int
            The index of the particle.
        """

        return int(np.argmin([distanceToPolygon(point, vertices) for vertices in particleVertices]))

    def updateConnectivity(self, model: MPMModel) -> bool:
        """Integrate the gap's derivatives on the configuration at the start of the increment, after the particle
        managers have assigned the kernel functions to the particles. The kernel nodes are those the interface
        particles at the quadrature points interpolate from.

        Parameters
        ----------
        model
            The model, whose displacement field holds the converged total displacement of the finite element nodes.

        Returns
        -------
        bool
            True if the kernel nodes of the tie changed.
        """

        displacementField = model.nodeFields["displacement"]
        convergedPositions = np.array(
            [
                node.coordinates[: self._fieldSize] + displacementField["U"][displacementField.indexOfNode(node)]
                for node in self._finiteElementNodes
            ]
        )
        particleVertices = [np.array(p.getVertexCoordinates()) for p in self._interfaceParticles]

        quadraturePoints = []
        for firstNodeOfEdge in range(0, len(self._finiteElementNodes) - 1, 2):
            edgeNodes = slice(firstNodeOfEdge, firstNodeOfEdge + 3)
            x_q = self._N_edge @ convergedPositions[edgeNodes]
            dx_dXi = self._dN_dXi_edge @ convergedPositions[edgeNodes]
            lengthMeasure = self._gaussWeights * np.linalg.norm(dx_dXi, axis=1)
            for q in range(len(self._xi)):
                particle = self._interfaceParticles[self._particleAt(x_q[q], particleVertices)]
                quadraturePoints.append(
                    (edgeNodes, self._N_edge[q], lengthMeasure[q], particle, np.ascontiguousarray(x_q[q]))
                )

        kernelNodes = sorted(
            dict.fromkeys(kf.node for *_, particle, _ in quadraturePoints for kf in particle.kernelFunctions),
            key=lambda node: node.label,
        )
        indexOfKernelNode = {node: i for i, node in enumerate(kernelNodes)}

        nFiniteElementNodes = len(self._finiteElementNodes)
        dGap_dFiniteElementIncrement = np.zeros((nFiniteElementNodes, nFiniteElementNodes))
        dGap_dKernelIncrement = np.zeros((nFiniteElementNodes, len(kernelNodes)))

        for edgeNodes, N, lengthMeasure, particle, x in quadraturePoints:
            Psi = particle.getInterpolationVector(x)
            kernelIndices = [indexOfKernelNode[kf.node] for kf in particle.kernelFunctions]
            dGap_dFiniteElementIncrement[edgeNodes, edgeNodes] += lengthMeasure * np.outer(N, N)
            dGap_dKernelIncrement[edgeNodes, kernelIndices] -= lengthMeasure * np.outer(N, Psi)

        hasChanged = kernelNodes != self._kernelNodes
        self._kernelNodes = kernelNodes
        self.dGap_dFiniteElementIncrement = dGap_dFiniteElementIncrement
        self.dGap_dKernelIncrement = dGap_dKernelIncrement

        return hasChanged

    def computeKernels(self, U_np: np.ndarray, dU: np.ndarray, P: np.ndarray, K: np.ndarray, timeStep: TimeStep):
        """Add the tie's residual contribution and its tangent; see :class:`ParticleFiniteElementMortarTie`.

        Parameters
        ----------
        U_np
            The current solution; its last entries are the multipliers, the interface tractions.
        dU
            The increment since the last converged state; the tie couples the increments at the nodes.
        P
            The residual contribution to augment.
        K
            The tangent to augment.
        timeStep
            The time step; unused.
        """

        nL = self._nLagrangianMultipliers
        fieldSize = self._fieldSize

        dGap_dIncrement = np.hstack((self.dGap_dFiniteElementIncrement, self.dGap_dKernelIncrement))

        dU_nodes = dU[:-nL]
        P_nodes = P[:-nL]
        P_multipliers = P[-nL:]
        tractions = U_np[-nL:]

        K = K.reshape((self.nDof, self.nDof))
        K_nodesMultipliers = K[:-nL, -nL:]
        K_multipliersNodes = K[-nL:, :-nL]

        for i in range(fieldSize):
            traction_i = tractions[i::fieldSize]

            P_multipliers[i::fieldSize] += dGap_dIncrement @ dU_nodes[i::fieldSize]
            P_nodes[i::fieldSize] += dGap_dIncrement.T @ traction_i

            K_nodesMultipliers[i::fieldSize, i::fieldSize] += dGap_dIncrement.T
            K_multipliersNodes[i::fieldSize, i::fieldSize] += dGap_dIncrement

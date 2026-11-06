# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
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
"""Helpers for checking the constraints of EdelweissMeshfree against the rule every constraint follows: it adds a
residual contribution ``P`` and the tangent ``K = dP/dU`` (see the EdelweissFE solver documentation, "The residual and
the tangent").

The check itself is EdelweissFE's, :mod:`edelweissfe.utils.residualtangentcheck`. This module adds what the
constraints of EdelweissMeshfree need besides: stand-ins for the particles, cells and material points they act on, so
that a check needs no Marmot.
"""

import numpy as np


def interpolationVector(nodes: list, coordinates: np.ndarray) -> np.ndarray:
    """A smooth, positive partition of unity over some nodes, evaluated at a point: the values of shape functions
    that are fixed during an increment, which is all a constraint asks of them.

    Parameters
    ----------
    nodes
        The nodes.
    coordinates
        The point.

    Returns
    -------
    np.ndarray
        One value per node, summing to one.
    """

    distances = np.linalg.norm(np.array([node.coordinates for node in nodes]) - coordinates, axis=1)
    weights = 1.0 / (1.0 + distances**2)
    return weights / weights.sum()


class KernelFunctionStandIn:
    """What a constraint asks of a kernel function: its node."""

    def __init__(self, node):
        self.node = node


class ParticleStandIn:
    """What a constraint asks of a particle: its kernel functions, some points on it, and its shape functions.

    Parameters
    ----------
    nodes
        The nodes of the particle's kernel functions.
    center
        The center.
    vertices
        The vertices.
    volume
        The undeformed volume.
    """

    def __init__(self, nodes: list, center, vertices, volume: float = 1.0):
        self._nodes = nodes
        self.kernelFunctions = [KernelFunctionStandIn(node) for node in nodes]
        self.center = np.asarray(center, dtype=float)
        self.vertices = np.asarray(vertices, dtype=float)
        self._volume = volume

    def getCenterCoordinates(self) -> np.ndarray:
        return self.center

    def getVertexCoordinates(self) -> np.ndarray:
        return self.vertices

    def getFaceCoordinates(self, faceID: int) -> np.ndarray:
        return 0.5 * (self.vertices[faceID - 1] + self.vertices[faceID % len(self.vertices)])

    def getVolumeUndeformed(self) -> float:
        return self._volume

    def getInterpolationVector(self, coordinates: np.ndarray) -> np.ndarray:
        return interpolationVector(self._nodes, coordinates)


class CellStandIn:
    """What a constraint asks of a cell: its nodes and its shape functions."""

    def __init__(self, nodes: list):
        self.nodes = nodes

    def getInterpolationVector(self, coordinates: np.ndarray) -> np.ndarray:
        return interpolationVector(self.nodes, coordinates)


class MaterialPointStandIn:
    """What a constraint asks of a material point: its center and the cells it lies in."""

    def __init__(self, center, cells: list):
        self.center = np.asarray(center, dtype=float)
        self.assignedCells = cells

    def getCenterCoordinates(self) -> np.ndarray:
        return self.center

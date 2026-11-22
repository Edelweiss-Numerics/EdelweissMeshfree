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
"""Tests of the two operations the constraints on points of particles share
(:mod:`edelweissmeshfree.constraints.base.pointsofparticles`): where the nodes of the particles sit in the block of a
constraint, and how the forces on points of a particle reach them."""

import numpy as np
from edelweissfe.points.node import Node

from edelweissmeshfree.constraints.base.pointsofparticles import (
    addForcesOnPointsOfParticle,
    locateTheNodesOfParticles,
)
from edelweissmeshfree.constraints.test._residualtangent import KernelFunctionStandIn


def test_the_nodes_are_located_once_in_the_order_the_particles_name_them():
    nodes = [Node(label, np.zeros(2)) for label in range(4)]
    kernelFunctions = [KernelFunctionStandIn(node) for node in nodes]
    localIndexOfNode = {}

    localNodes = locateTheNodesOfParticles(
        [[kernelFunctions[2], kernelFunctions[0]], [kernelFunctions[0], kernelFunctions[3]]], localIndexOfNode
    )

    assert [located.tolist() for located in localNodes] == [[0, 1], [1, 2]]
    assert list(localIndexOfNode) == [nodes[2], nodes[0], nodes[3]]


def test_a_particle_without_kernel_functions_takes_no_force_and_names_no_node():
    """The nodes of a particle are indices into the block of the constraint, also if there are none: a force on a point
    of a particle without kernel functions is added to no node, instead of failing at an index array of floats."""

    kernelFunction = KernelFunctionStandIn(Node(1, np.zeros(2)))
    localIndexOfNode = {}

    withoutNodes, withANode = locateTheNodesOfParticles([[], [kernelFunction]], localIndexOfNode)

    assert withoutNodes.dtype == np.intp and withoutNodes.size == 0
    assert withANode.tolist() == [0]

    PAtNodes = np.zeros((1, 2))
    addForcesOnPointsOfParticle(PAtNodes, withoutNodes, np.zeros((1, 0)), np.array([[3.0, 4.0]]))
    np.testing.assert_array_equal(PAtNodes, 0.0)
    addForcesOnPointsOfParticle(PAtNodes, withANode, np.ones((1, 1)), np.array([[3.0, 4.0]]))
    np.testing.assert_array_equal(PAtNodes, [[-3.0, -4.0]])

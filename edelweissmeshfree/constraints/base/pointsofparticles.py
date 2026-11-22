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
"""The two operations every constraint on points of particles needs besides its own physics: where the nodes of its
particles sit in its block, and how a force on a point of a particle reaches those nodes.

A constraint that measures a point of a particle -- a vertex held by a support, a vertex in contact -- acts on the nodes
of the particle's kernel functions, which change as the particle moves. Its block holds each of these nodes once
(:func:`locateTheNodesOfParticles`). A force on the point reaches the nodes with the interpolation vector with which
the particle moves the point, :math:`G_B`
(:meth:`~edelweissmeshfree.particles.base.baseparticle.BaseParticle.getAttachedPointInterpolationVector`), the
transpose of the point's motion, so that a penalty on the point has a symmetric stiffness and does no work over a
closed cycle (:func:`addForcesOnPointsOfParticle`). The constraints of
:mod:`edelweissmeshfree.constraints.explicit` and
:class:`~edelweissmeshfree.constraints.particlepenaltyweakdirichtlet.ParticlePenaltyWeakDirichlet` use both; a new one
is then its physics -- the gap, the pairing, the force -- and a constructor.
"""

from collections.abc import Iterable

import numpy as np


def locateTheNodesOfParticles(kernelFunctionsOfParticles: Iterable[list], localIndexOfNode: dict) -> list[np.ndarray]:
    """Locate the nodes of the kernel functions of particles in the block of a constraint: a node not located yet is
    added to ``localIndexOfNode`` at the next index, in the order in which the particles name their kernel functions;
    a node shared by several particles is located once.

    Parameters
    ----------
    kernelFunctionsOfParticles
        The kernel functions of each particle.
    localIndexOfNode
        Where each node located so far sits in the block of the constraint; extended here. The keys, in their order,
        are the nodes of the constraint.

    Returns
    -------
    list[numpy.ndarray]
        Per particle, where the nodes of its kernel functions sit in the block, in the order of its kernel functions.
    """

    # indices also for a particle without kernel functions: an empty array without a type would be one of floats
    return [
        np.array(
            [localIndexOfNode.setdefault(kernelFunction.node, len(localIndexOfNode)) for kernelFunction in kfs],
            dtype=np.intp,
        )
        for kfs in kernelFunctionsOfParticles
    ]


def addForcesOnPointsOfParticle(
    PAtNodes: np.ndarray, localNodes: np.ndarray, G: np.ndarray, forces: np.ndarray, components: list[int] = None
):
    """Add the residual contribution of forces on points of one particle, the negative of the forces on the nodes of its
    kernel functions: :math:`\\boldsymbol{P}_B \\mathrel{-}= \\sum_p G_{pB}\\, \\boldsymbol{f}_p`, with the interpolation
    vector :math:`G_{pB}` with which the particle moves the point :math:`p`.

    Parameters
    ----------
    PAtNodes
        The residual contribution of the constraint, node by node: ``nNodes x fieldSize``; augmented.
    localNodes
        Where the nodes of the particle's kernel functions sit in the block (:func:`locateTheNodesOfParticles`). They
        are distinct, so the fancy indexing drops no contribution.
    G
        The interpolation vectors of the points, one row per point: ``nPoints x nNodesOfTheParticle``.
    forces
        The forces on the points, one row per point: ``nPoints x fieldSize``, or ``nPoints x len(components)``.
    components
        The components the forces act in, e.g. those a support holds; all if None.
    """

    if components is None:
        PAtNodes[localNodes] -= G.T @ forces
    else:
        PAtNodes[np.ix_(localNodes, components)] -= G.T @ forces

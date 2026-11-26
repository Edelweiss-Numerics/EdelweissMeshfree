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
(:func:`locateTheNodesOfParticles`). A force on the point reaches the nodes with the vector the constraint chose for
the point (:func:`addForcesOnPointsOfParticle`;
:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`): the
shape functions at the point, or the particle's interpolation linearized about its center, with which it moves the
point, the transpose of the point's motion, so that a penalty on the point has a symmetric stiffness and does no work
over a closed cycle. The explicit solver bounds that stiffness for its critical time step
(:func:`addBoundOfAbsoluteStiffnessOfPointsOfParticle`, the one bound of penalty springs of EdelweissFE,
:mod:`edelweissfe.constraints.base.penaltysprings`). A penalty constraint gives its springs on the points of a particle
as :class:`SpringsOnPointsOfParticle`, from one method for its force and its bound, so that the two cannot drift apart.
A contact of points of one particle with an obstacle in an explicit analysis derives from
:class:`~edelweissmeshfree.constraints.base.penaltyspringsexplicit.PenaltySpringsOnPointsOfOneParticleExplicit`, which
owns the force and the bound: a new one is then its physics -- the gap and its gradient -- and a constructor. The
support :class:`~edelweissmeshfree.constraints.particlepenaltyweakdirichtlet.ParticlePenaltyWeakDirichlet` gives its
springs per particle the same way.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from edelweissfe.constraints.base.penaltysprings import (
    SpringGradientOnSharedNodes,
    addBoundOfAbsoluteStiffnessOfSprings,
)


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
    PAtNodes: np.ndarray,
    localNodes: np.ndarray,
    interpolations: np.ndarray,
    forces: np.ndarray,
):
    """Add the residual contribution of forces on points of one particle, the negative of the forces on the nodes of its
    kernel functions: :math:`\\boldsymbol{P}_B \\mathrel{-}= \\sum_p I_{pB}\\, \\boldsymbol{f}_p`, with the vector
    :math:`I_{pB}` the constraint distributes the force on the point :math:`p` with
    (:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`).

    Parameters
    ----------
    PAtNodes
        The residual contribution of the constraint, node by node: ``nNodes x fieldSize``; augmented.
    localNodes
        Where the nodes of the particle's kernel functions sit in the block (:func:`locateTheNodesOfParticles`). They
        are distinct, so the fancy indexing drops no contribution.
    interpolations
        The vectors of the points, one row per point: ``nPoints x nNodesOfTheParticle``.
    forces
        The forces on the points, one row per point: ``nPoints x fieldSize``.
    """

    PAtNodes[localNodes] -= interpolations.T @ forces


@dataclass(frozen=True)
class SpringsOnPointsOfParticle:
    """Penalty springs on points of one particle. Spring :math:`p` measures a gap :math:`g_p` of its point, positive
    where a contact is open, and acts along the unit direction :math:`\\boldsymbol{n}_p` of the gradient of the gap
    (for a contact the outward normal of the obstacle, for a support the axis it holds). Its law gives the force
    :math:`f_p = -k_p\\, g_p` on the point along :math:`\\boldsymbol{n}_p`, which pushes a closed point out
    (:math:`g_p < 0`, :math:`f_p > 0`) and pulls a held point back to where it is held, and the stiffness :math:`k_p =
    -\\mathrm{d}f_p/\\mathrm{d}g_p > 0`.
    The force :math:`f_p \\boldsymbol{n}_p` reaches the nodes with the vector of the point
    (:func:`addForcesOnPointsOfParticle`), and the stiffness is :math:`k_p\\, \\boldsymbol{a}_p \\otimes
    \\boldsymbol{a}_p` with :math:`\\boldsymbol{a}_p = \\boldsymbol{I}_p \\otimes \\boldsymbol{n}_p`
    (:func:`addBoundOfAbsoluteStiffnessOfPointsOfParticle`). The bound takes the vector of the point for its motion as
    well, which it is where the constraint uses the particle's interpolation linearized about its center, or at the
    center; elsewhere the point moves with that interpolation, not with the shape functions the force is distributed
    with (:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`).
    """

    #: The vectors of the points of the springs, one row per spring: ``nSprings x nNodesOfTheParticle``, those the
    #: constraint measures and distributes with
    #: (:meth:`~edelweissmeshfree.constraints.base.mpmconstraintbase.MPMConstraintBase.interpolationVectorAtPointOf`).
    interpolations: np.ndarray
    #: The unit direction of each spring, that of the gradient of its gap: ``nSprings x fieldSize``.
    directions: np.ndarray
    #: The force of each spring on its point along its direction, :math:`f_p = -k_p\\, g_p`, at the gap it was given.
    forces: np.ndarray
    #: The stiffness of each spring, :math:`k_p = -\\mathrm{d}f_p/\\mathrm{d}g_p`.
    stiffness: np.ndarray


def addBoundOfAbsoluteStiffnessOfPointsOfParticle(
    bound: np.ndarray, weights: np.ndarray, localNodes: np.ndarray, springs: SpringsOnPointsOfParticle
):
    """Add the row sums of the absolute value of the stiffness of penalty springs on points of one particle, weighted
    by column (see :meth:`~edelweissfe.constraints.base.constraintbase.ConstraintBase.addBoundOfAbsoluteStiffnessTimes`).
    A spring of stiffness :math:`k_p` on the point :math:`p` along the unit direction :math:`\\boldsymbol{n}_p` has the
    stiffness :math:`k_p\\, \\boldsymbol{a}_p \\otimes \\boldsymbol{a}_p`, with the vector it acts with,
    :math:`\\boldsymbol{a}_p = \\boldsymbol{I}_p \\otimes \\boldsymbol{n}_p`, and the vector :math:`\\boldsymbol{I}_p`
    of the point (:attr:`SpringsOnPointsOfParticle.interpolations`), so, with the weights :math:`w_{Cj}` of the columns,

    .. math::

        b_{Bi} \\mathrel{+}= \\sum_p |k_p|\\, |I_{pB}|\\, |n_{pi}| \\sum_{C,j} |I_{pC}|\\, |n_{pj}|\\, w_{Cj} ,

    the bound of every penalty spring
    (:func:`~edelweissfe.constraints.base.penaltysprings.addBoundOfAbsoluteStiffnessOfSprings`), over the nodes of the
    particle's kernel functions.

    Parameters
    ----------
    bound
        The row sums of the constraint, one per degree of freedom of its block (node by node); augmented.
    weights
        The weights of the columns, likewise.
    localNodes
        Where the nodes of the particle's kernel functions sit in the block (:func:`locateTheNodesOfParticles`).
    springs
        The springs.
    """

    fieldSize = springs.directions.shape[1]
    dofsOfNodes = localNodes[:, None] * fieldSize + np.arange(fieldSize)
    addBoundOfAbsoluteStiffnessOfSprings(
        bound,
        weights,
        springs.stiffness,
        [SpringGradientOnSharedNodes(springs.interpolations, springs.directions, dofsOfNodes)],
    )


def addForcesOfSpringsOnPointsOfParticle(
    PAtNodes: np.ndarray, localNodes: np.ndarray, springs: SpringsOnPointsOfParticle
):
    """Add the residual contribution of penalty springs on points of one particle: the negative of the forces
    :math:`f_p \\boldsymbol{n}_p` they exert on the nodes of its kernel functions (:func:`addForcesOnPointsOfParticle`).

    Parameters
    ----------
    PAtNodes
        The residual contribution of the constraint, node by node: ``nNodes x fieldSize``; augmented.
    localNodes
        Where the nodes of the particle's kernel functions sit in the block (:func:`locateTheNodesOfParticles`).
    springs
        The springs.
    """

    addForcesOnPointsOfParticle(
        PAtNodes, localNodes, springs.interpolations, springs.forces[:, None] * springs.directions
    )

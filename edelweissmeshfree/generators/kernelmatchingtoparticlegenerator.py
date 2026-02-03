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

import typing

from edelweissfe.journal.journal import Journal
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet

from edelweissmeshfree.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.sets.particleset import ParticleSet


def generateKernelMatchingToParticle(
    model: MPMModel,
    journal: Journal,
    kernelFunctionFactoryCallback: typing.Callable[[Node, float], BaseMeshfreeKernelFunction],
    theParticlesToBeMatched: ParticleSet,
    supportScalingFactor: float = 1.5,
    name: str = "matching_kernels",
    firstKernelFunctionNumber: int = 1,
):
    """Generate kernel functions that match the support size to the particles in a given particle set.

    Parameters
    ----------
    model : MPMModel
        The MPM model to which the kernel functions will be added.
    journal : Journal
        The journal for logging information.
    kernelFunctionFactoryCallback : typing.Callable[[Node, float], BaseMeshfreeKernelFunction]
        A callback function that creates a kernel function given a node and support size.
    theParticlesToBeMatched : ParticleSet
        The set of particles to which the kernel functions will be matched.
    supportScalingFactor : float, optional
        A scaling factor to adjust the support size of the kernel functions relative to the particle size, by default 1.5.
    name : str, optional
        The name of the kernel function set, by default "matching_kernels".
    firstKernelFunctionNumber : int, optional
        The starting number for the kernel functions, by default 1.

    Returns
    -------
    MPMModel
        The updated model.
    """
    journal.message(f"Generating kernel functions matching to particles in set '{theParticlesToBeMatched.name}'.", name)

    kernelFunctionNumber = firstKernelFunctionNumber

    nDim = model.domainSize

    nodes = []

    currentKernelFunctionNumber = firstKernelFunctionNumber

    minCharacteristicLength = 0.0
    maxCharacteristicLength = 0.0
    for particle in theParticlesToBeMatched:

        position = particle.getCenterCoordinates()
        particleVolume = particle.getVolumeUndeformed()

        characteristicLength = particleVolume ** (1.0 / nDim)
        supportSize = supportScalingFactor * characteristicLength

        minCharacteristicLength = (
            min(minCharacteristicLength, characteristicLength)
            if minCharacteristicLength > 0.0
            else characteristicLength
        )
        maxCharacteristicLength = max(maxCharacteristicLength, characteristicLength)

        node = Node(
            currentKernelFunctionNumber,
            position,
        )

        kernelFunction = kernelFunctionFactoryCallback(node, supportSize)

        model.meshfreeKernelFunctions[currentKernelFunctionNumber] = kernelFunction
        currentKernelFunctionNumber += 1
        nodes.append(kernelFunction.node)

    for n in nodes:
        if n.label in model.nodes:
            raise ValueError("Node with label {:} already exists in model.".format(n.label))

    journal.message(
        f"Characteristic length of particles: min = {minCharacteristicLength:.6e}, max = {maxCharacteristicLength:.6e}.",
        name,
    )

    model.nodes.update({n.label: n for n in nodes})
    model.nodeSets["{:}_all".format(name)] = NodeSet("{:}_all".format(name), nodes)

    journal.message(
        f"Generated {kernelFunctionNumber - firstKernelFunctionNumber} kernel functions matching to particles.", name
    )

    return model

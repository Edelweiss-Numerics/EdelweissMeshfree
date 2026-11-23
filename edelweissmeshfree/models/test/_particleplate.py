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
"""A plate of 8 x 4 plane strain quadrilateral particles of the size ``H``, on B-spline kernel functions bonded to
them, for the tests of the models and the constraints: a displacement particle with J2 plasticity (elastic here) and a
gradient-enhanced one with damage, in ``MATERIALS``. Skipped where Marmot is not built."""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal

E, NU, RHO = 2.5e4, 0.18, 2.4e-9
K, G = E / (3 * (1 - 2 * NU)), E / (2 * (1 + NU))
H = 10.0

MATERIALS = {
    "DisplacementSQCNIxNSNI/PlaneStrain/Quad": {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K, G, 1e9, 1e9, 1, 0, 1, RHO]),
    },
    "GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad": {
        "material": "GRADIENTENHANCEDCOMPRESSIBLENEOHOOKEDAMAGE",
        "properties": np.array([K, G, 1e-3, 1e-2, 5.0, RHO]),
    },
}


def particlePlate(particleType: str, supportFactor: float, kernelOrder: int):
    """The model of the plate, connected.

    Parameters
    ----------
    particleType
        The Marmot particle, a key of ``MATERIALS``.
    supportFactor
        The support radius of the kernel functions, in units of ``H``.
    kernelOrder
        The continuity order of the B-spline kernel functions.

    Returns
    -------
    MPMModel
        The model.
    """
    from edelweissmeshfree.generators.rectangularkernelfunctiongridgenerator import (
        generateRectangularKernelFunctionGrid,
    )
    from edelweissmeshfree.generators.rectangularquadparticlegridgenerator import (
        generateRectangularQuadParticleGrid,
    )
    from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )
    from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )
    from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
    from edelweissmeshfree.models.mpmmodel import MPMModel
    from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
        KDBinOrganizedParticleManager,
    )
    from edelweissmeshfree.particles.marmot import marmotparticlewrapper

    if marmotparticlewrapper.__file__.endswith(".py"):
        pytest.skip("Marmot is not built")

    journal = Journal()
    model = MPMModel(2)
    nX, nY = 8, 4

    def kernelFunction(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportFactor * H, continuityOrder=kernelOrder
        )

    def particle(number, vertexCoordinates, volume):
        return marmotparticlewrapper.MarmotParticleWrapper(
            particleType, number, vertexCoordinates, 0.0, approximation, MATERIALS[particleType]
        )

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1)
    model = generateRectangularKernelFunctionGrid(
        model, journal, kernelFunction, x0=0.5 * H, y0=0.5 * H, l=(nX - 1) * H, h=(nY - 1) * H, nX=nX, nY=nY
    )
    model = generateRectangularQuadParticleGrid(model, journal, particle, l=nX * H, h=nY * H, nX=nX, nY=nY, name="b")
    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    manager = KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    model.particleKernelDomains["b"] = domain
    model.prepareYourself(journal)
    model.particleManagers.append(manager)
    model.updateConnectivity()
    return model

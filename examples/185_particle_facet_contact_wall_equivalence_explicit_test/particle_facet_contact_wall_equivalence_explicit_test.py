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
"""A particle projectile falls onto a floor: once onto an analytic wall
(:class:`~edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit.ParticlePenaltyCartesianBoundaryConstraintExplicit`),
once onto a strip of Line2 contact facets at the same place
(:class:`~edelweissmeshfree.constraints.explicit.particlepenaltyfacetcontactexplicit.ParticlePenaltyFacetContactExplicit`).
The contact points (the vertices of the bottom particles) and the penalty are the same, so the two must agree to
round-off: the facet contact sums the forces of all particles in one block, the wall in one block per particle, so the
sums are taken in another order. The floor is rigid: the facets are not part of the model, and the contact applies no
reaction to them.
"""

import argparse

import numpy as np
import pytest
from edelweissfe.elements.contactsurfaceelement import Line2ContactFacet
from edelweissfe.journal.journal import Journal
from edelweissfe.points.node import Node
from edelweissfe.solvers.nonlinearexplicitdynamicparallel import NEDParallel
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
    ParticleExplicitPenaltyCartesianBoundaryConstraintFactory,
)
from edelweissmeshfree.constraints.explicit.particlepenaltyfacetcontactexplicit import (
    ParticlePenaltyFacetContactExplicit,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
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
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)

PENALTY = 1e5
#: The floor is at y = 0, the projectile starts this far above it.
STANDOFF = 1.0
VELOCITY = -400e3


def floorFacets(xMin: float, xMax: float, nFacets: int) -> list:
    """Line2 contact facets along y = 0, traversed from +x to -x so that their normals point up, out of the floor."""

    x = np.linspace(xMax, xMin, nFacets + 1)
    nodes = [Node(100000 + i, np.array([xi, 0.0])) for i, xi in enumerate(x)]
    facets = []
    for i in range(nFacets):
        facet = Line2ContactFacet("Line2ContactFacet", 100000 + i)
        facet.setNodes([nodes[i], nodes[i + 1]])
        facet.initializeElement()
        facets.append(facet)
    return facets


def run_sim(floor: str):
    """Let the projectile fall onto the floor.

    Parameters
    ----------
    floor
        "wall" for the analytic wall, "facets" for the contact facets, "none" for no floor.

    Returns
    -------
    tuple
        The field output controller, and the facet contact (None without it).
    """

    dimension = 2
    theJournal = Journal(verbose=False)
    theModel = MPMModel(dimension)

    particleSize = 10.0
    length, height = 100.0, 200.0
    nX, nY = int(length / particleSize), int(height / particleSize)

    def theKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=3.0 * particleSize, continuityOrder=3
        )

    generateRectangularKernelFunctionGrid(
        theModel, theJournal, theKernelFunctionFactory, x0=0.0, y0=STANDOFF, h=height, l=length, nX=nX, nY=nY,
        name="projectile",
    )  # fmt: skip

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    E, nu = 78200.0, 0.3
    K, G = E / (3 * (1 - 2 * nu)), E / (2 * (1 + nu))
    theMaterial = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([K, G, 290e10, 290e10, 1, 0, 1, 1.0e-8]),
    }

    def theParticleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/PlaneStrain/Quad", number, vertexCoordinates, 0.0, theApproximation, theMaterial
        )

    generateRectangularQuadParticleGrid(
        theModel, theJournal, theParticleFactory, x0=0.0, y0=STANDOFF, h=height, l=length, nX=nX, nY=nY,
        name="projectile",
    )  # fmt: skip

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True
    )
    theModel.particleKernelDomains["projectile"] = theParticleKernelDomain

    bottomParticles = list(theModel.particleSets["projectile_bottom"])
    facetContact = None
    if floor == "wall":
        theModel.constraints.update(
            ParticleExplicitPenaltyCartesianBoundaryConstraintFactory(
                "floor", 0.0, 1, bottomParticles, "displacement", theModel, location="vertex",
                vertexIDs=[0, 1, 2, 3], penaltyParameter=PENALTY,
            )
        )  # fmt: skip
    elif floor == "facets":
        facetContact = ParticlePenaltyFacetContactExplicit(
            "floor", theModel, bottomParticles, floorFacets(-50.0, 150.0, 40), PENALTY, 5.0, theJournal,
            twoWayCoupling=False,
        )  # fmt: skip
        theModel.constraints["floor"] = facetContact

    theModel.prepareYourself(theJournal)

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)
    fieldOutputController.addPerParticleFieldOutput("displacement", theModel.particleSets["all"], "displacement")
    fieldOutputController.initializeJob()

    for particle in theModel.particleSets["all"]:
        particle.getResultArray("velocity")[1] = VELOCITY
        particle.acceptStateAndPosition()

    theModel.particleManagers.append(theParticleManager)
    incrementSize = 3e-3
    step = AdaptiveStepForExplicitSimulations(
        0,
        theModel,
        fieldOutputController,
        theJournal,
        {},
        NEDParallel({}, theJournal, **{"output-frequency": 1}),
        [],
        StepActionCollection(),
        stepLength=2.0e-4,
        startInc=incrementSize,
        maxInc=incrementSize,
        minInc=incrementSize / 1e8,
        maxNumInc=101,
    )

    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()

    return fieldOutputController, facetContact


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent directory."""

    monkeypatch.chdir(request.fspath.dirname)


def _displacement(fieldOutputController) -> np.ndarray:
    return fieldOutputController.fieldOutputs["displacement"].getLastResult()


def test_sim(assert_gold):
    onFacets, facetContact = run_sim("facets")
    onWall, _ = run_sim("wall")
    inFreeFlight, _ = run_sim("none")

    uFacets, uWall, uFree = _displacement(onFacets), _displacement(onWall), _displacement(inFreeFlight)

    # not vacuous: the floor changes the answer
    assert np.abs(uWall - uFree).max() > 0.1 * np.abs(uFree).max()
    assert facetContact.numberOfPointsFoundInside == 0

    np.testing.assert_allclose(uFacets, uWall, rtol=0.0, atol=1e-10 * np.abs(uWall).max())

    assert_gold(uFacets, np.loadtxt("gold.csv"), atol=1e-12)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    onFacets, facetContact = run_sim("facets")
    onWall, _ = run_sim("wall")
    inFreeFlight, _ = run_sim("none")
    uFacets, uWall, uFree = _displacement(onFacets), _displacement(onWall), _displacement(inFreeFlight)
    print("max |u_facets - u_wall| = {:.3e}".format(np.abs(uFacets - uWall).max()))
    print(
        "max |u_wall - u_free| = {:.3e}, max |u_free| = {:.3e}".format(np.abs(uWall - uFree).max(), np.abs(uFree).max())
    )
    print("points found inside: {:}".format(facetContact.numberOfPointsFoundInside))

    if args.create_gold:
        np.savetxt("gold.csv", uFacets.flatten())

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
#  Thomas Mader    |  thomas.mader@boku.ac.at
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
"""Cook's membrane with the mixed u-p-J (DisplacementPressureJacobi) particle.

Nearly incompressible elasticity (K/G = 500). The right edge is driven
upwards by a weak Dirichlet constraint; the reaction force on the left edge
and the tip displacement are recorded. The VMS pressure stabilization of the
particle is controlled via the particle property 'vms alpha'.
"""

import argparse

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.constraints.particlelagrangianweakdirichlet import (
    ParticleLagrangianWeakDirichletOnParticleSetFactory,
)
from edelweissmeshfree.constraints.particlemortarweakdirichlet import (
    ParticleMortarWeakDirichletOnParticleSetFactory,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.cooksmembranekernelfunctiongridgenerator import (
    generateCooksMembraneKernelFunctionGrid,
)
from edelweissmeshfree.generators.cooksmembranequadparticlegridgenerator import (
    generateCooksMembraneParticleGrid,
)
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver


def run_sim(
    particleType="DisplacementPressureJacobiSQCNIxNSNI/PlaneStrain/Quad",
    vmsAlpha=0.1,
    vmsMode=0,
    nX=12,
    nY=12,
    uYTip=5.0,
    constraintType="mortar",
    multiplierOrder=6,
    constraintStride=1,
    cwfCorrection=False,
    outputName=None,
):
    """vmsMode: 0 = pressure-only VMS stabilization (grad(p) part of the strong-form
    momentum residual only), 1 = full VMS (grad(p) + div(S_dev) - rho0*a).

    cwfCorrection: add the consistent-weak-form boundary traction term
    int_Gamma T.(S_dev + p I) DeltaF^-T (N dA) on the fully constrained left edge
    (distributed load type 'cwfcorrection'). The weak Dirichlet enforcement
    otherwise imposes the traction-free natural condition between the constrained
    points, whose residual excites the pressure boundary layer in the first
    particle columns.

    constraintType: 'mortar' (default) or 'lagrange' (point collocation).

    Point collocation of BOTH displacement components at every particle center of a
    nearly incompressible edge produces alternating point reactions and, with them, a
    pressure checkerboard along the boundary column. This is independent of the
    constraint formulation (Lagrange multipliers and penalty give the identical
    pattern) and cannot be removed by the VMS stabilization (it is a forced response,
    not a spurious mode). The mortar constraint enforces the boundary condition in the
    integrated sense with a smooth, low-order polynomial multiplier field: the boundary
    traction is smooth by construction and the pressure field stays clean.

    constraintStride: with constraintType='lagrange', clamp only every n-th particle
    (a crude approximation of the mortar idea)."""
    dimension = 2

    np.set_printoptions(linewidth=200, precision=4)

    theJournal = Journal()

    theModel = MPMModel(dimension)

    x0 = 0
    y0 = 0
    heightLeft = 44
    heightRight = 16
    length = 48
    supportRadius = 2.2 * length / nX

    if outputName is None:
        outputName = f"cooks_upj_{particleType.split('/')[0]}_alpha{vmsAlpha}_mode{vmsMode}" + (
            "_cwf" if cwfCorrection else ""
        )

    def theMeshfreeKernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=2
        )

    theModel = generateCooksMembraneKernelFunctionGrid(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        x0=x0,
        y0=y0,
        l=length,
        h0=heightLeft,
        h1=heightRight,
        nX=nX,
        nY=nY,
    )

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    # nearly incompressible elasticity: K/G = 500 (nu ~ 0.499)
    K = 40000.0
    G = 10.0
    theMaterial = {
        "material": "FiniteStrainJ2Plasticity",
        # K, G, fy, fyInf, eta, H, implementationType, density
        "properties": np.array([K, G, 1e8, 1e8, 1e0, 1e0, 1, 1e-7]),
    }

    def TheParticleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            particleType,
            number,
            vertexCoordinates,
            0.0,  # volume is computed from the vertex coordinates
            theApproximation,
            theMaterial,
        )

    theModel = generateCooksMembraneParticleGrid(
        theModel,
        theJournal,
        TheParticleFactory,
        x0=x0,
        y0=y0,
        l=length,
        h0=heightLeft,
        h1=heightRight,
        nX=nX,
        nY=nY,
    )

    for particle in theModel.particles.values():
        particle.setProperty("newmark-beta beta", 0.0)
        particle.setProperty("newmark-beta gamma", 0.0)
        particle.setProperty("vms alpha", vmsAlpha)
        particle.setProperty("vms mode", float(vmsMode))

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )

    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True
    )

    print(theParticleManager)

    theModel.particleKernelDomains["my_all_with_all"] = theParticleKernelDomain

    if constraintType == "mortar":
        multiplierOrder = min(multiplierOrder, nY - 1)
        dirichletLeft = ParticleMortarWeakDirichletOnParticleSetFactory(
            "left",
            theModel.particleSets["cooks_membrane_left"],
            "displacement",
            {0: 0, 1: 0},
            theModel,
            multiplierOrder=multiplierOrder,
        )
        dirichletRight = ParticleMortarWeakDirichletOnParticleSetFactory(
            "right",
            theModel.particleSets["cooks_membrane_right"],
            "displacement",
            {1: uYTip},
            theModel,
            multiplierOrder=multiplierOrder,
        )
    elif constraintType == "lagrange":
        dirichletLeft = ParticleLagrangianWeakDirichletOnParticleSetFactory(
            "left",
            list(theModel.particleSets["cooks_membrane_left"])[::constraintStride],
            "displacement",
            {0: 0, 1: 0},
            theModel,
        )
        dirichletRight = ParticleLagrangianWeakDirichletOnParticleSetFactory(
            "right",
            theModel.particleSets["cooks_membrane_right"],
            "displacement",
            {1: uYTip},
            theModel,
        )
    else:
        raise ValueError(f"unknown constraintType {constraintType}")

    theModel.constraints.update(dirichletLeft)
    theModel.constraints.update(dirichletRight)

    particleDistributedLoads = []
    if cwfCorrection:
        from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface

        from edelweissmeshfree.stepactions.particledistributedload import (
            ParticleDistributedLoad,
        )

        # face 4 = left edge of the quad particles; both displacement components are
        # constrained there, so the full traction correction is consistent
        surfaceCWF = EntityBasedSurface(
            name="surfaceCWFLeft",
            faceToEntities={4: list(theModel.particleSets["cooks_membrane_left"])},
        )
        particleDistributedLoads.append(
            ParticleDistributedLoad(
                name="cwf_dirichlet_left",
                model=theModel,
                journal=theJournal,
                particleSurface=surfaceCWF,
                distributedLoadType="cwfcorrection",
                loadVector=np.array([0.0]),
                f_t=lambda t: 1.0,
            )
        )

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["all"],
        "displacement",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "pressure",
        theModel.particleSets["all"],
        "pressure",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "jacobi",
        theModel.particleSets["all"],
        "jacobi",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements",
        theModel.particleSets["all"],
        "vertex displacements",
        f_x=lambda x: np.pad(np.reshape(x, (-1, 2)), ((0, 0), (0, 1)), mode="constant", constant_values=0),
    )
    fieldOutputController.addPerParticleFieldOutput(
        "deformation gradient",
        theModel.particleSets["all"],
        "deformation gradient",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "stress",
        theModel.particleSets["all"],
        "stress",
    )
    fieldOutputController.addExpressionFieldOutput(
        None,
        lambda: sum(c.reactionForce for c in dirichletLeft.values()),
        "reaction force",
        saveHistory=True,
        export=f"RF_{outputName}",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "U_Right",
        theModel.particleSets["cooks_membrane_rightTop"],
        "displacement",
        saveHistory=True,
        export=f"U_{outputName}",
        f_x=lambda x: x[-1, 1],
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager(f"_ensight_{outputName}", theModel, fieldOutputController, theJournal, None)
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["pressure"], create="perElement")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["jacobi"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        create="perNode",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perElement"
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["stress"], create="perElement")
    ensightOutput.initializeJob()

    incSize = 5e-2
    adaptiveTimeStepper = AdaptiveTimeStepper(0.0, 1.0, incSize, incSize, incSize / 16, 5000, theJournal)

    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)

    iterationOptions = dict()
    iterationOptions["max. iterations"] = 15
    iterationOptions["critical iterations"] = 5
    iterationOptions["allowed residual growths"] = 10

    linearSolver = pardisoSolve

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=theModel.constraints.values(),
            userIterationOptions=iterationOptions,
            particleDistributedLoads=particleDistributedLoads,
        )

    except StepFailed as e:
        theJournal.message(f"Step failed: {str(e)}", "error")
        raise

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = theJournal.linewidth
        theJournal.printPrettyTable(prettytable, "Summary")

    return theModel, fieldOutputController


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim():
    # disable plots and suppress warnings
    import matplotlib

    matplotlib.use("Agg")
    import warnings

    warnings.filterwarnings("ignore")

    theModel, fieldOutputController = run_sim(nX=6, nY=6, uYTip=2.0)

    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    gold = np.loadtxt("gold.csv")

    assert np.isclose(res.flatten(), gold.flatten(), rtol=1e-12).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    parser.add_argument("--particleType", "-p", dest="particleType", default="DisplacementPressureJacobiSQCNIxNSNI/PlaneStrain/Quad")
    parser.add_argument("--vmsAlpha", "-a", dest="vmsAlpha", type=float, default=0.1)
    parser.add_argument("--vmsMode", "-m", dest="vmsMode", type=int, choices=[0, 1], default=0,
                        help="0: pressure-only VMS, 1: full VMS (grad p + div S_dev - rho0 a)")
    parser.add_argument("--nX", type=int, default=12)
    parser.add_argument("--nY", type=int, default=12)
    parser.add_argument("--uYTip", type=float, default=5.0)
    parser.add_argument("--constraintType", choices=["mortar", "lagrange"], default="mortar")
    parser.add_argument("--multiplierOrder", type=int, default=6, help="polynomial order of the mortar multiplier field")
    parser.add_argument("--constraintStride", type=int, default=1, help="with 'lagrange': clamp every n-th left-edge particle")
    parser.add_argument("--cwf", dest="cwf", action="store_true",
                        help="consistent-weak-form traction correction on the clamped left edge")
    args = parser.parse_args()

    theModel, fieldOutputController = run_sim(
        particleType=args.particleType,
        vmsAlpha=args.vmsAlpha,
        vmsMode=args.vmsMode,
        nX=args.nX,
        nY=args.nY,
        uYTip=args.uYTip,
        constraintType=args.constraintType,
        multiplierOrder=args.multiplierOrder,
        constraintStride=args.constraintStride,
        cwfCorrection=args.cwf,
    )

    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    if args.create_gold:
        np.savetxt("gold.csv", res.flatten())

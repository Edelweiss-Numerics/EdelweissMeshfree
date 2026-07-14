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
    cwfCorrection="off",
    vci=False,
    vciOrder=1,
    completenessOrder=1,
    outputName=None,
):
    """vmsMode: 0 = pressure-only VMS stabilization (grad(p) part of the strong-form
    momentum residual only), 1 = full VMS (grad(p) + div(S_dev) - rho0*a).

    cwfCorrection: add the consistent-weak-form boundary traction term
    int_Gamma T.(S_dev + p I) DeltaF^-T (N dA) (distributed load type
    'cwfcorrection'; the load vector is the per-component Dirichlet mask). The
    weak Dirichlet enforcement otherwise imposes the traction-free natural
    condition between the constrained points, whose residual excites the
    pressure boundary layer in the first particle columns.
    - 'off'  : no correction (default)
    - 'left' : fully clamped left edge, mask (1, 1)
    - 'both' : additionally the driven right edge with mask (0, 1) (only u_y
               constrained there)

    CAUTION -- the correction is the CONSISTENCY term of a Nitsche-type
    formulation WITHOUT the accompanying penalty/stabilization term, so it is
    only conditionally beneficial and conditionally stable (measured, 12x12,
    alpha=0.1, absolute column oscillation):
    - uYTip=30, 'left': columns 1-3 improve strongly (5.3 -> 2.8, 2.7 -> 1.0),
      the edge column itself carries the (physical) corner reaction. Use it.
    - uYTip=30, 'both': the heavily loaded driven edge destabilizes the
      saddle point -- the run FAILS at t ~ 0.9.
    - uYTip=5 (any): net WORSE -- the traction feeds the discrete corner
      stress back as an edge load; with little inconsistency error to fix,
      the edge-column oscillation triples.
    A robust always-on variant needs the Nitsche penalty term (future work).

    vci: first-order variationally consistent integration (particle property
    'VCI order' = 1 plus a VariationallyConsistentIntegrationManager over the
    four boundary edges). Works with both the SQCNIxNSNI and the SQCNIxSDI
    particle. Measured (12x12, alpha=0.1, mode 0, checkerboard metrics
    cb_lap / cb_alt and RMS vertical p-oscillation of columns 0/1):
    - uYTip=30, NSNI: 15.3% / 1.73% -> 13.7% / 1.34%, columns 2.32/2.66 ->
      1.37/2.29 -- clear improvement, same solve cost.
    - uYTip=30, SDI: 10.1% / 0.55% -> 8.2% / 0.16%, columns 1.42/1.84 ->
      1.29/1.51, and the streaky bands near the driven edge shrink. SDI+VCI
      is the cleanest of the four variants.
    - uYTip=5: differences are small (fields are already clean at this load).

    vciOrder: order of the VCI constraint basis. The basis is KERNEL-CENTERED
    (shifted monomials, see the particle vci_* overrides) -- with global
    monomials the order-2 M matrices are numerically degenerate (cond ~1e10,
    NaN in the first increment). Measured at uYTip=20 (all variants complete):
    - NSNI + order 2: WORSE than order 1 (cb_lap 12.6% -> 16.2%, column 0
      0.74 -> 2.73). The NSNI volume integrals use a single center evaluation
      point; the quadratic correction over-fits it.
    - SDI + order 2: the BEST variant overall (cb_lap 11.5% -> 9.3%, columns
      0/1 0.84/0.75 -> 0.45/0.53, right-edge streaks halved vs order 1) --
      the 4 subdomain evaluation points support the quadratic constraints.
    - LIMIT: order 2 destabilizes at extreme distortion (both particles NaN
      at tip ~22.6 mm; with completenessOrder=2 even earlier). The truncated
      pseudo-inverse guard in vci.py bounds but does not remove this.

    cwfCorrection with the SQCNIxSDI particle: implemented per attached
    subcell (each with its own material-point stress). SDI + CWF-left runs to
    30 mm and clears the boundary layer (columns 1-3), but the edge column
    carries the corner reaction (col 0: 1.42 -> 3.8) and the interior is
    slightly noisier than SDI + VCI. SDI + VCI + CWF together FAIL at
    t ~ 0.52 (u ~ 15.6 mm) -- do not combine them; NSNI + VCI + CWF-left is
    stable to 30 mm and gives the lowest boundary-layer oscillation of the
    NSNI variants (columns 1-3: 1.12/1.13/0.71).

    The SQCNIxSDI particle integrates with per-subdomain material points
    (4 subdomains) instead of the NSNI second-derivative term; it runs
    slightly softer initially and stiffer at large deformation than NSNI
    (left-edge reaction ~110 vs ~100 at 30 mm). It needs the absolute jacobi
    flux tolerance below (the jacobi residual reaches its roundoff floor
    ~1e-10 for K = 4e4).

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
    # quadratic completeness needs a larger support (more nodes than monomials everywhere)
    supportRadius = (2.2 if completenessOrder == 1 else 3.2) * length / nX

    if outputName is None:
        outputName = f"cooks_upj_{particleType.split('/')[0]}_alpha{vmsAlpha}_mode{vmsMode}" + (
            f"_cwf-{cwfCorrection}" if cwfCorrection != "off" else ""
        ) + (f"_vci{vciOrder}" if vci else "") + (
            f"_compl{completenessOrder}" if completenessOrder != 1 else ""
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

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=completenessOrder)

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
        if vci:
            # default order 1 matches the completeness order 1 of the RKPM approximation
            particle.setProperty("VCI order", float(vciOrder))

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

    vciManagers = []
    if vci:
        from edelweissmeshfree.meshfree.vci import (
            BoundaryParticleDefinition,
            VariationallyConsistentIntegrationManager,
        )

        theBoundary = [
            BoundaryParticleDefinition(theModel.particleSets["cooks_membrane_left"], np.empty(2), 4),
            BoundaryParticleDefinition(theModel.particleSets["cooks_membrane_right"], np.empty(2), 2),
            BoundaryParticleDefinition(theModel.particleSets["cooks_membrane_bottom"], np.empty(2), 1),
            BoundaryParticleDefinition(theModel.particleSets["cooks_membrane_top"], np.empty(2), 3),
        ]
        vciManagers.append(
            VariationallyConsistentIntegrationManager(
                list(theModel.particles.values()),
                list(theModel.meshfreeKernelFunctions.values()),
                theBoundary,
            )
        )

    particleDistributedLoads = []
    if cwfCorrection != "off":
        from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface

        from edelweissmeshfree.stepactions.particledistributedload import (
            ParticleDistributedLoad,
        )

        # the load vector is the per-component Dirichlet mask (1 = component constrained on
        # this edge -> apply its traction component, 0 = free -> keep the traction-free
        # natural condition). Left edge (face 4): fully clamped -> (1, 1). Right edge
        # (face 2): only u_y prescribed -> (0, 1).
        cwfEdges = [("cwf_dirichlet_left", 4, "cooks_membrane_left", (1.0, 1.0))]
        if cwfCorrection == "both":
            cwfEdges.append(("cwf_dirichlet_right", 2, "cooks_membrane_right", (0.0, 1.0)))
        for name, faceID, particleSet, mask in cwfEdges:
            surface = EntityBasedSurface(
                name=f"surface_{name}",
                faceToEntities={faceID: list(theModel.particleSets[particleSet])},
            )
            particleDistributedLoads.append(
                ParticleDistributedLoad(
                    name=name,
                    model=theModel,
                    journal=theJournal,
                    particleSurface=surface,
                    distributedLoadType="cwfcorrection",
                    loadVector=np.array(mask),
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
    # driving-edge reaction: with the CWF correction active on the LEFT edge, the left
    # mortar multiplier only carries the difference to the stress traction (the physical
    # reaction is multiplier + CWF traction integral); the right-edge multiplier is then
    # the meaningful total driving force for load-displacement curves
    fieldOutputController.addExpressionFieldOutput(
        None,
        lambda: sum(c.reactionForce for c in dirichletRight.values()),
        "reaction force right",
        saveHistory=True,
        export=f"RFright_{outputName}",
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
    # the jacobi flux r_J = T (kP - p) V0 is a difference of O(K) quantities: its roundoff
    # floor is ~1e-10 for K = 4e4, and the relative check compares against the CURRENT
    # (vanishing) flux magnitude -- give it an honest absolute tolerance
    iterationOptions["spec. absolute flux residual tolerances"] = {"jacobi": 1e-9}

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
            vciManagers=vciManagers,
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
    parser.add_argument("--vmsMode", "-m", dest="vmsMode", type=int, choices=[0, 1, 2, 3], default=0,
                        help="0: pressure-only VMS, 1: full VMS (grad p + div S_dev - rho0 a)")
    parser.add_argument("--nX", type=int, default=12)
    parser.add_argument("--nY", type=int, default=12)
    parser.add_argument("--uYTip", type=float, default=5.0)
    parser.add_argument("--constraintType", choices=["mortar", "lagrange"], default="mortar")
    parser.add_argument("--multiplierOrder", type=int, default=6, help="polynomial order of the mortar multiplier field")
    parser.add_argument("--constraintStride", type=int, default=1, help="with 'lagrange': clamp every n-th left-edge particle")
    parser.add_argument("--cwf", dest="cwf", choices=["off", "left", "both"], default="off",
                        help="consistent-weak-form traction correction on the Dirichlet edges (see run_sim docstring)")
    parser.add_argument("--vci", dest="vci", action="store_true",
                        help="variationally consistent integration (test gradient correction)")
    parser.add_argument("--vciOrder", dest="vciOrder", type=int, default=1,
                        help="polynomial order of the VCI constraints (default 1)")
    parser.add_argument("--completenessOrder", dest="completenessOrder", type=int, default=1,
                        help="completeness order of the RKPM approximation (default 1); "
                             "must be >= vciOrder for the VCI constraints to be satisfiable")
    parser.add_argument("--outputName", dest="outputName", default=None,
                        help="basename for the RF/U exports and the ensight directory")
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
        vci=args.vci,
        vciOrder=args.vciOrder,
        completenessOrder=args.completenessOrder,
        outputName=args.outputName,
    )

    res = fieldOutputController.fieldOutputs["displacement"].getLastResult()

    if args.create_gold:
        np.savetxt("gold.csv", res.flatten())

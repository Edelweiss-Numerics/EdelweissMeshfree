import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

from edelweissmeshfree.constraints.explicit.discreterigidbodypenaltycontactexplicit import (
    DiscreteRigidBodyPenaltyContactExplicitFactory,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
)
from edelweissmeshfree.generators.particlesfromexodus import generateParticlesFromExodus
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.explicitmultiphysicssolver import (
    ExplicitMultiphysicsSolver,
)


def run_explicit_sim():
    theJournal = Journal()
    theModel = MPMModel(3)

    theMaterial = {
        "material": "CompressibleNeoHooke",
        "properties": np.array([40000.0, 0.3, 1.0]),  # E, nu, rho. E increased to make block stiffer
    }

    from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1)

    def TheParticleFactory(particleNumber, coordinates):
        return MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/3D/Hexa",
            particleNumber,
            coordinates,
            0.0,
            theApproximation,
            theMaterial,
        )

    print("Loading particles...")
    theModel = generateParticlesFromExodus(
        theModel,
        theJournal,
        "particles.exo",
        {"HEX": TheParticleFactory, "HEX8": TheParticleFactory},
        "mesh_particles",
        1,
    )

    def theMeshfreeKernelFunctionFactory(node, characteristicLength):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=characteristicLength, continuityOrder=3
        )

    print("Generating kernel matching...")
    theModel = generateKernelMatchingToParticle(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        theModel.particleSets["mesh_particles_all"],
        supportScalingFactor=2.2,
    )

    from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theModel.particleKernelDomains["all"] = theParticleKernelDomain

    from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
        KDBinOrganizedParticleManager,
    )

    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain,
        3,  # dimension
        theJournal,
        bondParticlesToKernelFunctions=True,
    )

    # Add custom particle set for bottom boundary
    bottom_particles = [p for p in theModel.particleSets["mesh_particles_all"] if p.getCenterCoordinates()[1] < -4.5]
    theModel.particleSets["bottom"] = bottom_particles

    from edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
        ParticleExplicitPenaltyCartesianBoundaryConstraintFactory,
    )

    dirichletBottom = ParticleExplicitPenaltyCartesianBoundaryConstraintFactory(
        "bottom_fix",
        boundaryPosition=-5.0,
        component=1,  # y-axis
        particleCollection=theModel.particleSets["bottom"],
        field="displacement",
        model=theModel,
        location="center",
        penaltyParameter=1e5,
    )
    theModel.constraints.update(dirichletBottom)
    theModel.constraintSets["bottom_fix"] = dirichletBottom

    # Add Discrete Rigid Body Contact constraint
    import pyvista as pv

    # 1. Discrete Rigid Body
    from edelweissmeshfree.rigidbodies.discreterigidbody import DiscreteRigidBody

    # The rigid body mesh spans y=10..20. We shift down by 4.9.
    # At velocity -10 m/s: impact at t = 0.1/10 = 0.01 s.
    rigid_body = DiscreteRigidBody.from_mesh_file(
        name="rigid_body",
        model=theModel,
        filename="rigid_body.exo",
        translation=[0.0, -4.9, 0.0],
    )

    # 2. RP Kinematic Boundary Condition
    from edelweissmeshfree.stepactions.dirichlet import Dirichlet as DirichletMF

    def velocity_amp(t):
        return t

    # Total displacement = -6.0 over the step. Cylinder starts at 10, hits block at 5.
    rp_bc = DirichletMF(
        "rp_bc", theModel.nodeSets["rigid_body_rp"], "displacement", {"2": -6.0}, theModel, theJournal, velocity_amp
    )

    # 3. Contact Constraint
    DiscreteRigidBodyPenaltyContactExplicitFactory(
        name="rigid_impact",
        particleCollection=theModel.particleSets["mesh_particles_all"],
        model=theModel,
        rigidBody=rigid_body,
        penaltyParameter=1e5,
    )

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)
    fieldOutputController.addPerParticleFieldOutput(
        "displacement", theModel.particleSets["mesh_particles_all"], "displacement"
    )
    fieldOutputController.addPerParticleFieldOutput("velocity", theModel.particleSets["mesh_particles_all"], "velocity")
    fieldOutputController.addPerParticleFieldOutput(
        "deformation gradient", theModel.particleSets["mesh_particles_all"], "deformation gradient"
    )
    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements",
        theModel.particleSets["mesh_particles_all"],
        "vertex displacements",
        reshape_to_dimensions=3,
    )

    # Add field output for rigid body (via elements)
    # The fields on discrete rigid elements will be captured by Ensight
    rigid_surface_field = theModel.nodeFields["displacement"].subset(theModel.elementSets["rigid_body_surface"])
    fieldOutputController.addPerNodeFieldOutput("rigid_displacement", rigid_surface_field, "U")

    fieldOutputController.initializeJob()

    out_folder = "explicit_sim_out"
    ensightOutput = EnsightOutputManager(
        out_folder, theModel, fieldOutputController, theJournal, None, intermediateSaveInterval=1
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["rigid_displacement"], create="perNode"
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["velocity"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perElement"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        name="vertex displacements",
        create="perNode",
    )
    ensightOutput.initializeJob()

    solver = ExplicitMultiphysicsSolver(theJournal)

    # Simulate 1.0 s: impact at t=~0.714 s, observe ~0.28 s of post-impact deformation.
    # stepLength = 1.0 s. incSize = 5e-3 -> dT = 5e-3 s -> 200 increments.
    incSize = 5e-3
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0,
        1.0,
        incSize,
        incSize,
        incSize,
        2000,
        theJournal,
    )

    try:
        solver.solveStep(
            adaptiveTimeStepper,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            dirichlets=[rp_bc],
            userIterationOptions={"field orders": {"displacement": 2, "rotation": 2}},
        )

    except Exception as e:
        theJournal.message(f"Step failed: {str(e)}", "error")
        raise
    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()
        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = theJournal.linewidth
        theJournal.printPrettyTable(prettytable, "Summary")


if __name__ == "__main__":
    run_explicit_sim()

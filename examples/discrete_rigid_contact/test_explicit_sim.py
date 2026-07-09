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

    # 1. Parse rigid body mesh and create nodes
    mesh = pv.read("rigid_body.exo")
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    # Extract surface for visualization elements
    points = mesh.points.copy()
    points[:, 1] -= 4.9
    surf = mesh.extract_surface()

    # surf faces reference surf.points, not mesh.points.
    # But since we just want to create DiscreteRigidElements, maybe we can just skip creating DiscreteRigidElement entirely!
    # Or just use mesh.cells? Let's try mesh.cells if they are triangles/quads.

    cells = mesh.cells
    # Just parse cells if they are 3-node or 4-node
    faces = []
    i = 0
    while i < len(cells):
        n = cells[i]
        faces.append(cells[i + 1 : i + 1 + n])
        i += 1 + n

    rigid_nodes = []
    # Nodes in EdelweissFE need a unique label, we start from a high number
    start_label = 1000000
    for i, pt in enumerate(points):
        from edelweissfe.points.node import Node

        n = Node(start_label + i, pt.copy())
        theModel.nodes[n.label] = n
        rigid_nodes.append(n)

    from edelweissfe.sets.nodeset import NodeSet

    theModel.nodeSets["rigid_surface_nodes"] = NodeSet("rigid_surface_nodes", rigid_nodes)

    # Add dummy displacement field to rigid nodes so it doesn't crash when exporting
    from edelweissfe.variables.fieldvariable import FieldVariable

    for n in rigid_nodes:
        n.fields["displacement"] = FieldVariable(n, "displacement")

    # Create DiscreteRigidElements for visualization
    from edelweissfe.elements.discreterigid import DiscreteRigidElement
    from edelweissfe.sets.elementset import ElementSet

    rigid_elements = []
    for i, face in enumerate(faces):
        if len(face) == 4:
            el_nodes = [rigid_nodes[face[0]], rigid_nodes[face[1]], rigid_nodes[face[2]], rigid_nodes[face[3]]]
            el = DiscreteRigidElement(start_label + i, el_nodes, theModel, "quad4")
        else:
            el_nodes = [rigid_nodes[face[0]], rigid_nodes[face[1]], rigid_nodes[face[2]]]
            el = DiscreteRigidElement(start_label + i, el_nodes, theModel, "tria3")
        theModel.elements[el.elNumber] = el
        rigid_elements.append(el)
    theModel.elementSets["rigid_surface"] = ElementSet("rigid_surface", rigid_elements)

    # 2. Reference Point and Kinematics
    # The rigid body mesh spans y=10..20, shifted down by 4.9 to span y=5.1..15.1, centroid at y=10.1.
    # Particles span y=-5..5 (top face at y=5).
    # The rigid body bottom face sits at y=5.1, giving a gap of 0.1 units to the particle top.
    # RP is placed at the mesh centroid y=10.1.
    # At velocity -10 m/s: impact at t = 0.1/10 = 0.01 s.
    rp = Node(start_label + 999999, np.array([0.0, 15.0 - 4.9, 0.0]))
    theModel.nodes[rp.label] = rp
    theModel.nodeSets["rigid_rp"] = NodeSet("rigid_rp", [rp])
    rp.fields["displacement"] = FieldVariable(rp, "displacement")
    rp.fields["rotation"] = FieldVariable(rp, "rotation")

    # Add rigid body nodes to the global 'all' nodeset so node fields are properly bundled
    if "all" in theModel.nodeSets:
        all_nodes = list(theModel.nodeSets["all"])
        all_nodes.extend(rigid_nodes)
        all_nodes.append(rp)
        theModel.nodeSets["all"] = NodeSet("all", all_nodes)

    # RP Kinematic Boundary Condition
    from edelweissmeshfree.stepactions.dirichlet import Dirichlet as DirichletMF

    def velocity_amp(t):
        return t

    # Total displacement = -7.0 over the step. Cylinder starts at 10, hits block at 5.
    # Impact at t = (5/7) * 0.1s = ~0.071 s.
    rp_bc = DirichletMF(
        "rp_bc", theModel.nodeSets["rigid_rp"], "displacement", {"2": -6.0}, theModel, theJournal, velocity_amp
    )

    # 3. Discrete Rigid Body
    from edelweissmeshfree.rigidbodies.discreterigidbody import DiscreteRigidBody

    rigid_body = DiscreteRigidBody("rigid_body", theModel, nSet="rigid_surface_nodes", referencePoint="rigid_rp")
    theModel.discreteRigidBodies = {"rigid_body": rigid_body}

    # 4. Contact Constraint
    contact = DiscreteRigidBodyPenaltyContactExplicitFactory(
        baseName="rigid_impact",
        filename="rigid_body.exo",
        particleCollection=theModel.particleSets["mesh_particles_all"],
        model=theModel,
        rigidBody=rigid_body,
        penaltyParameter=1e5,
        initial_offset=np.array([0.0, -4.9, 0.0]),
    )
    theModel.constraints.update(contact)
    theModel.constraintSets["rigid_impact"] = contact

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
    rigid_surface_field = theModel.nodeFields["displacement"].subset(theModel.elementSets["rigid_surface"])
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

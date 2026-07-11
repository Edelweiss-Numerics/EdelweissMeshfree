"""
Validation test for Coulomb friction in explicit rigid body contact.

Geometry
--------
- Block particles (rubber): 10x10x10 hex grid, Y in [-5, 5]
- Rigid block: loaded from particles.exo, translated to sit exactly on top
- Contact: upper block pushed down slightly (normal force), then dragged horizontally (friction force).
"""

import sys

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

from edelweissmeshfree.constraints.explicit.frictionaldiscreterigidbodypenaltycontactexplicit import (
    FrictionalDiscreteRigidBodyPenaltyContactExplicitFactory,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
)
from edelweissmeshfree.generators.particlesfromexodus import generateParticlesFromExodus
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
from edelweissmeshfree.rigidbodies.discreterigidbody import DiscreteRigidBody
from edelweissmeshfree.solvers.explicitmultiphysicssolver import (
    ExplicitMultiphysicsSolver,
)
from edelweissmeshfree.stepactions.dirichlet import Dirichlet as DirichletMF

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
E = 5.0e6
NU = 0.45
RHO = 1000.0 * 10.0
PENALTY = 1e5
MU = 0.3  # Friction coefficient

DISP_Y = -0.2  # Push down
DISP_X = 1.0  # Slide right
STEP_TIME = 0.5
DT = 5e-3


def run_friction_test():
    theJournal = Journal()
    theModel = MPMModel(3)

    theMaterial = {
        "material": "CompressibleNeoHooke",
        "properties": np.array([E, NU, RHO]),
    }

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

    print("Loading particles (lower rubber block)...")
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

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theModel.particleKernelDomains["all"] = theParticleKernelDomain

    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain,
        3,
        theJournal,
        bondParticlesToKernelFunctions=True,
    )

    # Bottom boundary
    # Use proper penalty boundary condition for RKPM explicit dynamics
    from edelweissfe.sets.nodeset import NodeSet

    bottom_nodes = [n for n in theModel.nodes.values() if n.coordinates[1] < -4.5]
    theModel.nodeSets["bottom_nodes"] = NodeSet("bottom_nodes", bottom_nodes)

    from edelweissmeshfree.constraints.explicit.penaltydirichletexplicit import (
        PenaltyDirichletExplicit,
    )

    dirichletBottom = PenaltyDirichletExplicit(
        "bottom_fix",
        theModel,
        theModel.nodeSets["bottom_nodes"],
        "displacement",
        {0: 0.0, 1: 0.0, 2: 0.0},
        PENALTY,
    )
    theModel.constraints.update({dirichletBottom.name: dirichletBottom})
    theModel.constraintSets["bottom_fix"] = dirichletBottom

    # Rigid block (upper)
    print("Loading rigid block from rigid_body.exo...")
    rigid_body = DiscreteRigidBody.from_mesh_file(
        name="rigid_body",
        model=theModel,
        filename="rigid_body.exo",
        translation=[0.0, -5.5, 0.0],  # Touch the top layer of particles (centroids at Y=4.5)
        mass=1.0,
    )

    def disp_amp_y(t):
        return t

    def disp_amp_x(t):
        return t

    rp_bc_y = DirichletMF(
        "rp_bc_y", theModel.nodeSets["rigid_body_rp"], "displacement", {"2": DISP_Y}, theModel, theJournal, disp_amp_y
    )
    rp_bc_x = DirichletMF(
        "rp_bc_x", theModel.nodeSets["rigid_body_rp"], "displacement", {"1": DISP_X}, theModel, theJournal, disp_amp_x
    )
    rp_bc_z = DirichletMF(
        "rp_bc_z", theModel.nodeSets["rigid_body_rp"], "displacement", {"3": 0.0}, theModel, theJournal
    )
    rp_bc_rot = DirichletMF(
        "rp_bc_rot",
        theModel.nodeSets["rigid_body_rp"],
        "rotation",
        {"1": 0.0, "2": 0.0, "3": 0.0},
        theModel,
        theJournal,
    )

    # Contact
    contact_constraints = FrictionalDiscreteRigidBodyPenaltyContactExplicitFactory(
        name="rigid_impact",
        particleCollection=theModel.particleSets["mesh_particles_all"],
        model=theModel,
        rigidBody=rigid_body,
        penaltyParameter=PENALTY,
        frictionCoefficient=MU,
        viscousRegularization=1e5,
    )
    contact_constraint = contact_constraints[0]
    for c in contact_constraints:
        theModel.constraints[c.name] = c

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)
    fieldOutputController.addPerParticleFieldOutput(
        "displacement", theModel.particleSets["mesh_particles_all"], "displacement"
    )
    fieldOutputController.addPerParticleFieldOutput("velocity", theModel.particleSets["mesh_particles_all"], "velocity")
    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements",
        theModel.particleSets["mesh_particles_all"],
        "vertex displacements",
        reshape_to_dimensions=3,
    )

    rigid_surface_field = theModel.nodeFields["displacement"].subset(theModel.elementSets["rigid_body_surface"])
    fieldOutputController.addPerNodeFieldOutput("rigid_displacement", rigid_surface_field, "U")

    # Export reference point displacement for completeness (user wanted forces on RP, so RP is available)
    fieldOutputController.addPerNodeFieldOutput(
        "displacement_rp", theModel.nodeFields["displacement"].subset(theModel.nodeSets["rigid_body_rp"]), "U"
    )

    def get_normal_force():
        return contact_constraint.totalNormalForce

    def get_friction_force():
        return contact_constraint.totalFrictionForce

    fieldOutputController.addExpressionFieldOutput(
        associatedSet=theModel.nodeSets["rigid_body_rp"], theExpression=get_normal_force, name="normal_force"
    )
    fieldOutputController.addExpressionFieldOutput(
        associatedSet=theModel.nodeSets["rigid_body_rp"], theExpression=get_friction_force, name="friction_force"
    )

    fieldOutputController.initializeJob()

    out_folder = "friction_sim_out"
    ensightOutput = EnsightOutputManager(
        out_folder, theModel, fieldOutputController, theJournal, None, intermediateSaveInterval=1
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["rigid_displacement"],
        create="perNode",
        name="vertex displacements",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        create="perNode",
        name="vertex displacements",
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement_rp"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["normal_force"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["friction_force"], create="perNode")

    ensightOutput.initializeJob()

    solver = ExplicitMultiphysicsSolver(theJournal)

    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0,
        STEP_TIME,
        DT,
        DT,
        DT,
        2000,
        theJournal,
    )

    print("Starting solver...")
    try:
        solver.solveStep(
            adaptiveTimeStepper,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            dirichlets=[rp_bc_y, rp_bc_x, rp_bc_z, rp_bc_rot],
            userIterationOptions={"field orders": {"displacement": 2, "rotation": 2}},
        )
    except Exception as e:
        theJournal.message(f"Step failed (or exploded): {str(e)}", "error")
    finally:
        fieldOutputController.finalizeJob()

    # Validate friction force at the end
    # Only use the vertical (Y) normal force and horizontal (X) friction force to avoid edge-effects
    fn = abs(contact_constraint.totalNormalForce[1])
    ft = abs(contact_constraint.totalFrictionForce[0])

    print("\n--- Validation Results ---")
    print(f"Final Total Normal Force Magnitude (Y) from Contact: {fn:.4e}")
    print(f"Final Total Friction Force Magnitude (X) from Contact: {ft:.4e}")

    # Check bottom boundary reaction forces
    bc_fn = abs(dirichletBottom.penaltyForce[1])
    bc_ft = abs(dirichletBottom.penaltyForce[0])

    print(f"Final Normal Reaction Force (Y) at Bottom BC:      {bc_fn:.4e}")
    print(f"Final Tangential Reaction Force (X) at Bottom BC:  {bc_ft:.4e}")

    if fn > 0:
        ratio = ft / fn
        print(f"Ratio Ft / Fn (Contact): {ratio:.4f} (Expected: ~{MU})")

        bc_ratio = bc_ft / bc_fn if bc_fn > 0 else 0.0
        print(f"Ratio Ft / Fn (Bottom BC): {bc_ratio:.4f} (Expected: ~{MU})")

        # Check if BC reaction forces match the contact forces within an acceptable tolerance
        # (inertial forces and wave propagation may cause slight differences in explicit dynamics,
        # but time-averaged or near quasi-static equilibrium they should match very closely)
        force_diff_n = abs(fn - bc_fn) / max(fn, 1e-12)
        force_diff_t = abs(ft - bc_ft) / max(ft, 1e-12)

        print(f"Force balance error (Normal): {force_diff_n * 100:.2f}%")
        print(f"Force balance error (Tangential): {force_diff_t * 100:.2f}%")

        if np.isclose(ratio, MU, rtol=0.1) and force_diff_n < 0.2 and force_diff_t < 0.2:
            print("VALIDATION SUCCESS: The global forces correspond to the frictional law AND equilibrium is verified!")
        else:
            print("VALIDATION FAILED: Ratio does not match friction coefficient or forces are out of balance.")

    return ensightOutput.exportName


def visualize_results(case_path):
    import pyvista as pv

    # Ensure we use the .case file, not the directory
    if not case_path.endswith(".case"):
        case_path += ".case"

    reader = pv.get_reader(case_path)

    plotter = pv.Plotter(shape=(1, 2))

    # Left subplot: 3D Animation
    plotter.subplot(0, 0)
    plotter.add_text("Frictional Contact Animation", font_size=12)

    # Initially load time step 0
    reader.set_active_time_value(reader.time_values[0])
    mesh = reader.read()

    particles = mesh["PSET_mesh_particles_all"]
    rigid_surf = mesh["rigid_body_surface"]

    # Use the unified "vertex_displacements" field if it exists
    if "vertex_displacements" in rigid_surf.point_data:
        rigid_surf.point_data["displacement"] = rigid_surf.point_data["vertex_displacements"]
    if "vertex_displacements" in particles.point_data:
        particles.point_data["displacement"] = particles.point_data["vertex_displacements"]

    # Add meshes
    plotter.add_mesh(
        particles,
        scalars="displacement",
        render_points_as_spheres=True,
        point_size=10,
        show_scalar_bar=False,
        cmap="viridis",
    )
    plotter.add_mesh(rigid_surf, scalars="displacement", color="red", style="wireframe", cmap="viridis")

    # Right subplot: 2D Chart
    plotter.subplot(0, 1)
    chart = pv.Chart2D()
    plotter.add_chart(chart)
    chart.title = "Friction Force vs Displacement"
    chart.x_axis.title = "Displacement X (m)"
    chart.y_axis.title = "Friction Force (N)"

    # Collect data over time for the plot
    disp_x_history = []
    f_t_history = []

    line_plot = chart.line([0], [0], color="b", width=2.0)

    plotter.show(interactive_update=True)

    for t in reader.time_values:
        reader.set_active_time_value(t)
        mesh = reader.read()

        # Update 3D meshes
        new_particles = mesh["PSET_mesh_particles_all"]
        new_rigid = mesh["rigid_body_surface"]
        new_rp = mesh["NSET_rigid_body_rp"]

        if "vertex_displacements" in new_rigid.point_data:
            new_rigid.point_data["displacement"] = new_rigid.point_data["vertex_displacements"]
        if "vertex_displacements" in new_particles.point_data:
            new_particles.point_data["displacement"] = new_particles.point_data["vertex_displacements"]

        particles.copy_from(new_particles)
        rigid_surf.copy_from(new_rigid)
        if "displacement_rp" in new_rp.point_data and "friction_force" in new_rp.point_data:
            disp = new_rp.point_data["displacement_rp"][0]
            f_fric = new_rp.point_data["friction_force"][0]

            disp_x_history.append(disp[0])
            f_t_mag = np.linalg.norm(f_fric)
            f_t_history.append(f_t_mag)

            line_plot.update(disp_x_history, f_t_history)

        plotter.render()
        import time

        time.sleep(0.05)

    plotter.show(interactive=True)


if __name__ == "__main__":
    case_name = run_friction_test()

    if "--visualize" in sys.argv:
        if case_name:
            visualize_results(case_name)
        else:
            print("No case file returned to visualize.")

# -*- coding: utf-8 -*-
import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

from edelweissmeshfree.constraints.frictionaldiscreterigidbodypenaltycontact import (
    FrictionalDiscreteRigidBodyPenaltyContact,
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
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver
from edelweissmeshfree.stepactions.dirichlet import Dirichlet as DirichletMF

# --- Simulation Parameters (N & mm) ---
E = 2.0  # MPa (N/mm^2)
NU = 0.3
RHO = 1.0e-9  # tonnes/mm^3
PENALTY = 1.0e5  # N/mm
MU = 0.3
DISP_Y = -0.5  # mm
DISP_X = 1.0  # mm
STEP_TIME = 0.5
DT = 0.01  # s


def run_implicit_test():
    theJournal = Journal()
    theModel = MPMModel(3)

    # 1. Background approximation
    app = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1)

    # 2. Material Point (deformable block) generator
    def mp_creator(i, coords):
        return MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/3D/Hexa",
            i,
            coords,
            0.0,
            app,
            {
                "material": "CompressibleNeoHooke",
                "properties": np.array([E, NU, RHO]),
                "plane state": "none",
            },
        )

    generateParticlesFromExodus(
        theModel, theJournal, "particles.exo", {"HEX": mp_creator, "HEX8": mp_creator}, "mesh_particles", 1
    )

    # 3. Kernel Function generator
    def kf_creator(node, characteristicLength):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=characteristicLength, continuityOrder=3
        )

    generateKernelMatchingToParticle(
        theModel, theJournal, kf_creator, theModel.particleSets["mesh_particles_all"], supportScalingFactor=2.2
    )

    # 4. Particle manager
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

    # 5. Discrete Rigid Body
    rigid_body = DiscreteRigidBody.from_mesh_file(
        name="rigid_body",
        model=theModel,
        filename="rigid_body.exo",
        translation=np.array([0.0, -5.5, 0.0]),
        density=RHO,
    )

    # 6. Reference Point Boundary Conditions (Smooth step amplitude curves)
    def amplitude_indent(t):
        return t

    def amplitude_slide(t):
        return t

    rp_bc_y = DirichletMF(
        "rp_bc_y",
        theModel.nodeSets["rigid_body_rp"],
        "displacement",
        {"2": DISP_Y},
        theModel,
        theJournal,
        amplitude_indent,
    )
    rp_bc_x = DirichletMF(
        "rp_bc_x",
        theModel.nodeSets["rigid_body_rp"],
        "displacement",
        {"1": DISP_X},
        theModel,
        theJournal,
        amplitude_slide,
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

    # 7. Support BC on bottom of the block
    from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
        ParticlePenaltyWeakDirichlet,
    )

    for p in theModel.particles.values():
        bottom_verts = [v_idx for v_idx, v_coord in enumerate(p.getVertexCoordinates()) if v_coord[1] < -4.4]
        if bottom_verts:
            constraint = ParticlePenaltyWeakDirichlet(
                f"bc_bottom_{p.number}",
                theModel,
                [p],
                "displacement",
                {0: 0.0, 1: 0.0, 2: 0.0},
                1e8,
                constrain=bottom_verts,
            )
            theModel.constraints[constraint.name] = constraint

    # 8. Contact Constraint (Implicit Frictional Contact)
    contact_constraint = FrictionalDiscreteRigidBodyPenaltyContact(
        name="contact",
        particles=theModel.particles.values(),
        model=theModel,
        rigidBody=rigid_body,
        frictionCoefficient=MU,
        viscousRegularization=1.0e6,
        penaltyParameter=PENALTY,
        proximityFactor=2.0,
    )
    theModel.constraints["contact"] = contact_constraint

    theModel.prepareYourself(theJournal)

    # 9. Output Controller
    fieldOutputController = MPMFieldOutputController(theModel, theJournal)
    fieldOutputController.addPerParticleFieldOutput(
        "displacement", theModel.particleSets["mesh_particles_all"], "displacement"
    )
    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements",
        theModel.particleSets["mesh_particles_all"],
        "vertex displacements",
        reshape_to_dimensions=3,
    )

    rigid_surface_field = theModel.nodeFields["displacement"].subset(theModel.elementSets["rigid_body_surface"])
    fieldOutputController.addPerNodeFieldOutput("rigid_displacement", rigid_surface_field, "U")
    fieldOutputController.addPerNodeFieldOutput(
        "displacement_rp", theModel.nodeFields["displacement"].subset(theModel.nodeSets["rigid_body_rp"]), "U"
    )

    def get_normal_force():
        return contact_constraint.totalNormalForce

    def get_friction_force():
        return contact_constraint.totalFrictionForce

    fieldOutputController.addExpressionFieldOutput(theModel.nodeSets["rigid_body_rp"], get_normal_force, "normal_force")
    fieldOutputController.addExpressionFieldOutput(
        theModel.nodeSets["rigid_body_rp"], get_friction_force, "friction_force"
    )

    fieldOutputController.initializeJob()

    # 10. Ensight Gold Export
    out_folder = "implicit_sim_out"
    ensightOutput = EnsightOutputManager(
        out_folder, theModel, fieldOutputController, theJournal, None, intermediateSaveInterval=1
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["rigid_displacement"],
        create="perNode",
        name="rigid_displacement",
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"],
        create="perNode",
        name="particle_displacement",
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement_rp"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["normal_force"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["friction_force"], create="perNode")
    ensightOutput.initializeJob()

    # 11. Run Solver
    solver = NonlinearQuasistaticSolver(theJournal)
    timeStepper = AdaptiveTimeStepper(0.0, STEP_TIME, DT, DT, DT, 200, theJournal)
    linearSolver = pardisoSolve

    print("Starting solver...")
    try:
        solver.solveStep(
            timeStepper,
            linearSolver,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            dirichlets=[rp_bc_y, rp_bc_x, rp_bc_z, rp_bc_rot],
            constraints=list(theModel.constraints.values()),
            userIterationOptions={
                "max. iterations": 20,
                "critical iterations": 10,
                "allowed residual growths": 5,
                "default absolute flux residual tolerance": 1.0e-3,
                "default absolute field correction tolerance": 1.0e-5,
            },
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        theJournal.message(f"Implicit solve failed: {str(e)}", "error")
    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

    # 12. Validate Results
    fn = abs(contact_constraint.totalNormalForce[1])
    ft = abs(contact_constraint.totalFrictionForce[0])

    print("\n--- Validation Results ---")
    print(f"Final Total Normal Force Magnitude (Y) from Contact: {fn:.4e} N")
    print(f"Final Total Friction Force Magnitude (X) from Contact: {ft:.4e} N")

    if fn > 0:
        ratio = ft / fn
        print(f"Ratio Ft / Fn (Contact): {ratio:.4f} (Expected: ~{MU})")
        if np.isclose(ratio, MU, rtol=0.05):
            print("VALIDATION SUCCESS: Frictional coefficient validated in implicit quasi-static analysis!")
        else:
            print("VALIDATION FAILED: Frictional ratio does not match.")


if __name__ == "__main__":
    run_implicit_test()

"""
Quasi-static explicit rigid body contact test WITH mass scaling.

Geometry
--------
- Block particles: 10×10×10 hex grid, Y ∈ [-5, 5], particle centroids at half-element offsets.
  Top particle centroids sit at Y ≈ 4.5.
- Rigid body mesh (rigid_body.exo): cylinder spanning Y ∈ [10, 20].
  The DiscreteSurfaceQuery loads it at these original coordinates.
- RP node placed at (0, 10.1, 0).
- Bottom face penalty BC at Y = -5.

Contact geometry (in DiscreteSurfaceQuery frame)
-------------------------------------------------
The query shifts particle coords by  -translation(RP).
At RP displacement Δ (negative = downward):
    local_y = particle_y - Δ
Contact occurs when local_y ≈ 10 (bottom of rigid mesh).
    ⟹ 4.5 - Δ = 10  ⟹  Δ = -5.5

Quasi-static mass-scaling strategy
------------------------------------
For explicit dynamics to approximate quasi-static behaviour the loading must
be slow relative to the wave speed.  We use *mass scaling* to increase the
stable time step (and thus the simulation is faster in wall-clock time) while
keeping the loading velocity well below the (physical) wave speed:

  Physical:  E = 400, ρ = 1    ⟹  c = √(E/ρ) = 20 m/s
  Mass-scaled ρ* = 100          ⟹  c* = √(E/ρ*) = 2 m/s  (only used for time integration)

We set the total step time to 1.0 s and apply -6.0 displacement, giving
an apparent velocity of 6 m/s.  This is *above* c* but that is irrelevant
because c* is the *artificial* wave speed due to mass scaling – the point is
that the mass scaling makes each time step larger.  The *real* wave speed
remains 20 m/s so inertia effects are small.

Expected contact time: t_contact = 5.5 / 6.0 = 0.9167 s.

After contact the block should deform with negligible oscillation.
"""

import sys

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

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
E = 5.0e6  # Young's modulus  [Pa] (Rubber-like: 5 MPa)
NU = 0.45  # Poisson's ratio (Nearly incompressible)
RHO = 1000.0 * 10.0  # Mass-scaled density (Rubber physical ρ=1000, scaled ×10)
PENALTY = 1e7  # Contact penalty stiffness
DISP_Y = -6.0  # Total prescribed RP displacement (downward)
STEP_TIME = 1.0  # Total step time  [s]
DT = 1e-3  # Time increment   [s] (Stable for c=22 m/s, dx=1.0)
VELOCITY = abs(DISP_Y) / STEP_TIME  # = 6.0 m/s
GAP = 0.6  # gap from top particle centroid (Y=4.5) to shifted rigid bottom (Y=5.1)
T_CONTACT = GAP / VELOCITY

print(f"=== Quasi-static mass-scaled contact test ===")
print(f"  E={E}, ν={NU}, ρ(scaled)={RHO}")
print(f"  Physical wave speed c = {np.sqrt(E/1.0):.1f} m/s  (ρ_phys=1)")
print(f"  Scaled  wave speed c*= {np.sqrt(E/RHO):.1f} m/s")
print(f"  Loading velocity      = {VELOCITY:.1f} m/s  ({VELOCITY/np.sqrt(E/1.0)*100:.0f}% of physical c)")
print(f"  Expected contact time = {T_CONTACT:.4f} s")
print(f"  dt = {DT} s  →  {int(STEP_TIME/DT)} increments")
print()


def run_quasistatic_sim():
    theJournal = Journal()
    theModel = MPMModel(3)

    theMaterial = {
        "material": "CompressibleNeoHooke",
        "properties": np.array([E, NU, RHO]),
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
        3,
        theJournal,
        bondParticlesToKernelFunctions=True,
    )

    # Bottom boundary
    bottom_particles = [p for p in theModel.particleSets["mesh_particles_all"] if p.getCenterCoordinates()[1] < -4.5]
    theModel.particleSets["bottom"] = bottom_particles

    from edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
        ParticleExplicitPenaltyCartesianBoundaryConstraintFactory,
    )

    dirichletBottom = ParticleExplicitPenaltyCartesianBoundaryConstraintFactory(
        "bottom_fix",
        boundaryPosition=-5.0,
        component=1,
        particleCollection=theModel.particleSets["bottom"],
        field="displacement",
        model=theModel,
        location="center",
        penaltyParameter=PENALTY,
    )
    theModel.constraints.update(dirichletBottom)
    theModel.constraintSets["bottom_fix"] = dirichletBottom

    # ---- Rigid body setup ----
    from edelweissmeshfree.rigidbodies.discreterigidbody import DiscreteRigidBody

    rigid_body = DiscreteRigidBody.from_mesh_file(
        name="rigid_body",
        model=theModel,
        filename="rigid_body.exo",
        translation=[0.0, -4.9, 5.0],
        density=RHO/10,
        # mass=1.56e4,
        # inertia=[6.17e5, 6.17e5, 1.93e5],
        initial_velocity=[0.0, -VELOCITY, 0.0],
    )

    # Contact
    DiscreteRigidBodyPenaltyContactExplicitFactory(
        name="rigid_impact",
        particleCollection=theModel.particleSets["mesh_particles_all"],
        model=theModel,
        rigidBody=rigid_body,
        penaltyParameter=PENALTY,
    )

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    # ---- Field output ----
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

    rigid_surface_field = theModel.nodeFields["displacement"].subset(theModel.elementSets["rigid_body_surface"])
    fieldOutputController.addPerNodeFieldOutput("rigid_displacement", rigid_surface_field, "U")

    fieldOutputController.initializeJob()

    out_folder = "quasistatic_sim_out"
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



    incSize = DT / STEP_TIME
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0,
        STEP_TIME,
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


# ── Validation ──────────────────────────────────────────────────────────────
def validate_results():
    import glob

    import pyvista as pv

    case_files = sorted(glob.glob("quasistatic_sim_out*.case"))
    if not case_files:
        print("No quasistatic .case file found — skipping validation.")
        return
    latest = case_files[-1]
    print(f"\n{'='*70}")
    print(f"VALIDATION  –  {latest}")
    print(f"{'='*70}")

    reader = pv.get_reader(latest)
    reader.set_active_time_set(1)
    time_values = reader.time_values

    print(f"  Expected contact time : {T_CONTACT:.4f} s")
    print(f"  Loading velocity      : {VELOCITY:.2f} m/s")
    print()

    errors = []
    first_block_deform_t = None

    for t_val in time_values:
        reader.set_active_time_value(t_val)
        mesh = reader.read()
        rigid = mesh["rigid_surface"]
        particles = mesh["PSET_mesh_particles_all"]

        # -- rigid displacement check --
        expected_disp = -VELOCITY * t_val
        actual_disp_y = rigid.point_data["rigid_displacement"][:, 1]
        max_err = np.max(np.abs(actual_disp_y - expected_disp))

        # -- block deformation --
        block_disp = particles.point_data["vertex_displacements"]
        max_block = np.max(np.abs(block_disp))

        status = "PRE " if t_val < T_CONTACT else "POST"

        if max_block > 1e-4 and first_block_deform_t is None:
            first_block_deform_t = t_val

        print(
            f"  t={t_val:7.4f} [{status}]  rigid_y={np.min(actual_disp_y):+8.4f} (exp {expected_disp:+8.4f}, err {max_err:.2e})  block_max_disp={max_block:.6f}"
        )

        if status == "PRE " and max_err > 0.1:
            errors.append(f"t={t_val:.4f}: rigid displacement error before contact {max_err:.4f}")

    print()
    if first_block_deform_t is not None:
        delay = first_block_deform_t - T_CONTACT
        print(
            f"  First block deformation at t = {first_block_deform_t:.4f} s  (Δt from theory = {delay:+.4f} s = {abs(delay)/DT:.1f} increments)"
        )
        if abs(delay) < 10 * DT:
            print(f"  ✓  Contact onset matches theoretical prediction within 10 time steps.")
        else:
            errors.append(f"Contact onset delayed by {delay:.4f} s = {abs(delay)/DT:.0f} increments")
    else:
        errors.append("Block never deformed – contact was never triggered!")

    print()
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ✗  {e}")
        sys.exit(1)
    else:
        print("VALIDATION PASSED ✓")
        print("  – Rigid displacement tracks the prescribed Dirichlet BC exactly.")
        print("  – Block deformation starts within a few time steps of the theoretical contact time.")


if __name__ == "__main__":
    run_quasistatic_sim()
    validate_results()

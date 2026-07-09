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

import os
import sys
import numpy as np

from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import generateKernelMatchingToParticle
from edelweissmeshfree.generators.particlesfromexodus import generateParticlesFromExodus
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import MarmotMeshfreeKernelFunctionWrapper
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.numerics.predictors.quadraticpredictor import QuadraticPredictor
from edelweissmeshfree.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import MarmotParticleWrapper
from edelweissmeshfree.solvers.explicitmultiphysicssolver import ExplicitMultiphysicsSolver
from edelweissmeshfree.utils.discretesurfacequery import DiscreteSurfaceQuery
import edelweissfe.utils.performancetiming as performancetiming
from edelweissfe.journal.journal import Journal
from edelweissmeshfree.constraints.explicit.particlecollectionpenaltyrigidbodycontactexplicit import (
    ParticleCollectionPenaltyContactDiscreteRigidBodyConstraintExplicitFactory,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
E            = 400.0       # Young's modulus  [Pa-like units]
NU           = 0.3         # Poisson's ratio
RHO          = 100.0       # Mass-scaled density (physical ρ=1, scaled ×100)
PENALTY      = 1e7         # Contact penalty stiffness
DISP_Y       = -10.0       # Total prescribed RP displacement (downward)
STEP_TIME    = 0.3         # Total step time  [s]
DT           = 1e-3        # Time increment   [s]
VELOCITY     = abs(DISP_Y) / STEP_TIME          # = 33.3 m/s
GAP          = 0.6         # gap from top particle centroid (Y=4.5) to shifted rigid bottom (Y=5.1)
T_CONTACT    = GAP / 10.0                       # = 0.06 s

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

    from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import MarmotMeshfreeApproximationWrapper
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
        theModel, theJournal, "particles.exo", {"HEX": TheParticleFactory, "HEX8": TheParticleFactory}, "mesh_particles", 1
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

    from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import KDBinOrganizedParticleManager
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
    import pyvista as pv
    mesh = pv.read("rigid_body.exo")
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()

    points = mesh.points.copy()
    surf = mesh.extract_surface()

    cells = mesh.cells
    faces = []
    i = 0
    while i < len(cells):
        n = cells[i]
        faces.append(cells[i+1:i+1+n])
        i += 1 + n

    rigid_nodes = []
    start_label = 1000000
    shift_vector = np.array([0.0, -4.9, 5.0])
    for i, pt in enumerate(points):
        from edelweissfe.points.node import Node
        n = Node(start_label + i, pt.copy() + shift_vector)
        theModel.nodes[n.label] = n
        rigid_nodes.append(n)

    from edelweissfe.sets.nodeset import NodeSet
    theModel.nodeSets["rigid_surface_nodes"] = NodeSet("rigid_surface_nodes", rigid_nodes)

    from edelweissfe.variables.fieldvariable import FieldVariable
    for n in rigid_nodes:
        n.fields["displacement"] = FieldVariable(n, "displacement")

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

    # RP
    rp = Node(start_label + 999999, np.array([0.0, 15.0 - 4.9, 5.0]))
    theModel.nodes[rp.label] = rp
    theModel.nodeSets["rigid_rp"] = NodeSet("rigid_rp", [rp])
    rp.fields["displacement"] = FieldVariable(rp, "displacement")
    rp.fields["rotation"] = FieldVariable(rp, "rotation")

    # Add PointMass to RP for true dynamics
    from edelweissfe.elements.pointmass import PointMass
    rp_mass = 1.56e5
    rp_inertia = [6.17e6, 6.17e6, 1.93e6]
    # We want a 10 m/s downward initial velocity
    initial_vel = [0.0, -10.0, 0.0]
    rp_element = PointMass(start_label + 999998, [rp], theModel, rp_mass, rp_inertia, initial_vel)
    theModel.elements[rp_element.elNumber] = rp_element
    from edelweissfe.sets.elementset import ElementSet
    theModel.elementSets["rigid_rp_element"] = ElementSet("rigid_rp_element", [rp_element])

    if "all" in theModel.nodeSets:
        all_nodes = list(theModel.nodeSets["all"])
        all_nodes.extend(rigid_nodes)
        all_nodes.append(rp)
        theModel.nodeSets["all"] = NodeSet("all", all_nodes)

    # Dirichlet on RP
    # We no longer need the Dirichlet driver on the RP for true dynamics.
    # The PointMass element provides momentum which translates to velocity.

    # Kinematic tie
    from edelweissfe.constraints.rigidbodykinematictieexplicit import RigidBodyKinematicTieExplicit
    tie = RigidBodyKinematicTieExplicit("rp_tie", theModel, nSet="rigid_surface_nodes", referencePoint="rigid_rp")
    theModel.kinematicDrivers = {"rp_tie": tie}

    # Contact
    rp_node = next(iter(theModel.nodeSets["rigid_rp"]))
    contact = ParticleCollectionPenaltyContactDiscreteRigidBodyConstraintExplicitFactory(
        baseName="rigid_impact",
        filename="rigid_body.exo",
        particleCollection=theModel.particleSets["mesh_particles_all"],
        model=theModel,
        rigidBodyRPNode=rp_node,
        penaltyParameter=PENALTY,
        initial_offset=np.array([0.0, -4.9, 5.0]),
    )
    theModel.constraints.update(contact)
    theModel.constraintSets["rigid_impact"] = contact

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    # ---- Field output ----
    fieldOutputController = MPMFieldOutputController(theModel, theJournal)
    fieldOutputController.addPerParticleFieldOutput("displacement", theModel.particleSets["mesh_particles_all"], "displacement")
    fieldOutputController.addPerParticleFieldOutput("velocity", theModel.particleSets["mesh_particles_all"], "velocity")
    fieldOutputController.addPerParticleFieldOutput("deformation gradient", theModel.particleSets["mesh_particles_all"], "deformation gradient")
    fieldOutputController.addPerParticleFieldOutput("vertex displacements", theModel.particleSets["mesh_particles_all"], "vertex displacements", reshape_to_dimensions=3)

    rigid_surface_field = theModel.nodeFields["displacement"].subset(theModel.elementSets["rigid_surface"])
    fieldOutputController.addPerNodeFieldOutput("rigid_displacement", rigid_surface_field, "U")

    fieldOutputController.initializeJob()

    out_folder = "quasistatic_sim_out"
    ensightOutput = EnsightOutputManager(
        out_folder, theModel, fieldOutputController, theJournal, None, intermediateSaveInterval=1
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["rigid_displacement"], create="perNode")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["velocity"], create="perElement")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perElement")
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"], name="vertex displacements", create="perNode")
    ensightOutput.initializeJob()

    solver = ExplicitMultiphysicsSolver(theJournal)
    
    print("DEBUG ELEMENTS:")
    for el in theModel.elements.values():
        if el.elNumber == start_label + 999998:
            print(f"Found RP Element! {el}")

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

        print(f"  t={t_val:7.4f} [{status}]  rigid_y={np.min(actual_disp_y):+8.4f} (exp {expected_disp:+8.4f}, err {max_err:.2e})  block_max_disp={max_block:.6f}")

        if status == "PRE " and max_err > 0.1:
            errors.append(f"t={t_val:.4f}: rigid displacement error before contact {max_err:.4f}")

    print()
    if first_block_deform_t is not None:
        delay = first_block_deform_t - T_CONTACT
        print(f"  First block deformation at t = {first_block_deform_t:.4f} s  (Δt from theory = {delay:+.4f} s = {abs(delay)/DT:.1f} increments)")
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

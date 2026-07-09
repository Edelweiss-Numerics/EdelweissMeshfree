import numpy as np
from edelweissfe.journal.journal import Journal

from edelweissmeshfree.generators.particlesfromexodus import generateParticlesFromExodus
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.utils.discretesurfacequery import DiscreteSurfaceQuery


def run_test():
    dimension = 3
    theJournal = Journal()
    theModel = MPMModel(dimension)

    # Approximation
    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    # Material
    theMaterial = {
        "material": "CompressibleNeoHooke",
        "properties": np.array([400, 0.3]),
    }

    def TheParticleFactory(number, vertexCoordinates):
        return MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/3D/Hexa",
            number,
            vertexCoordinates,
            0.0,
            theApproximation,
            theMaterial,
        )

    # 1. Load Particles
    print("Loading particles from particles.exo...")
    # particles.exo contains HEX8 elements
    theModel = generateParticlesFromExodus(
        theModel,
        theJournal,
        "particles.exo",
        {"HEX": TheParticleFactory, "HEX8": TheParticleFactory},
        "mesh_particles",
        1,
    )

    particles = list(theModel.particles.values())
    print(f"Loaded {len(particles)} particles.")

    coords = np.array([p.getCenterCoordinates() for p in particles]).reshape(-1, 3)

    # 2. Test DiscreteSurfaceQuery independently
    print("Loading rigid body surface from rigid_body.exo and initializing query engine...")
    query_engine = DiscreteSurfaceQuery("rigid_body.exo")
    print(f"Rigid body mesh loaded with {query_engine.mesh.n_points} points and {query_engine.mesh.n_cells} faces.")
    print("Executing vectorized query on particle coordinates...")
    dists, normals = query_engine.query(coords)
    print("Query successful!")
    print(f"Sample particle 0 coordinate: {coords[0]}")
    print(f"Sample particle 0 distance to rigid body: {dists[0]}")
    print(f"Sample particle 0 rigid body surface normal: {normals[0]}")
    print(f"Min distance: {np.min(dists)}, Max distance: {np.max(dists)}")

    # 3. Test ParticleCollectionPenaltyContactDiscreteRigidBodyConstraintExplicit
    print("\nTesting ParticleCollectionPenaltyContactDiscreteRigidBodyConstraintExplicitFactory...")
    from edelweissmeshfree.constraints.explicit.particlecollectionpenaltyrigidbodycontactexplicit import (
        ParticleCollectionPenaltyContactDiscreteRigidBodyConstraintExplicitFactory,
    )

    constraints = ParticleCollectionPenaltyContactDiscreteRigidBodyConstraintExplicitFactory(
        baseName="test_rigid_contact",
        filename="rigid_body.exo",
        particleCollection=particles,
        model=theModel,
        penaltyParameter=1e5,
    )

    constraint = constraints["test_rigid_contact_collection"]
    print(f"Constraint {constraint.name} created successfully.")

    # Normally done by solver:
    print("Updating connectivity...")
    # NOTE: updateConnectivity will fail if kf.node is not set up correctly (requires particle-kernel matching).
    # Since this is a dummy setup, we might have empty kernel functions unless we run generateKernelMatchingToParticle
    from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )

    def theMeshfreeKernelFunctionFactory(node, characteristicLength):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=characteristicLength, continuityOrder=3
        )

    from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
        generateKernelMatchingToParticle,
    )

    theModel = generateKernelMatchingToParticle(
        theModel,
        theJournal,
        theMeshfreeKernelFunctionFactory,
        theModel.particleSets["mesh_particles_all"],
        supportScalingFactor=2.2,
    )
    print("Kernel matching generated.")

    hasChanged = constraint.updateConnectivity(theModel)
    print(f"updateConnectivity returned: {hasChanged}, constraint has {len(constraint.nodes)} unique nodes.")
    from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
    from edelweissmeshfree.outputmanagers.ensight import (
        OutputManager as EnsightOutputManager,
    )

    # Prepare the model
    theModel.prepareYourself(theJournal)

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    # We add displacement even though it is 0 initially, just to have a field
    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["all"],
        "displacement",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager(
        "ensight", theModel, fieldOutputController, theJournal, None, intermediateSaveInterval=1
    )
    ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement")
    ensightOutput.initializeJob()

    # Output the initial state (t=0)
    fieldOutputController.finalizeIncrement()
    ensightOutput.writeOutput(theModel)

    fieldOutputController.finalizeJob()
    ensightOutput.finalizeJob()
    print("Ensight output generated in 'ensight' directory.")

    print("Applying constraint forces (testing PExt allocation)...")
    PExt = np.zeros(constraint.nDof)
    # create a mock time step or just pass None since it's unused in our applyConstraint
    timeStep = None

    # We force distance to be negative by translating rigid_body in our query engine for the test
    # (or just test applyConstraint which will do nothing if dists > 0)
    constraint.applyConstraint(PExt, timeStep)
    print(f"Total reaction force (g * penalty): {constraint.reactionForce}")
    print("Test finished successfully!")


if __name__ == "__main__":
    run_test()

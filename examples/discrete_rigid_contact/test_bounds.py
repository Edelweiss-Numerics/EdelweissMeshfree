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

m = MPMModel(3)
app = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 3, completenessOrder=1)


def f(i, c):
    return MarmotParticleWrapper(
        "Displacement/SQCNIxNSNI/3D/Hexa",
        i,
        c,
        0.0,
        app,
        {"material": "CompressibleNeoHooke", "properties": np.array([1.0, 1.0, 1.0])},
    )


generateParticlesFromExodus(m, Journal(), "particles.exo", {"HEX": f, "HEX8": f}, "mp", 1)
y = [p.getCenterCoordinates()[1] for p in m.particles.values()]
print("Y range:", min(y), max(y))

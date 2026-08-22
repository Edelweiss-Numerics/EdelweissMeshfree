# -*- coding: utf-8 -*-
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
#  Thomas Mader    |  thomas.mader@bokut.ac.at
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

import itertools
from typing import Any, List, Set, Tuple, Union

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    getThreadPool,
    isFreeThreadingSupported,
)
from numpy.typing import NDArray

from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.particlemanagers.base.baseparticlemanager import (
    BaseParticleManager,
)


class _FastKDBinOrganizer:
    """
    Optimized Bin Organizer using pure NumPy vectorization and integer indexing.
    """

    def __init__(self, kernelFunctions: List[Any], dimension: int) -> None:
        self._dimension = dimension

        # --- 1. Vectorized Bounding Box Extraction ---
        boundingBoxesPerKernel = [sf.getBoundingBox() for sf in kernelFunctions]

        if not boundingBoxesPerKernel:
            self._mins = np.empty((0, dimension))
            self._maxs = np.empty((0, dimension))
            self._bins = []
            self._boundingBoxMin = np.zeros(dimension)
            self._boundingBoxMax = np.zeros(dimension)
            self._nBins = np.zeros(dimension, dtype=int)
            self._binSize = np.ones(dimension)
            self._strides = np.ones(3, dtype=int)
            return

        boundingBoxes = np.array(boundingBoxesPerKernel)
        self._mins = boundingBoxes[:, 0, :]
        self._maxs = boundingBoxes[:, 1, :]

        # --- 2. Grid Setup ---
        self._boundingBoxMin = np.min(self._mins, axis=0) - 1e-12
        self._boundingBoxMax = np.max(self._maxs, axis=0) + 1e-12

        averageBoundingBoxExtent = np.mean(self._maxs - self._mins, axis=0)
        self._binSize = averageBoundingBoxExtent / 2.0

        self._nBins = np.ceil((self._boundingBoxMax - self._boundingBoxMin) / self._binSize).astype(int)

        self._strides = np.ones(3, dtype=int)
        if dimension >= 2:
            self._strides[1] = self._nBins[0]
        if dimension == 3:
            self._strides[2] = self._nBins[0] * self._nBins[1]

        numberOfBins = int(np.prod(self._nBins))
        self._bins = [[] for _ in range(numberOfBins)]

        # --- 3. Vectorized Bin Index Calculation ---
        lowestBinIndexPerKernel = ((self._mins - self._boundingBoxMin) / self._binSize).astype(int)
        highestBinIndexPerKernel = ((self._maxs - self._boundingBoxMin) / self._binSize).astype(int)

        # --- 4. Fill Bins ---
        _, strideAlongY, strideAlongZ = self._strides[0], self._strides[1], self._strides[2]
        bins = self._bins

        for kernelIndex, (lowestBinIndex, highestBinIndex) in enumerate(
            zip(lowestBinIndexPerKernel, highestBinIndexPerKernel)
        ):
            if dimension == 3:
                for z in range(lowestBinIndex[2], highestBinIndex[2] + 1):
                    offsetOfPlane = z * strideAlongZ
                    for y in range(lowestBinIndex[1], highestBinIndex[1] + 1):
                        offsetOfRow = offsetOfPlane + y * strideAlongY
                        start = offsetOfRow + lowestBinIndex[0]
                        end = offsetOfRow + highestBinIndex[0] + 1
                        for binIndex in range(start, end):
                            bins[binIndex].append(kernelIndex)
            elif dimension == 2:
                for y in range(lowestBinIndex[1], highestBinIndex[1] + 1):
                    offsetOfRow = y * strideAlongY
                    start = offsetOfRow + lowestBinIndex[0]
                    end = offsetOfRow + highestBinIndex[0] + 1
                    for binIndex in range(start, end):
                        bins[binIndex].append(kernelIndex)
            else:
                for binIndex in range(lowestBinIndex[0], highestBinIndex[0] + 1):
                    bins[binIndex].append(kernelIndex)

    def getCandidateIndices(self, queryBoxMin: NDArray[np.float64], queryBoxMax: NDArray[np.float64]) -> Set[int]:
        if not self._bins:
            return set()

        lowestQueriedBinIndex = ((queryBoxMin - self._boundingBoxMin) / self._binSize).astype(int)
        highestQueriedBinIndex = ((queryBoxMax - self._boundingBoxMin) / self._binSize).astype(int)

        np.maximum(lowestQueriedBinIndex, 0, out=lowestQueriedBinIndex)
        np.minimum(highestQueriedBinIndex, self._nBins - 1, out=highestQueriedBinIndex)

        bins = self._bins
        _, strideAlongY, strideAlongZ = self._strides[0], self._strides[1], self._strides[2]

        kernelListsInQueriedBins = []

        if self._dimension == 3:
            for z in range(lowestQueriedBinIndex[2], highestQueriedBinIndex[2] + 1):
                offsetOfPlane = z * strideAlongZ
                for y in range(lowestQueriedBinIndex[1], highestQueriedBinIndex[1] + 1):
                    offsetOfRow = offsetOfPlane + y * strideAlongY
                    start = offsetOfRow + lowestQueriedBinIndex[0]
                    end = offsetOfRow + highestQueriedBinIndex[0] + 1
                    for binIndex in range(start, end):
                        if bins[binIndex]:
                            kernelListsInQueriedBins.append(bins[binIndex])

        elif self._dimension == 2:
            for y in range(lowestQueriedBinIndex[1], highestQueriedBinIndex[1] + 1):
                offsetOfRow = y * strideAlongY
                start = offsetOfRow + lowestQueriedBinIndex[0]
                end = offsetOfRow + highestQueriedBinIndex[0] + 1
                for binIndex in range(start, end):
                    if bins[binIndex]:
                        kernelListsInQueriedBins.append(bins[binIndex])

        elif self._dimension == 1:
            for binIndex in range(lowestQueriedBinIndex[0], highestQueriedBinIndex[0] + 1):
                if bins[binIndex]:
                    kernelListsInQueriedBins.append(bins[binIndex])

        if not kernelListsInQueriedBins:
            return set()

        return set(itertools.chain.from_iterable(kernelListsInQueriedBins))


class KDBinOrganizedParticleManager(BaseParticleManager):
    def __init__(
        self,
        particleKernelDomain: ParticleKernelDomain,
        dimension: int,
        journal: Journal,
        bondParticlesToKernelFunctions: bool = False,
        randomlyShiftPartliceShapeFunctions: Union[bool, float] = False,
    ):

        self._meshfreeKernelFunctions = particleKernelDomain.meshfreeKernelFunctions
        self._particles = particleKernelDomain.particles
        self._dimension = dimension
        self._bondParticlesToKernelFunctions = bondParticlesToKernelFunctions
        self._journal = journal

        if isFreeThreadingSupported():
            self._numThreads = getNumberOfThreads()
        else:
            self._numThreads = 1

        # Pre-fetch labels for integer sorting
        self._kernelLabels = np.array([k.node.label for k in self._meshfreeKernelFunctions], dtype=int)

        self._particlesWithChangedKernelFunctions = []

        # If every kernel's support is exactly its bounding box, the precise support check reduces
        # to a strict box test that can be vectorised over all surviving candidates at once.
        self._allKernelsHaveBoxSupport = all(k.hasBoxSupport for k in self._meshfreeKernelFunctions)

        if not isinstance(randomlyShiftPartliceShapeFunctions, (bool, float)):
            raise ValueError("randomlyShiftPartliceShapeFunctions must be a boolean or a float.")
        self._randomlyShiftPartliceShapeFunctions = randomlyShiftPartliceShapeFunctions

        if self._bondParticlesToKernelFunctions:
            if len(self._particles) != len(self._meshfreeKernelFunctions):
                raise ValueError("The number of particles and kernel functions must be equal.")

            for particle, kernelFunction in zip(self._particles, self._meshfreeKernelFunctions):
                particleCoordinates = particle.getCenterCoordinates()
                kernelFunction.moveTo(particleCoordinates)

        self.signalizeKernelFunctionUpdate()

    @property
    def particlesWithChangedKernelFunctions(self) -> list:
        return self._particlesWithChangedKernelFunctions

    def signalizeKernelFunctionUpdate(self) -> None:
        self._theBins = _FastKDBinOrganizer(list(self._meshfreeKernelFunctions), self._dimension)

    def updateConnectivity(self) -> bool:
        if self._bondParticlesToKernelFunctions:
            self._journal.message("Updating kernel function positions...", "ParticleManager")
            for particle, kernelFunction in zip(self._particles, self._meshfreeKernelFunctions):
                particleCoordinates = particle.getCenterCoordinates()

                if self._randomlyShiftPartliceShapeFunctions:
                    if isinstance(self._randomlyShiftPartliceShapeFunctions, float):
                        particleVol = particle.getVolumeUndeformed()
                        particleSize = particleVol ** (1.0 / self._dimension)

                        randdisp = (
                            (np.random.rand(self._dimension) - 0.5)
                            * np.sqrt(particle.getVolumeUndeformed())
                            * self._randomlyShiftPartliceShapeFunctions
                            * particleSize
                        )
                        particleCoordinates += randdisp

                kernelFunction.moveTo(particleCoordinates)

            self.signalizeKernelFunctionUpdate()

        # Capture variables for closure
        allKernels = self._meshfreeKernelFunctions
        kernelBoundingBoxMins = self._theBins._mins
        kernelBoundingBoxMaxs = self._theBins._maxs
        binOrganizer = self._theBins
        kernelLabels = self._kernelLabels
        dim = self._dimension
        boxSupport = self._allKernelsHaveBoxSupport

        def processParticleChunk(particleChunk: List[Any]) -> List[Any]:
            particlesInChunkWithChangedKernelFunctions = []

            for particle in particleChunk:
                evaluationCoordinates = particle.getEvaluationCoordinates()

                # Broad Phase Min/Max Calculation
                if len(evaluationCoordinates) == 1:
                    particleBoxMin = evaluationCoordinates[0]
                    particleBoxMax = evaluationCoordinates[0]
                else:
                    particleBoxMin = np.min(evaluationCoordinates, axis=0)
                    particleBoxMax = np.max(evaluationCoordinates, axis=0)

                # 1. Grid Search
                candidateKernelIndices = binOrganizer.getCandidateIndices(particleBoxMin, particleBoxMax)

                # 2. Vectorized AABB Filter
                # np.fromiter consumes the candidate set directly; going through list() first
                # materialises a throwaway Python list of a few hundred ints per particle.
                candidateKernelIndices = np.fromiter(
                    candidateKernelIndices, dtype=np.intp, count=len(candidateKernelIndices)
                )

                candidateBoundingBoxMins = kernelBoundingBoxMins[candidateKernelIndices, :dim]
                candidateBoundingBoxMaxs = kernelBoundingBoxMaxs[candidateKernelIndices, :dim]
                particleBoxMaxInDomainDimensions = particleBoxMax[:dim]
                particleBoxMinInDomainDimensions = particleBoxMin[:dim]

                candidateBoxOverlapsParticleBox = np.all(
                    (particleBoxMaxInDomainDimensions >= candidateBoundingBoxMins)
                    & (particleBoxMinInDomainDimensions <= candidateBoundingBoxMaxs),
                    axis=1,
                )
                overlappingKernelIndices = candidateKernelIndices[candidateBoxOverlapsParticleBox]

                # 3. Precise Check (Geometric)
                # Ensure coordinates are 2D for the Cython signature
                evaluationCoordinates2D = evaluationCoordinates
                if evaluationCoordinates2D.ndim == 1:
                    # Reshape (dim,) -> (1, dim)
                    evaluationCoordinates2D = evaluationCoordinates2D.reshape(1, -1)

                if boxSupport:
                    # A kernel with box support covers a coordinate exactly when the coordinate lies
                    # strictly inside its bounding box, and the bounds of every candidate are already
                    # to hand from the broad phase -- so the whole per-candidate support query becomes
                    # one vectorised test. Both comparisons are strict: a coordinate on the boundary
                    # sits where the kernel is exactly zero and is not covered.
                    overlappingBoundingBoxMins = kernelBoundingBoxMins[overlappingKernelIndices, :dim]
                    overlappingBoundingBoxMaxs = kernelBoundingBoxMaxs[overlappingKernelIndices, :dim]
                    evaluationPoints = evaluationCoordinates2D[:, None, :dim]
                    kernelCoversParticle = np.any(
                        np.all(
                            (evaluationPoints > overlappingBoundingBoxMins[None, :, :])
                            & (evaluationPoints < overlappingBoundingBoxMaxs[None, :, :]),
                            axis=2,
                        ),
                        axis=0,
                    )
                    coveringKernelIndicesUnsorted = overlappingKernelIndices[kernelCoversParticle]
                    coveringKernelIndices = list(
                        coveringKernelIndicesUnsorted[
                            np.argsort(kernelLabels[coveringKernelIndicesUnsorted], kind="stable")
                        ]
                    )
                else:
                    coveringKernelIndices = []
                    for kernelIndex in overlappingKernelIndices:
                        candidateKernel = allKernels[kernelIndex]

                        if candidateKernel.isAnyCoordinateInSupport(evaluationCoordinates2D):
                            coveringKernelIndices.append(kernelIndex)

                    coveringKernelIndices.sort(key=lambda idx: kernelLabels[idx])
                validKernels = [allKernels[i] for i in coveringKernelIndices]

                if not validKernels:
                    raise ValueError(
                        f"Particle at {particle.getCenterCoordinates()} has no associated kernel functions after connectivity update."
                    )

                if validKernels != particle.kernelFunctions:
                    particlesInChunkWithChangedKernelFunctions.append(particle)

                particle.assignKernelFunctions(
                    validKernels
                )  # assign the kernel functions. This happens even if they are the same, because the overlap with the particle usually changes due to movement.

            return particlesInChunkWithChangedKernelFunctions

        if self._numThreads <= 1:
            changedParticlesPerChunk = [processParticleChunk(self._particles)]
        else:
            chunkSize = len(self._particles) // self._numThreads + 1
            chunks = [self._particles[i : i + chunkSize] for i in range(0, len(self._particles), chunkSize)]

            executor = getThreadPool(self._numThreads)
            changedParticlesPerChunk = list(executor.map(processParticleChunk, chunks))

        # Chunks are handed out in particle order and executor.map preserves that order, so the
        # collected list is the same for any number of threads.
        self._particlesWithChangedKernelFunctions = [
            particle for chunk in changedParticlesPerChunk for particle in chunk
        ]

        hasChanged = len(self._particlesWithChangedKernelFunctions) > 0

        return hasChanged

    def getCoveredDomain(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._theBins._boundingBoxMin, self._theBins._boundingBoxMax

    def __str__(self) -> str:
        return (
            f"KDBinOrganizedParticleManager with {len(self._particles)} particles "
            f"and {len(self._meshfreeKernelFunctions)} shape functions "
            f"in {self._dimension} dimensions. Covered domain: {self.getCoveredDomain()}."
        )

    def visualize(self) -> None:
        if self._dimension != 2:
            raise ValueError("Visualization only supported for 2D.")

        import matplotlib.pyplot as plt

        nBins = self._theBins._nBins
        nKernelFunctions = np.zeros(nBins)

        for i in range(nBins[0]):
            for j in range(nBins[1]):
                flatBinIndex = j * self._theBins._strides[1] + i
                nKernelFunctions[i, j] = len(self._theBins._bins[flatBinIndex])

        plt.figure()
        plt.imshow(nKernelFunctions.T, origin="lower")
        plt.title("Number of kernel functions in the bins")

        for i in range(nBins[0] + 1):
            plt.plot([i - 0.5, i - 0.5], [0 - 0.5, nBins[1] - 0.5], "k")
        for j in range(nBins[1] + 1):
            plt.plot([0 - 0.5, nBins[0] - 0.5], [j - 0.5, j - 0.5], "k")

        plt.colorbar()
        plt.show()

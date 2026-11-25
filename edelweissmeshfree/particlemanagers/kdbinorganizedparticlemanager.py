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

from typing import Any, List, NamedTuple, Tuple, Union

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.parallelizationutilities import mapChunksInParallel
from numpy.typing import NDArray

from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.particlemanagers.base.baseparticlemanager import (
    BaseParticleManager,
)
from edelweissmeshfree.particlemanagers.binsofkernelfunctions import (
    BinsOfKernelFunctions,
)


class _KernelFunctionsOfAChunk(NamedTuple):
    """What the search found for a chunk of consecutive particles. The offsets are those of the chunk: they start at
    0."""

    firstKernelOfParticle: NDArray[np.intp]
    kernelsOfParticles: NDArray[np.intp]
    particlesWithChangedKernelFunctions: list


def _keepKernelsWhoseSupportHoldsAPoint(
    firstKernelOfParticle: NDArray[np.intp],
    kernelsOfParticles: NDArray[np.intp],
    evaluationPointsOfParticles: List[NDArray[np.float64]],
    kernelFunctions: list,
) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The narrow phase for kernel functions whose support is smaller than their box: of the kernel functions whose box
    holds a point of a particle, keep those whose support holds one.

    Parameters
    ----------
    firstKernelOfParticle
        The offsets of the kernel functions of the particles: those of particle ``i`` are
        ``kernelsOfParticles[firstKernelOfParticle[i] : firstKernelOfParticle[i + 1]]``.
    kernelsOfParticles
        The numbers of the kernel functions whose box holds a point of the particle.
    evaluationPointsOfParticles
        The evaluation points of every particle.
    kernelFunctions
        The kernel functions, in the order of their numbers.

    Returns
    -------
    tuple[NDArray[np.intp], NDArray[np.intp]]
        The kept kernel functions, in the same layout.
    """

    isKept = np.zeros(len(kernelsOfParticles), dtype=bool)
    for i, points in enumerate(evaluationPointsOfParticles):
        for n in range(firstKernelOfParticle[i], firstKernelOfParticle[i + 1]):
            isKept[n] = kernelFunctions[kernelsOfParticles[n]].isAnyCoordinateInSupport(points)

    # the first kept kernel function of a particle comes after all kept ones of the particles before it
    numberKeptBefore = np.concatenate(([0], np.cumsum(isKept))).astype(np.intp)
    return numberKeptBefore[firstKernelOfParticle], kernelsOfParticles[isKept]


class KDBinOrganizedParticleManager(BaseParticleManager):
    """Assigns every particle the kernel functions that cover one of its evaluation points, searched in the bins of
    the kernel functions (:class:`BinsOfKernelFunctions`) in every increment, or, with a neighbour list skin, when the
    motion since the last search makes it due. See the documentation page "Particle managers" for the steps of the
    search.

    Parameters
    ----------
    particleKernelDomain
        The particles and the kernel functions that may cover them.
    dimension
        The spatial dimension.
    journal
        The journal for messages.
    bondParticlesToKernelFunctions
        Whether every kernel function is bonded to the particle of the same position in the domain and moves with it.
    randomlyShiftPartliceShapeFunctions
        With bonded kernel functions and a float: shift each kernel function off its particle centre at random, scaled
        by this factor.
    neighbourListSkinFraction
        The skin, as a fraction of the smallest support half-width; 0 searches in every increment.
    """

    def __init__(
        self,
        particleKernelDomain: ParticleKernelDomain,
        dimension: int,
        journal: Journal,
        bondParticlesToKernelFunctions: bool = False,
        randomlyShiftPartliceShapeFunctions: Union[bool, float] = False,
        neighbourListSkinFraction: float = 0.0,
    ):

        self._meshfreeKernelFunctions = particleKernelDomain.meshfreeKernelFunctions
        self._particles = particleKernelDomain.particles
        self._dimension = dimension
        self.kernelFunctionsAreBondedToParticles = bondParticlesToKernelFunctions
        self._journal = journal

        # A particle lists its kernel functions in the order of their labels, so that the order of its nodes, and so of
        # its degrees of freedom, does not depend on where the search found them. The bins and the record of the last
        # search number the kernel functions in this order; bonding pairs them with the particles in the order of the
        # domain (self._meshfreeKernelFunctions).
        self._kernelFunctionsInLabelOrder = sorted(self._meshfreeKernelFunctions, key=lambda k: k.node.label)

        self._particlesWithChangedKernelFunctions = []

        # The record of the last search, as numbers: the kernel functions of particle i (positions in the label order)
        # are _kernelsOfParticlesAtLastSearch[_firstKernelOfParticleAtLastSearch[i] : ...[i + 1]]. A search compares
        # what it finds against this record to decide whether the kernel functions of a particle changed. The record is
        # what the particles hold because the manager is the only one that assigns their kernel functions
        # (BaseParticle.assignKernelFunctions). Before the first search, no particle has kernel functions, so the first
        # search finds every particle changed.
        self._firstKernelOfParticleAtLastSearch = np.zeros(len(self._particles) + 1, dtype=np.intp)
        self._kernelsOfParticlesAtLastSearch = np.empty(0, dtype=np.intp)

        # If every kernel's support is exactly its bounding box, the box test of the bins is the whole search.
        self._allKernelsHaveBoxSupport = all(k.hasBoxSupport for k in self._meshfreeKernelFunctions)

        self._neighbourListSkin = self._computeNeighbourListSkin(neighbourListSkinFraction)

        # Positions as of the last search, against which motion is measured. Empty means no search
        # has happened yet, so the next call has to be one.
        #
        # The evaluation coordinates of every particle are held end to end in one array rather than one
        # array per particle. Measuring motion is then a single reduction over that array, instead of a
        # few small NumPy calls per particle -- which, at tens of thousands of particles, cost more than
        # the search this criterion exists to avoid.
        self._kernelCentresAtLastSearch = None
        self._evaluationCoordinatesAtLastSearch = None
        self._currentEvaluationCoordinates = None
        self._firstEvaluationCoordinateOfParticle = None

        if not isinstance(randomlyShiftPartliceShapeFunctions, (bool, float)):
            raise ValueError("randomlyShiftPartliceShapeFunctions must be a boolean or a float.")
        self._randomlyShiftPartliceShapeFunctions = randomlyShiftPartliceShapeFunctions

        if self.kernelFunctionsAreBondedToParticles:
            if len(self._particles) != len(self._meshfreeKernelFunctions):
                raise ValueError("The number of particles and kernel functions must be equal.")

            for particle, kernelFunction in zip(self._particles, self._meshfreeKernelFunctions):
                particleCoordinates = particle.getCenterCoordinates()
                kernelFunction.moveTo(particleCoordinates)

        self.signalizeKernelFunctionUpdate()

    @property
    def particlesWithChangedKernelFunctions(self) -> list:
        return self._particlesWithChangedKernelFunctions

    def _computeNeighbourListSkin(self, neighbourListSkinFraction: float) -> float:
        """Turn the requested skin fraction into an absolute distance.

        Parameters
        ----------
        neighbourListSkinFraction
            The skin, as a fraction of the smallest kernel support half-width in the domain. Zero
            disables neighbour list reuse, so that every increment performs a full search.

        Returns
        -------
        float
            The skin as an absolute distance.
        """

        if neighbourListSkinFraction < 0.0:
            raise ValueError("neighbourListSkinFraction must not be negative.")

        if neighbourListSkinFraction == 0.0:
            return 0.0

        if not self._allKernelsHaveBoxSupport:
            raise ValueError(
                "A neighbour list skin requires every kernel function to have box support, because "
                "reuse relies on the search testing an inflated bounding box. Use a skin of zero for "
                "kernel functions whose support is smaller than their bounding box."
            )

        # The smallest half-width in the domain, so that the skin is a conservative fraction of even
        # the tightest support.
        boundingBoxes = np.array([k.getBoundingBox() for k in self._meshfreeKernelFunctions])
        smallestSupportHalfWidth = np.min(0.5 * (boundingBoxes[:, 1, :] - boundingBoxes[:, 0, :]))

        return neighbourListSkinFraction * smallestSupportHalfWidth

    def signalizeKernelFunctionUpdate(self) -> None:
        """Sort the kernel functions into the bins anew (:class:`BinsOfKernelFunctions`), at the boxes they have now.
        Every box is grown by the skin on all sides: a search then reports the kernel functions that cover a particle
        *or come within the skin of doing so*, which is what allows a later increment to reuse the answer."""

        kernelFunctions, dim = self._kernelFunctionsInLabelOrder, self._dimension
        lowerCorners, upperCorners = np.empty((len(kernelFunctions), dim)), np.empty((len(kernelFunctions), dim))

        def boxesOfChunk(kernelIndices) -> None:
            for k in kernelIndices:
                lower, upper = kernelFunctions[k].getBoundingBox()
                lowerCorners[k], upperCorners[k] = lower[:dim], upper[:dim]

        mapChunksInParallel(range(len(kernelFunctions)), boxesOfChunk)

        skin = self._neighbourListSkin
        self._theBins = BinsOfKernelFunctions(lowerCorners - skin, upperCorners + skin)

    def updateConnectivity(self) -> bool:
        """Bring the particle-to-kernel-function connectivity up to date.

        Bonded kernel functions are moved to their particles first, and sorted into the bins anew before a search. A
        particle whose kernel functions changed is assigned the new ones (``assignKernelFunctions``). Every other
        particle keeps its kernel functions and only evaluates its shape functions anew (``updateShapeFunctions``),
        because it has moved.

        With a neighbour list skin the search reports every kernel function that covers a
        particle *or comes within the skin of covering it*, and that answer stays usable until
        accumulated motion could have carried something across the remaining margin. Increments in
        between only update the shape functions, which have to be updated in any case because the
        particles have moved.

        The extra kernel functions this admits are harmless rather than approximate: they evaluate to
        exactly zero at the particle, and the reconstruction discards anything that does, so the shape
        functions are the same ones a full search would have produced.

        Returns
        -------
        bool
            True if any particle's set of kernel functions changed.
        """

        if self.kernelFunctionsAreBondedToParticles:
            self._moveKernelFunctionsToTheirParticles()

        if not self._aSearchIsDue():
            self._updateShapeFunctionsOfAllParticles()
            self._particlesWithChangedKernelFunctions = []
            return False

        if self.kernelFunctionsAreBondedToParticles:
            self.signalizeKernelFunctionUpdate()

        self._searchForCoveringKernelFunctions()
        self._recordPositionsAtThisSearch()

        return len(self._particlesWithChangedKernelFunctions) > 0

    def _moveKernelFunctionsToTheirParticles(self) -> None:
        """Move every kernel function onto the centre of the particle it is bonded to."""

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

    def _allocateMotionTrackingBuffers(self) -> None:
        """Lay out the contiguous buffers the motion criterion works in.

        Particles may report different numbers of evaluation coordinates, so the buffers are indexed
        through a start offset per particle rather than assuming a fixed count.
        """

        numbersOfCoordinates = [
            np.atleast_2d(particle.getEvaluationCoordinates()).shape[0] for particle in self._particles
        ]

        self._firstEvaluationCoordinateOfParticle = np.concatenate(([0], np.cumsum(numbersOfCoordinates)))

        totalNumberOfCoordinates = int(self._firstEvaluationCoordinateOfParticle[-1])

        self._evaluationCoordinatesAtLastSearch = np.zeros((totalNumberOfCoordinates, self._dimension))
        self._currentEvaluationCoordinates = np.zeros((totalNumberOfCoordinates, self._dimension))

    def _gatherEvaluationCoordinates(self, destination: NDArray[np.float64]) -> None:
        """Copy every particle's evaluation coordinates into one contiguous array.

        Parameters
        ----------
        destination
            The array to fill, with one row per evaluation coordinate of the whole domain.
        """

        particles = self._particles
        firstCoordinate = self._firstEvaluationCoordinateOfParticle
        dim = self._dimension

        def gatherChunk(particleIndices) -> None:
            for i in particleIndices:
                coordinates = np.atleast_2d(particles[i].getEvaluationCoordinates())
                destination[firstCoordinate[i] : firstCoordinate[i + 1], :] = coordinates[:, :dim]

        mapChunksInParallel(range(len(particles)), gatherChunk)

    def _aSearchIsDue(self) -> bool:
        """Whether the neighbour lists have to be searched for again.

        A kernel function covers a particle when one of the particle's evaluation coordinates lies
        inside the kernel's box, and both ends of that test move between increments. What eats into
        the skin is the *relative* displacement of a pair, which for any common reference displacement
        d is bounded by

            || dx_i - dc_j || <= || dx_i - d || + || dc_j - d ||

        for every pair, so taking the two maxima separately is conservative whatever d is. Choosing d
        to be the mean displacement is what makes the bound useful here: a bar flying towards a wall
        translates far more than it deforms, and rigid translation cannot change which kernel function
        covers which particle. Measured against the mean, only the departure from rigid motion counts.

        Returns
        -------
        bool
            True if a full search is required.
        """

        if self._neighbourListSkin == 0.0:
            return True

        if self._evaluationCoordinatesAtLastSearch is None:
            return True

        kernelDisplacements = self._currentKernelCentres() - self._kernelCentresAtLastSearch
        rigidTranslation = np.mean(kernelDisplacements, axis=0)

        largestKernelMotion = np.max(np.abs(kernelDisplacements - rigidTranslation))

        self._gatherEvaluationCoordinates(self._currentEvaluationCoordinates)
        largestParticleMotion = np.max(
            np.abs(self._currentEvaluationCoordinates - self._evaluationCoordinatesAtLastSearch - rigidTranslation)
        )

        return largestKernelMotion + largestParticleMotion > self._neighbourListSkin

    def _currentKernelCentres(self) -> NDArray[np.float64]:
        """The current centre of every kernel function, as one array.

        Returns
        -------
        NDArray[np.float64]
            The kernel function centres, one row per kernel function.
        """

        return np.array([kernelFunction.center for kernelFunction in self._meshfreeKernelFunctions])

    def _recordPositionsAtThisSearch(self) -> None:
        """Remember the positions the current neighbour lists were searched at."""

        if self._neighbourListSkin == 0.0:
            return

        if self._firstEvaluationCoordinateOfParticle is None:
            self._allocateMotionTrackingBuffers()

        self._kernelCentresAtLastSearch = self._currentKernelCentres()
        self._gatherEvaluationCoordinates(self._evaluationCoordinatesAtLastSearch)

    def _updateShapeFunctionsOfAllParticles(self) -> None:
        """Update the shape functions of every particle, with the kernel functions it has: the particles have moved,
        so their shape functions are out of date even though their kernel functions are not."""

        def updateChunk(particleChunk: List[Any]) -> None:
            for particle in particleChunk:
                particle.updateShapeFunctions()

        mapChunksInParallel(self._particles, updateChunk)

    def _kernelsOfChunkAtLastSearch(self, first: int, last: int) -> Tuple[NDArray[np.intp], NDArray[np.intp]]:
        """The record of the last search for the particles ``first`` to ``last`` (exclusive), with the offsets of the
        chunk: they start at 0.

        Parameters
        ----------
        first
            The first particle of the chunk.
        last
            One past the last particle of the chunk.

        Returns
        -------
        tuple[NDArray[np.intp], NDArray[np.intp]]
            ``firstKernelAtLastSearch`` and ``kernelsAtLastSearch``: the kernel functions of particle ``first + i`` at
            the last search were ``kernelsAtLastSearch[firstKernelAtLastSearch[i] : firstKernelAtLastSearch[i + 1]]``.
        """

        firstKernelOfParticle = self._firstKernelOfParticleAtLastSearch
        return (
            firstKernelOfParticle[first : last + 1] - firstKernelOfParticle[first],
            self._kernelsOfParticlesAtLastSearch[firstKernelOfParticle[first] : firstKernelOfParticle[last]],
        )

    def _searchForCoveringKernelFunctions(self) -> None:
        """Search, for every particle, the kernel functions that cover one of its evaluation points (within the skin).
        Assign them to a particle whose kernel functions changed since the last search, and update the shape functions
        of every other particle. The particles are searched chunk by chunk, in parallel. See "Searching the kernel
        functions of the particles" on the page "Particle managers"."""

        self._journal.message("Searching particle-kernel connectivity...", "ParticleManager", level=1)

        if not self._particles:
            return

        kernelFunctions = self._kernelFunctionsInLabelOrder
        bins = self._theBins
        dim = self._dimension
        allKernelsHaveBoxSupport = self._allKernelsHaveBoxSupport
        particles = self._particles

        def searchChunk(particleIndices) -> _KernelFunctionsOfAChunk:
            first, last = particleIndices[0], particleIndices[-1] + 1
            particleChunk = particles[first:last]

            evaluationPointsOfParticles = [
                np.atleast_2d(particle.getEvaluationCoordinates()) for particle in particleChunk
            ]
            firstPointOfParticle = np.cumsum(
                [0] + [len(points) for points in evaluationPointsOfParticles], dtype=np.intp
            )
            firstKernelOfParticle, kernelsOfParticles = bins.kernelsWhoseBoxesHoldThePoints(
                np.ascontiguousarray(np.concatenate(evaluationPointsOfParticles)[:, :dim]), firstPointOfParticle
            )
            if not allKernelsHaveBoxSupport:
                firstKernelOfParticle, kernelsOfParticles = _keepKernelsWhoseSupportHoldsAPoint(
                    firstKernelOfParticle, kernelsOfParticles, evaluationPointsOfParticles, kernelFunctions
                )
            firstKernelAtLastSearch, kernelsAtLastSearch = self._kernelsOfChunkAtLastSearch(first, last)

            # Compared as lists of Python ints: slicing and comparing a list costs far less per particle than comparing
            # small NumPy arrays. Each chunk converts only its own part, so no thread takes references to ints that
            # another thread created (in free-threaded Python, every such reference is an atomic operation).
            found, firstFound = kernelsOfParticles.tolist(), firstKernelOfParticle.tolist()
            atLastSearch, firstAtLastSearch = kernelsAtLastSearch.tolist(), firstKernelAtLastSearch.tolist()

            particlesWithChangedKernelFunctions = []
            for i, particle in enumerate(particleChunk):
                kernels = found[firstFound[i] : firstFound[i + 1]]
                if not kernels:
                    raise ValueError(
                        f"Particle at {particle.getCenterCoordinates()} has no associated kernel functions after "
                        "connectivity update."
                    )

                if kernels == atLastSearch[firstAtLastSearch[i] : firstAtLastSearch[i + 1]]:
                    particle.updateShapeFunctions()
                else:
                    particle.assignKernelFunctions([kernelFunctions[k] for k in kernels])
                    particlesWithChangedKernelFunctions.append(particle)

            return _KernelFunctionsOfAChunk(
                firstKernelOfParticle, kernelsOfParticles, particlesWithChangedKernelFunctions
            )

        resultsOfChunks = mapChunksInParallel(range(len(particles)), searchChunk)

        # The chunks are consecutive and their results come in their order, so what is collected is the same for any
        # number of threads.
        self._particlesWithChangedKernelFunctions = [
            particle for chunk in resultsOfChunks for particle in chunk.particlesWithChangedKernelFunctions
        ]
        self._kernelsOfParticlesAtLastSearch = np.concatenate([chunk.kernelsOfParticles for chunk in resultsOfChunks])
        # the offsets of a chunk start at 0; the offsets of all particles are the cumulative numbers of kernel functions
        numbersOfKernels = np.concatenate([np.diff(chunk.firstKernelOfParticle) for chunk in resultsOfChunks])
        self._firstKernelOfParticleAtLastSearch = np.concatenate(([0], np.cumsum(numbersOfKernels))).astype(np.intp)

    def getCoveredDomain(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._theBins.gridLowerCorner, self._theBins.gridUpperCorner

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

        nBins = self._theBins.numberOfBins
        nKernelFunctions = np.zeros(nBins)

        for i in range(nBins[0]):
            for j in range(nBins[1]):
                flatBinIndex = j * self._theBins.strides[1] + i
                nKernelFunctions[i, j] = len(self._theBins.kernelsInBin(flatBinIndex))

        plt.figure()
        plt.imshow(nKernelFunctions.T, origin="lower")
        plt.title("Number of kernel functions in the bins")

        for i in range(nBins[0] + 1):
            plt.plot([i - 0.5, i - 0.5], [0 - 0.5, nBins[1] - 0.5], "k")
        for j in range(nBins[1] + 1):
            plt.plot([0 - 0.5, nBins[0] - 0.5], [j - 0.5, j - 0.5], "k")

        plt.colorbar()
        plt.show()

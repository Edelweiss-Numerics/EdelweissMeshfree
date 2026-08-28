#  ---------------------------------------------------------------------
#
#  _____    _      _              _         _____ _____
# | ____|__| | ___| |_      _____(_)___ ___|  ___| ____|
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |_  |  _|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \  _| | |___
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|   |_____|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2017 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissFE.
#  ---------------------------------------------------------------------

"""Where entities write their stiffness contribution, and how it becomes a CSR matrix.

Two implementations of one interface, so the solvers do not have to know which is in use:

:class:`VIJContribution`
    Stage into a ``sizeVIJ``-long value array, then gather duplicates into CSR. The long-standing
    behaviour.
:class:`DirectCSRContribution`
    Scatter each entity's block straight into CSR, with no value array at all. That array and its
    gather map are the dominant memory cost of a meshfree assembly -- 19.53 GiB of the 20.33 GiB the
    VIJ path spends at 43,350 DOF -- and the DOF ceiling is set by memory, not by assembly time.

The interface is deliberately the one the contributors already used: ``K[entity]`` hands back the
array an entity writes its block into. What differs is what happens afterwards. The VIJ array *is*
the accumulator, so nothing has to happen; the direct path has to push the block through the offset
map, which :meth:`DirectCSRContribution.flush` does.
"""

import threading
from abc import ABC, abstractmethod

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np


class SystemMatrixContributionBase(ABC):
    """Interface between an entity that has computed a stiffness block and the global matrix.

    Attributes
    ----------
    vijArray
        The staging value array, for the code that genuinely needs it -- the threaded Marmot kernels
        take it directly, and the assembly diagnostics compare against it. ``None`` on any path that
        has no such array, which is how a caller tells the two apart without a type test.
    """

    vijArray = None

    @abstractmethod
    def beginAssembly(self):
        """Start a fresh assembly, discarding everything accumulated so far.

        Also drops any block handed out but not yet committed: if an assembly was abandoned (a
        quasi-Newton iteration takes the previous matrix and never asks for this one), its last
        block must not leak into the next one.
        """

    @abstractmethod
    def __getitem__(self, entity):
        """Return the array ``entity`` is to write its stiffness contribution into.

        Shaped by the entity itself, via ``shapeVIJContribution``, so a contributor cannot tell which
        implementation it is writing into.
        """

    @abstractmethod
    def flush(self):
        """Commit anything handed out by :meth:`__getitem__` but not yet in the matrix."""

    @abstractmethod
    def toCSR(self, useInPlace: bool = False):
        """Return the assembled matrix in CSR format.

        Parameters
        ----------
        useInPlace
            Return the internal, reused CSR matrix rather than an independent copy. Only safe if the
            caller does not retain it across the next assembly.
        """


class VIJContribution(SystemMatrixContributionBase):
    """Stage contributions in a VIJ (COO) value array, then gather duplicates into CSR.

    Parameters
    ----------
    vijArray
        The ``VIJSystemMatrix`` to stage into.
    csrGenerator
        The generator holding the sparsity pattern and the gather map.
    """

    def __init__(self, vijArray, csrGenerator):
        self.vijArray = vijArray
        self._csrGenerator = csrGenerator

    def beginAssembly(self):
        self.vijArray[:] = 0.0

    def __getitem__(self, entity):
        return self.vijArray[entity]

    def flush(self):
        # The array is the accumulator: an entity's writes are already in it.
        pass

    @performancetiming.timeit("conversion VIJ to CSR")
    def toCSR(self, useInPlace: bool = False):
        if useInPlace:
            # In-place update: the returned matrix is the generator's internal CSR matrix. The
            # subsequent Dirichlet application only modifies values (the pattern is preserved),
            # which are fully overwritten again on the next update.
            return self._csrGenerator.updateInPlace(self.vijArray)
        return self._csrGenerator.updateCSR(self.vijArray)


class DirectCSRContribution(SystemMatrixContributionBase):
    """Scatter each entity's block straight into CSR, with no staging array.

    One scratch block is reused for every entity, so this holds kilobytes where the VIJ path holds
    gigabytes. It is handed out by :meth:`__getitem__` and committed by :meth:`flush`, and because
    handing out a *new* block flushes the previous one, a caller cannot forget to commit -- only the
    last block of an assembly needs :meth:`flush`, and :meth:`toCSR` does that itself.

    **Zeroed, not accumulated.** Where the VIJ array let two contributors for the same entity
    accumulate in the same slab, each gets a freshly zeroed block here and each block is scattered
    separately. The sum in CSR is the same, because no contributor reads what another wrote -- every
    one of them either adds to or overwrites a block it was given. Anything that starts *reading* its
    block would need this revisited.

    **Single-threaded.** Everything routed through here (cells, elements, constraints, loads) is
    evaluated sequentially and is under 2% of the runtime. The threaded particle kernels do not come
    through here at all: they call ``scatterBlock`` from inside their own ``prange``, with a real
    thread id and a per-thread scratch block. A threaded caller reaching this class would need a
    thread-to-buffer mapping to stay race-free, so :meth:`__getitem__` refuses rather than guessing.

    Parameters
    ----------
    assembler
        The ``DirectCSRAssembler`` to scatter into. Every entity that will be handed out must already
        be registered with it.
    entityIds
        Maps each entity to its index in the assembler's registration order.
    """

    def __init__(self, assembler, entityIds: dict):
        self._assembler = assembler
        self._entityIds = entityIds
        self._scratch = np.zeros(0)
        self._pendingEntity = None
        self._pendingBlock = None
        self._owningThread = None

    def beginAssembly(self):
        # anything not yet committed belongs to an assembly that was abandoned -- drop it
        self._pendingEntity = None
        self._pendingBlock = None
        self._assembler.beginAssembly()

    def __getitem__(self, entity):
        thread = threading.get_ident()
        if self._owningThread is None:
            self._owningThread = thread
        elif thread != self._owningThread:
            raise RuntimeError(
                "DirectCSRContribution was reached from more than one thread. It holds a single "
                "scratch block and scatters with thread id 0, so concurrent callers would both "
                "corrupt the block and race on one private CSR copy. Threaded assembly has to call "
                "scatterBlock directly with its own thread id and its own block, as the Marmot "
                "particle and cell kernels do."
            )

        self.flush()

        size = entity.getVIJContributionSize()
        if size != entity.nDof**2:
            raise NotImplementedError(
                "DirectCSRContribution cannot scatter {:} yet: it contributes {:} VIJ entries "
                "against nDof**2 = {:}, i.e. a sparse custom pattern. The assembler's scatter "
                "addresses a dense nDof x nDof column-major block, so a per-entry variant is needed "
                "for entities like this.".format(type(entity).__name__, size, entity.nDof**2)
            )

        if size > self._scratch.shape[0]:
            self._scratch = np.zeros(size)
        block = self._scratch[:size]
        block[:] = 0.0

        self._pendingEntity = entity
        self._pendingBlock = block

        return entity.shapeVIJContribution(block)

    def flush(self):
        if self._pendingEntity is None:
            return
        entity = self._pendingEntity
        block = self._pendingBlock
        self._pendingEntity = None
        self._pendingBlock = None
        self._assembler.scatterBlock(0, self._entityIds[entity], block)

    @performancetiming.timeit("reduction to CSR")
    def toCSR(self, useInPlace: bool = False):
        self.flush()
        KCsr = self._assembler.reduce()
        return KCsr if useInPlace else KCsr.copy()


class VIJPattern:
    """The VIJ index pattern, without the value array that normally carries it.

    ``CSRGenerator`` and ``DirectCSRAssembler`` need only ``I``, ``J`` and ``nDof`` from a system
    matrix. The direct path has no value array to take them from -- allocating one would defeat the
    point, at 12.84 GiB for 43,350 DOF -- so it passes this instead. The index arrays are the
    DofManager's own; they are referenced, not copied.

    Parameters
    ----------
    theDofManager
        The DofManager holding the pattern.
    """

    def __init__(self, theDofManager):
        self.I = theDofManager.I  # noqa: E741
        self.J = theDofManager.J
        self.nDof = theDofManager.nDof

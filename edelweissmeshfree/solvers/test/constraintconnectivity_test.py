# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
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
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------
"""The connectivity update of the constraints in the solvers of EdelweissMeshfree answers only once every constraint
is up to date.

In a particle simulation there is one constraint per particle, and the update spreads the constraints in chunks over
the threads. ``Executor.map`` hands out the answers of the chunks lazily, so reducing them directly with ``any`` would
stop at the first chunk whose constraints changed, and return while the later chunks are still rewriting the node lists
that the solver builds its DOF manager from next. Here a stand-in of the thread pool runs a chunk only when its answer
is asked for: a chunk that the update does not wait for is never run, so the test does not depend on thread timing.
"""

from edelweissfe.journal.journal import Journal

from edelweissmeshfree.solvers.base import nonlinearsolverbase
from edelweissmeshfree.solvers.base.nonlinearsolverbase import BaseNonlinearSolver

NUMBER_OF_THREADS = 4
#: Enough constraints for the update to spread them over the threads.
NUMBER_OF_CONSTRAINTS = 1000


class _ConstraintStandIn:
    """What the connectivity update asks of a constraint: to bring its connectivity up to date, and whether it
    changed."""

    def __init__(self, changes: bool):
        self.changes = changes
        self.isUpToDate = False

    def updateConnectivity(self, model) -> bool:
        self.isUpToDate = True
        return self.changes


class _LazyThreadPool:
    """A thread pool whose ``map`` hands out the answers lazily, as ``Executor.map`` does, and runs a chunk only when its
    answer is asked for: a chunk the caller does not wait for stays undone, as it would still be running in a real
    pool."""

    def map(self, function, iterable):
        return (function(item) for item in iterable)


def test_the_update_answers_once_every_constraint_is_up_to_date(monkeypatch):
    monkeypatch.setattr(nonlinearsolverbase, "isFreeThreadingSupported", lambda: True)
    monkeypatch.setattr(nonlinearsolverbase, "getNumberOfThreads", lambda: NUMBER_OF_THREADS)
    monkeypatch.setattr(nonlinearsolverbase, "getThreadPool", lambda numberOfThreads: _LazyThreadPool())

    # Only the first constraint changes, so a reduction that stops at the first change stops at the first chunk.
    constraints = [_ConstraintStandIn(changes=i == 0) for i in range(NUMBER_OF_CONSTRAINTS)]

    solver = BaseNonlinearSolver(Journal(verbose=False))
    connectivityHasChanged = solver._updateConstraintConnectivity(constraints, None)

    assert connectivityHasChanged
    assert all(constraint.isUpToDate for constraint in constraints)

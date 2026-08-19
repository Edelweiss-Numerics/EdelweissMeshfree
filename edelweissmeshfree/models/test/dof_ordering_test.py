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
"""Guards the reproducibility of the global DOF ordering.

``Node`` defines no ``__hash__``, so it hashes by identity and iterating a plain ``set`` of Node
objects yields an order that tracks memory addresses -- different on every run. ``DofManager``
assigns DOF indices strictly in ``nodeField.nodes`` order, so if the active-node collection is not
ordered deterministically, the global sparsity pattern changes from run to run. That is not merely
cosmetic: a permuted ordering changes direct-solver fill-in and AMG aggregation, so it injects
run-to-run variance into solve times and Krylov iteration counts.

The invariant asserted here -- active nodes come out sorted by label -- is what makes the numbering
reproducible across processes. Asserting the invariant rather than comparing two in-process
constructions matters: within one process a ``set`` of the same objects iterates consistently, so a
"build it twice and compare" test would pass even on the buggy code.
"""

import numpy as np
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet

from edelweissmeshfree.fields.nodefield import MPMNodeField
from edelweissmeshfree.models.mpmmodel import MPMModel

_FIELD = "displacement"
_N_NODES = 64


class _KernelFunctionStub:
    """The only thing the active-domain assembly asks of a kernel function is its node."""

    def __init__(self, node):
        self.node = node


class _ParticleStub:
    """The only thing the active-domain assembly asks of a particle is its kernel functions."""

    def __init__(self, nodes):
        self.kernelFunctions = [_KernelFunctionStub(n) for n in nodes]


def _buildModelWithShuffledNodes() -> tuple[MPMModel, list[Node]]:
    """Build a model whose nodes reach the assembly in deliberately non-ascending order."""

    nodes = [Node(label, np.zeros(3)) for label in range(1, _N_NODES + 1)]
    for node in nodes:
        node.fields[_FIELD] = None

    # A fixed permutation, so the test does not depend on the identity hashing it is guarding
    # against: if the assembly ever just preserved insertion order, the result would not be sorted.
    shuffled = list(nodes)
    np.random.default_rng(20260819).shuffle(shuffled)
    assert [n.label for n in shuffled] != sorted(n.label for n in shuffled)

    model = MPMModel(3)
    model.particles = {i: _ParticleStub(shuffled[i::4]) for i in range(4)}
    model.nodeFields = {_FIELD: MPMNodeField(_FIELD, 3, NodeSet("all", shuffled))}
    model.nodeSets = {"all": NodeSet("all", shuffled)}

    return model, nodes


def test_active_domain_nodes_are_ordered_by_label():
    """The active-node ordering must be a deterministic function of the labels, not of memory."""

    model, _ = _buildModelWithShuffledNodes()

    _, _, reducedNodeFields, reducedNodeSets = model.assembleActiveDomain({})

    labels = [n.label for n in reducedNodeFields[_FIELD].nodes]
    assert labels == sorted(labels), "active NodeField ordering is not label-sorted"
    assert labels == list(range(1, _N_NODES + 1)), "active NodeField lost or reordered nodes"

    for nodeSet in reducedNodeSets.values():
        setLabels = [n.label for n in nodeSet]
        assert setLabels == sorted(setLabels), "reduced NodeSet ordering is not label-sorted"


def test_solver_active_domain_assembly_does_not_diverge_from_the_model():
    """The solver must delegate, not carry its own copy of the assembly.

    These were once two near-identical implementations, and only the solver's was on the live path --
    so a fix applied to the model's copy silently did nothing. Keep them from drifting apart again.
    """

    from edelweissmeshfree.solvers.base.nonlinearsolverbase import BaseNonlinearSolver

    model, _ = _buildModelWithShuffledNodes()

    fromModel = model.assembleActiveDomain({})
    # unbound on purpose: the method must not depend on solver state
    fromSolver = BaseNonlinearSolver._assembleActiveDomain(None, {}, model)

    assert [n.label for n in fromSolver[2][_FIELD].nodes] == [n.label for n in fromModel[2][_FIELD].nodes]
    assert [n.label for n in fromSolver[0]] == [n.label for n in fromModel[0]]
    assert [n.label for n in fromSolver[1]] == [n.label for n in fromModel[1]]

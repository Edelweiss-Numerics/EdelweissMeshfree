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
"""``MeshfreeExplicitDynamic`` evaluates the constraints at the configuration it has just accepted: the particles have
moved, and the nodes of the kernel functions carry neither a solution nor an increment. It refuses a constraint that
would act as if nothing had moved there (``MPMConstraintBase.actsInExplicitAnalysis``): one on the increment, such as a
weak Dirichlet condition or an equal value, or one of EdelweissFE, whose nodes carry their solution. The penalty
contacts act, explicit or not. A constraint that is switched off takes no part in the step, and is not refused.

The particles are stand-ins, so that no Marmot is needed.
"""

import numpy as np
import pytest
from edelweissfe.constraints.directionalspringpenalty import (
    Constraint as DirectionalSpringPenalty,
)
from edelweissfe.constraints.directionalspringpenalty import (
    DirectionalSpringPenaltySchema,
)
from edelweissfe.journal.journal import Journal
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet

from edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
    ParticlePenaltyCartesianBoundaryConstraintExplicit,
)
from edelweissmeshfree.constraints.explicit.particlepenaltyrigidbodycontactexplicit import (
    ParticlePenaltyContactImplicitSurfaceConstraintExplicit,
)
from edelweissmeshfree.constraints.particlepenaltycartesianboundary import (
    ParticlePenaltyContactCartesianBoundaryConstraint,
)
from edelweissmeshfree.constraints.particlepenaltyequalvalue import (
    ParticlePenaltyEqualValue,
)
from edelweissmeshfree.constraints.particlepenaltyrigidbodycontact import (
    ParticlePenaltyContactImplicitSurfaceConstraint,
)
from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.constraints.test._residualtangent import ParticleStandIn
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.solvers.meshfreeexplicitdynamic import MeshfreeExplicitDynamic

NODES = [Node(label, np.array(x)) for label, x in enumerate([(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])]
SQUARE = np.array([[0.7, 0.6], [0.9, 0.6], [0.9, 0.8], [0.7, 0.8]])
BOUNDARY = 0.75


def _particle() -> ParticleStandIn:
    return ParticleStandIn(NODES, SQUARE.mean(axis=0), SQUARE)


def _signedDistanceBelowTheBoundary(x):
    return BOUNDARY - x[1]


def _gradientOfTheSignedDistance(x):
    return np.array([0.0, -1.0])


#: name -> a function building, from the model, a constraint that acts in an explicit analysis
ACTING = {
    "ParticlePenaltyCartesianBoundaryConstraintExplicit": lambda model: (
        ParticlePenaltyCartesianBoundaryConstraintExplicit("constraint", _particle(), 1, BOUNDARY, model)
    ),
    "ParticlePenaltyContactImplicitSurfaceConstraintExplicit": lambda model: (
        ParticlePenaltyContactImplicitSurfaceConstraintExplicit(
            "constraint", _particle(), _signedDistanceBelowTheBoundary, _gradientOfTheSignedDistance, model
        )
    ),
    "ParticlePenaltyContactCartesianBoundaryConstraint": lambda model: (
        ParticlePenaltyContactCartesianBoundaryConstraint("constraint", _particle(), 1, BOUNDARY, model)
    ),
    "ParticlePenaltyContactImplicitSurfaceConstraint": lambda model: (
        ParticlePenaltyContactImplicitSurfaceConstraint(
            "constraint", _particle(), _signedDistanceBelowTheBoundary, _gradientOfTheSignedDistance, model
        )
    ),
}

#: name -> a function building, from the model, a constraint that does not act in an explicit analysis
NOT_ACTING = {
    "ParticlePenaltyWeakDirichlet": lambda model: (
        ParticlePenaltyWeakDirichlet("constraint", model, [_particle()], "displacement", {1: 0.1}, 1e3)
    ),
    "ParticlePenaltyEqualValue": lambda model: (
        ParticlePenaltyEqualValue("constraint", model, [_particle(), _particle()], "displacement", [1], 1e3)
    ),
    "an EdelweissFE constraint": lambda model: (
        DirectionalSpringPenalty(
            "constraint",
            model,
            NodeSet("spring", NODES[:2]),
            configuration=DirectionalSpringPenaltySchema(field="displacement", component=1, penalty=1e3),
        )
    ),
}


def _modelWith(build) -> MPMModel:
    model = MPMModel(2)
    constraint = build(model)
    model.constraints["constraint"] = constraint
    return model


def _validate(model: MPMModel):
    """Run the solver's check, failing on a refusal: the test configuration skips a test that raises
    ``NotImplementedError``, so a refusal must not escape it."""

    try:
        MeshfreeExplicitDynamic({}, Journal(verbose=False)).validateModelCapabilities(model)
    except NotImplementedError as e:
        pytest.fail(f"refused: {e}")


@pytest.mark.parametrize("name", ACTING)
def test_a_constraint_that_acts_in_an_explicit_analysis_is_accepted(name):
    _validate(_modelWith(ACTING[name]))


@pytest.mark.parametrize("name", NOT_ACTING)
def test_a_constraint_that_would_act_as_if_nothing_had_moved_is_refused(name):
    model = _modelWith(NOT_ACTING[name])

    with pytest.raises(NotImplementedError, match="does not act in an explicit analysis"):
        MeshfreeExplicitDynamic({}, Journal(verbose=False)).validateModelCapabilities(model)


@pytest.mark.parametrize("name", NOT_ACTING)
def test_a_constraint_that_is_switched_off_is_not_refused(name):
    model = _modelWith(NOT_ACTING[name])
    model.constraints["constraint"].active = False

    _validate(model)

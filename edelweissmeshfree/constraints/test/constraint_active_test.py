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
"""A constraint takes part in the equation system while it is ``active`` (EdelweissFE's
``ConstraintBase.active``). The contact constraints of EdelweissMeshfree switch themselves on and off by the reach of
their particle (``ConstraintBase.withinReach``), and they can be switched off from outside like any constraint: then
they stay off, whatever they find in reach, until they are switched on again.

The particle is a stand-in, so that no Marmot is needed. It starts 0.1 below a boundary at y = 0.75, within the reach
of its contact, twice its characteristic length, 0.2; moved down to y = 0.3, it is out of reach.
"""

import numpy as np
import pytest
from edelweissfe.points.node import Node

from edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
    ParticlePenaltyCartesianBoundaryConstraintExplicit,
)
from edelweissmeshfree.constraints.explicit.particlepenaltyrigidbodycontactexplicit import (
    ParticlePenaltyContactImplicitSurfaceConstraintExplicit,
)
from edelweissmeshfree.constraints.particlepenaltycartesianboundary import (
    ParticlePenaltyContactCartesianBoundaryConstraint,
)
from edelweissmeshfree.constraints.particlepenaltyrigidbodycontact import (
    ParticlePenaltyContactImplicitSurfaceConstraint,
)
from edelweissmeshfree.constraints.test._residualtangent import ParticleStandIn
from edelweissmeshfree.models.mpmmodel import MPMModel

BOUNDARY = 0.75
IN_REACH, OUT_OF_REACH = (0.8, 0.65), (0.8, 0.3)


def _signedDistanceBelowTheBoundary(x):
    return BOUNDARY - x[1]


def _gradientOfTheSignedDistance(x):
    return np.array([0.0, -1.0])


#: name -> a function building the contact constraint from its particle and the model
CONTACTS = {
    "ParticlePenaltyContactCartesianBoundaryConstraint": lambda particle, model: (
        ParticlePenaltyContactCartesianBoundaryConstraint("contact", particle, 1, BOUNDARY, model)
    ),
    "ParticlePenaltyContactImplicitSurfaceConstraint": lambda particle, model: (
        ParticlePenaltyContactImplicitSurfaceConstraint(
            "contact", particle, _signedDistanceBelowTheBoundary, _gradientOfTheSignedDistance, model
        )
    ),
    "ParticlePenaltyCartesianBoundaryConstraintExplicit": lambda particle, model: (
        ParticlePenaltyCartesianBoundaryConstraintExplicit("contact", particle, 1, BOUNDARY, model)
    ),
    "ParticlePenaltyContactImplicitSurfaceConstraintExplicit": lambda particle, model: (
        ParticlePenaltyContactImplicitSurfaceConstraintExplicit(
            "contact", particle, _signedDistanceBelowTheBoundary, _gradientOfTheSignedDistance, model
        )
    ),
}


def _moveTo(particle: ParticleStandIn, center):
    particle.center[:] = center
    particle.vertices[:] = np.array(center) + 0.05 * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def _contactInReach(name: str) -> tuple:
    """The model, holding the contact constraint ``name`` of a particle within reach of the boundary, connected."""

    nodes = [Node(label, np.array(x)) for label, x in enumerate([(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)])]
    particle = ParticleStandIn(nodes, IN_REACH, np.zeros((4, 2)), volume=0.01)
    _moveTo(particle, IN_REACH)

    model = MPMModel(2)
    constraint = CONTACTS[name](particle, model)
    model.constraints[constraint.name] = constraint
    constraint.updateConnectivity(model)
    return model, constraint, particle


@pytest.mark.parametrize("name", CONTACTS)
def test_a_contact_switches_itself_off_and_on_by_the_reach_of_its_particle(name):
    model, constraint, particle = _contactInReach(name)
    assert constraint.active and model.activeConstraints() == [constraint]

    _moveTo(particle, OUT_OF_REACH)
    assert constraint.updateConnectivity(model), "switching itself off is a change of the connectivity"
    assert not constraint.withinReach and not constraint.active and model.activeConstraints() == []

    _moveTo(particle, IN_REACH)
    assert constraint.updateConnectivity(model), "switching itself on is a change of the connectivity"
    assert constraint.active


@pytest.mark.parametrize("name", CONTACTS)
def test_a_contact_switched_off_from_outside_stays_off_until_it_is_switched_on_again(name):
    model, constraint, particle = _contactInReach(name)

    constraint.active = False
    constraint.updateConnectivity(model)
    assert constraint.withinReach and not constraint.active, "within reach, but switched off"
    assert model.activeConstraints() == []

    constraint.active = True
    constraint.updateConnectivity(model)
    assert constraint.active and model.activeConstraints() == [constraint]

    _moveTo(particle, OUT_OF_REACH)
    constraint.updateConnectivity(model)
    assert not constraint.active, "switched on again, it follows the reach of its particle"


@pytest.mark.parametrize("name", CONTACTS)
def test_the_model_does_not_locate_a_contact_switched_off_from_outside(name):
    model, constraint, particle = _contactInReach(name)

    constraint.active = False
    _moveTo(particle, OUT_OF_REACH)
    assert not model.updateConnectivity()
    assert constraint.withinReach, "switched off from outside, the contact rests: the model does not locate it"

    constraint.active = True
    assert model.updateConnectivity(), "switched on again, it is located, and finds itself out of reach"
    assert not constraint.withinReach and not constraint.active

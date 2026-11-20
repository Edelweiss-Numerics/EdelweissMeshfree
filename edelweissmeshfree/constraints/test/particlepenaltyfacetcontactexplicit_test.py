"""Tests of the penalty contact between the vertices of particles and a faceted master surface.

They cover what fails silently rather than loudly: which side of a facet counts as inside, the exchange of momentum
between the two sides, the pairing that keeps a deep point from being pushed out of the wrong side, the points that are
found inside without having been seen outside, and the restart of the pairing. The particles and their shape functions
are stand-ins, so no Marmot is needed.
"""

import numpy as np
import pytest
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.elements.contactsurfaceelement import (
    Line2ContactFacet,
    Tria3ContactFacet,
)
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.variables.fieldvariable import FieldVariable

from edelweissmeshfree.constraints.explicit.particlepenaltyfacetcontactexplicit import (
    ContactState,
    ParticlePenaltyFacetContactExplicit,
)
from edelweissmeshfree.constraints.test._residualtangent import ParticleStandIn
from edelweissmeshfree.models.mpmmodel import MPMModel

PENALTY = 1e3
SEARCH = 0.2


class JournalStandIn:
    """Collects what the constraint reports."""

    def __init__(self):
        self.messages = []

    def message(self, text, senderIdentification, level=1):
        self.messages.append(text)


def _facets(facetClass, nodeIndicesOfFacets, nodes):
    facets = []
    for number, nodeIndices in enumerate(nodeIndicesOfFacets):
        facet = facetClass(facetClass.__name__, number + 1)
        facet.setNodes([nodes[i] for i in nodeIndices])
        facet.initializeElement()
        facets.append(facet)
    return facets


def _squareSurface(nPerSide: int) -> list:
    """The boundary of the unit square, counter-clockwise, so that the normals point out of it: ``nPerSide`` Line2
    facets per side."""

    t = np.linspace(0.0, 1.0, nPerSide + 1)[:-1]
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    coordinates = [corners[side] + s * (corners[(side + 1) % 4] - corners[side]) for side in range(4) for s in t]
    nodes = [Node(1000 + i, np.array(x)) for i, x in enumerate(coordinates)]
    for node in nodes:
        node.fields["displacement"] = FieldVariable(node, "displacement")
    return _facets(Line2ContactFacet, [(i, (i + 1) % len(nodes)) for i in range(len(nodes))], nodes)


def _particleWithOneVertexAt(point) -> ParticleStandIn:
    """A particle stand-in whose only vertex is the given point, on four kernel nodes around it."""

    point = np.asarray(point, dtype=float)
    offsets = np.array([[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
    nodes = [Node(label, point + offset) for label, offset in enumerate(offsets)]
    return ParticleStandIn(nodes, point, [point])


def _move(particle: ParticleStandIn, point):
    particle.vertices = np.array([point], dtype=float)


def _contact(particles, facets, journal=None, model=None, **kwargs) -> ParticlePenaltyFacetContactExplicit:
    return ParticlePenaltyFacetContactExplicit(
        "contact", model or MPMModel(2), particles, facets, PENALTY, SEARCH, journal or JournalStandIn(), **kwargs
    )


def _evaluate(contact) -> np.ndarray:
    P = np.zeros(contact.nDof)
    contact.computeKernelsExplicit(np.zeros(contact.nDof), np.zeros(contact.nDof), P, None)
    return P


def test_a_point_seen_outside_then_inside_is_pushed_back_along_the_normal_with_the_penalty_force():
    particle = _particleWithOneVertexAt([0.5, 1.1])
    contact = _contact([particle], _squareSurface(4))
    model = MPMModel(2)

    contact.updateConnectivity(model)
    assert contact.stateOfPoints[0] == ContactState.APPROACHING
    # a point that is seen is paired, so that it can be pushed back as soon as it penetrates, but is not pushed yet
    assert contact.withinReach
    assert not _evaluate(contact).any() and contact.numberOfPointsInContact == 0

    _move(particle, [0.55, 0.98])
    contact.updateConnectivity(model)
    assert contact.stateOfPoints[0] == ContactState.IN_CONTACT

    P = _evaluate(contact).reshape(-1, 2)
    assert contact.deepestPenetration == pytest.approx(0.02)
    # the residual contribution is the negative of the force: the particle is pushed up, the facet down
    np.testing.assert_allclose(contact.contactForceOnParticles, [0.0, PENALTY * 0.02], atol=1e-12)
    np.testing.assert_allclose(P[:4].sum(axis=0), -contact.contactForceOnParticles, atol=1e-12)
    np.testing.assert_allclose(P[4:].sum(axis=0), -contact.contactForceOnMaster, atol=1e-12)
    assert contact.penaltyEnergy == pytest.approx(0.5 * PENALTY * 0.02**2)


def test_the_forces_on_the_two_sides_are_equal_and_opposite():
    """The shape functions of the particle and the weights of the closest point on the facet are both partitions of
    unity, so the contact exchanges momentum and creates none, also at a facet seam and with several points."""

    particles = [_particleWithOneVertexAt([x, 1.05]) for x in (0.1, 0.25, 0.4, 0.77)]
    contact = _contact(particles, _squareSurface(4))
    model = MPMModel(2)
    contact.updateConnectivity(model)
    for particle, depth in zip(particles, (0.01, 0.03, 0.02, 0.04)):
        _move(particle, [particle.vertices[0, 0] + 0.01, 1.0 - depth])
    contact.updateConnectivity(model)

    P = _evaluate(contact)

    assert contact.numberOfPointsInContact == 4
    np.testing.assert_allclose(contact.contactForceOnParticles + contact.contactForceOnMaster, 0.0, atol=1e-12)
    np.testing.assert_allclose(P.reshape(-1, 2).sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(contact.contactForceOnParticles, [0.0, PENALTY * 0.10], atol=1e-12)


def test_a_point_deep_inside_is_pushed_back_the_way_it_came_not_out_of_the_nearest_side():
    """The closest facet of a point deep inside the master body is on the side it is nearest to, whose normal points the
    wrong way. A point in contact keeps the facet it came in through, and moves on only to its neighbours."""

    particle = _particleWithOneVertexAt([0.6, 1.05])
    contact = _contact([particle], _squareSurface(8))
    model = MPMModel(2)
    contact.updateConnectivity(model)

    for y in np.linspace(0.97, 0.1, 30):
        _move(particle, [0.6, y])
        contact.updateConnectivity(model)
        assert contact.stateOfPoints[0] == ContactState.IN_CONTACT

    _evaluate(contact)
    assert contact.deepestPenetration == pytest.approx(0.9)
    np.testing.assert_allclose(contact.contactForceOnParticles, [0.0, PENALTY * 0.9], atol=1e-9)


def test_a_point_in_contact_slides_on_to_the_neighbouring_facet():
    particle = _particleWithOneVertexAt([0.1, 1.05])
    contact = _contact([particle], _squareSurface(4))
    model = MPMModel(2)
    contact.updateConnectivity(model)
    _move(particle, [0.1, 0.99])
    contact.updateConnectivity(model)
    firstFacet = contact._pairedFacetOfPoint[0]

    for x in np.linspace(0.15, 0.9, 16):
        _move(particle, [x, 0.99])
        contact.updateConnectivity(model)

    assert contact.stateOfPoints[0] == ContactState.IN_CONTACT
    assert contact._pairedFacetOfPoint[0] != firstFacet
    _evaluate(contact)
    assert contact.deepestPenetration == pytest.approx(0.01)


def test_a_point_that_leaves_is_released_and_one_out_of_reach_takes_the_constraint_out_of_reach():
    particle = _particleWithOneVertexAt([0.5, 1.05])
    contact = _contact([particle], _squareSurface(4))
    model = MPMModel(2)
    contact.updateConnectivity(model)
    _move(particle, [0.5, 0.99])
    contact.updateConnectivity(model)
    _evaluate(contact)

    _move(particle, [0.5, 1.01])
    contact.updateConnectivity(model)
    _evaluate(contact)
    assert contact.stateOfPoints[0] == ContactState.APPROACHING
    np.testing.assert_array_equal(contact.contactForceOnParticles, 0.0)
    assert contact.penaltyEnergy == 0.0

    _move(particle, [0.5, 1.5])
    assert contact.updateConnectivity(model)
    assert contact.stateOfPoints[0] == ContactState.FREE
    assert not contact.withinReach and contact.nodes == []


def test_a_paired_point_that_penetrates_between_two_updates_is_pushed_back_from_where_it_is_now():
    """The gap is evaluated at the current configuration, so that the contact holds with an analysis that updates the
    connectivity less often than it evaluates the forces."""

    particle = _particleWithOneVertexAt([0.5, 1.05])
    contact = _contact([particle], _squareSurface(4))
    contact.updateConnectivity(MPMModel(2))

    _move(particle, [0.5, 0.96])
    _evaluate(contact)

    np.testing.assert_allclose(contact.contactForceOnParticles, [0.0, PENALTY * 0.04], atol=1e-12)


@pytest.mark.parametrize(
    "path",
    [
        pytest.param([[0.5, 0.9]], id="overlapping from the start"),
        pytest.param([[0.5, 1.5], [0.5, 0.9]], id="tunnelling through the search distance in one increment"),
    ],
)
def test_a_point_found_inside_without_having_been_seen_outside_gets_no_force_and_is_reported(path):
    journal = JournalStandIn()
    particle = _particleWithOneVertexAt(path[0])
    contact = _contact([particle], _squareSurface(4), journal)
    model = MPMModel(2)

    for point in path:
        _move(particle, point)
        contact.updateConnectivity(model)

    assert contact.stateOfPoints[0] == ContactState.FOUND_INSIDE
    assert contact.numberOfPointsFoundInside == 1
    assert not contact.withinReach
    assert any("found inside" in message for message in journal.messages)

    # seen outside again, it may come into contact as any other point
    _move(particle, [0.5, 1.1])
    contact.updateConnectivity(model)
    assert contact.stateOfPoints[0] == ContactState.APPROACHING


def test_a_point_entering_deeper_than_the_warning_fraction_of_the_search_distance_is_reported():
    journal = JournalStandIn()
    particle = _particleWithOneVertexAt([0.5, 1.01])
    contact = _contact([particle], _squareSurface(4), journal)
    model = MPMModel(2)
    contact.updateConnectivity(model)

    _move(particle, [0.5, 1.0 - 0.6 * SEARCH])
    contact.updateConnectivity(model)

    assert contact.stateOfPoints[0] == ContactState.IN_CONTACT
    assert any("entered" in message for message in journal.messages)


def test_without_two_way_coupling_the_master_gets_no_reaction_and_no_degrees_of_freedom():
    particle = _particleWithOneVertexAt([0.5, 1.05])
    contact = _contact([particle], _squareSurface(4), twoWayCoupling=False)
    model = MPMModel(2)
    contact.updateConnectivity(model)
    _move(particle, [0.5, 0.97])
    contact.updateConnectivity(model)

    _evaluate(contact)

    assert contact.nodes == [kf.node for kf in particle.kernelFunctions]
    np.testing.assert_allclose(contact.contactForceOnParticles, [0.0, PENALTY * 0.03], atol=1e-12)
    np.testing.assert_array_equal(contact.contactForceOnMaster, 0.0)


def test_the_master_follows_the_displacement_field_of_its_nodes():
    facets = _squareSurface(4)
    particle = _particleWithOneVertexAt([0.5, 1.15])
    model = MPMModel(2)
    contact = _contact([particle], facets, model=model, twoWayCoupling=False)
    contact.updateConnectivity(model)

    masterNodes = list(dict.fromkeys(node for facet in facets for node in facet.nodes))
    displacementField = NodeField("displacement", getFieldSize("displacement", 2), NodeSet("master", masterNodes))
    displacementField["U"][:, 1] = 0.2  # the master body moves up by 0.2
    model.nodeFields = {"displacement": displacementField}

    contact.updateConnectivity(model)
    _evaluate(contact)

    assert contact.stateOfPoints[0] == ContactState.IN_CONTACT
    assert contact.deepestPenetration == pytest.approx(0.05)


def test_with_two_way_coupling_the_facets_are_where_the_solution_places_them():
    particle = _particleWithOneVertexAt([0.5, 1.1])
    contact = _contact([particle], _squareSurface(4))
    contact.updateConnectivity(MPMModel(2))

    # the master body has moved up by 0.15 since the update: the point at 1.1 is now 0.05 inside it
    U_np = np.zeros(contact.nDof)
    U_np.reshape(-1, 2)[len(particle.kernelFunctions) :, 1] = 0.15
    P = np.zeros(contact.nDof)
    contact.computeKernelsExplicit(U_np, np.zeros(contact.nDof), P, None)

    np.testing.assert_allclose(contact.contactForceOnParticles, [0.0, PENALTY * 0.05], atol=1e-12)
    np.testing.assert_allclose(contact.contactForceOnMaster, [0.0, -PENALTY * 0.05], atol=1e-12)


def test_the_weight_transform_of_a_facet_distributes_the_reaction():
    facets = _squareSurface(1)
    top = facets[2]
    transform = np.array([[0.75, 0.25], [0.25, 0.75]])
    top.setWeightTransform(transform)
    particle = _particleWithOneVertexAt([0.25, 1.05])
    contact = _contact([particle], facets)
    model = MPMModel(2)
    contact.updateConnectivity(model)
    _move(particle, [0.25, 0.98])
    contact.updateConnectivity(model)

    P = _evaluate(contact).reshape(-1, 2)

    force = PENALTY * 0.02
    closestPointWeights = np.array([0.25, 0.75])  # (1, 1) -> (0, 1) at x = 0.25
    np.testing.assert_allclose(P[4:, 1], force * transform @ closestPointWeights, atol=1e-12)
    np.testing.assert_allclose(contact.contactForceOnParticles + contact.contactForceOnMaster, 0.0, atol=1e-12)


def test_the_restart_continues_the_pairing():
    particle = _particleWithOneVertexAt([0.6, 1.05])
    contact = _contact([particle], _squareSurface(8))
    model = MPMModel(2)
    contact.updateConnectivity(model)
    for y in (0.95, 0.7, 0.45):
        _move(particle, [0.6, y])
        contact.updateConnectivity(model)
    restartData = contact.getRestartData()

    resumed = _contact([particle], _squareSurface(8))
    resumed.setRestartData(restartData)
    _move(particle, [0.6, 0.2])
    resumed.updateConnectivity(model)
    contact.updateConnectivity(model)

    np.testing.assert_array_equal(resumed.stateOfPoints, contact.stateOfPoints)
    np.testing.assert_array_equal(resumed._pairedFacetOfPoint, contact._pairedFacetOfPoint)
    _evaluate(resumed)
    assert resumed.deepestPenetration == pytest.approx(0.8)


def test_tria3_facets_in_3d():
    nodes = [Node(10 + i, np.array(x)) for i, x in enumerate([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])]
    facets = _facets(Tria3ContactFacet, [(0, 1, 3), (1, 2, 3)], nodes)
    point = np.array([0.3, 0.6, 0.05])
    kernelNodes = [Node(i, point + offset) for i, offset in enumerate(0.1 * np.eye(3))]
    particle = ParticleStandIn(kernelNodes, point, [point])
    contact = ParticlePenaltyFacetContactExplicit(
        "contact", MPMModel(3), [particle], facets, PENALTY, SEARCH, JournalStandIn()
    )
    model = MPMModel(3)
    contact.updateConnectivity(model)
    particle.vertices = np.array([[0.3, 0.6, -0.04]])
    contact.updateConnectivity(model)

    _evaluate(contact)

    np.testing.assert_allclose(contact.contactForceOnParticles, [0.0, 0.0, PENALTY * 0.04], atol=1e-12)
    np.testing.assert_allclose(contact.contactForceOnParticles + contact.contactForceOnMaster, 0.0, atol=1e-12)


@pytest.mark.parametrize(
    "facetClass, nodeIndices, match",
    [
        pytest.param(Tria3ContactFacet, [(0, 1, 2)], "have 2 nodes", id="Tria3 facets in 2D"),
        pytest.param(Line2ContactFacet, [], "no master facets", id="no facets"),
    ],
)
def test_facets_that_do_not_fit_the_model_are_refused(facetClass, nodeIndices, match):
    nodes = [Node(i, np.array([float(i), 0.0, 0.0])) for i in range(3)]
    facets = []
    for number, indices in enumerate(nodeIndices):
        facet = facetClass(facetClass.__name__, number)
        facet.setNodes([nodes[i] for i in indices])
        facets.append(facet)

    with pytest.raises(ValueError, match=match):
        _contact([_particleWithOneVertexAt([0.0, 0.0])], facets)

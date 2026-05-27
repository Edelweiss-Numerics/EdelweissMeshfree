Variationally Consistent Integration (VCI)
===========================================

Variationally Consistent Integration is a technique introduced by
Chen, Hillman and Rüter (2013) to ensure Galerkin exactness of meshfree
methods by employing a Petrov-Galerkin formulation.

The VCI manager computes correction terms for the test function gradients
so that the integration satisfies the divergence theorem exactly up to the
desired polynomial completeness order.

``VariationallyConsistentIntegrationManager`` class
---------------------------------------------------

.. automodule:: edelweissmeshfree.meshfree.vci
   :members:
   :private-members:

``ParticleKernelDomain`` class
------------------------------

.. automodule:: edelweissmeshfree.meshfree.particlekerneldomain
   :members:
   :private-members:

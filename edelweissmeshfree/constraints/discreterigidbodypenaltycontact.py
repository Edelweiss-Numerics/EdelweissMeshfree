# -*- coding: utf-8 -*-
import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase


class DiscreteRigidContactStiffnessView:
    """Provides structured 2-D numpy views of the global flat sparse stiffness slice."""

    def __init__(self, flat_array: np.ndarray, nDim: int, nRot: int, nSlavesPerActive: list[int], nActive: int):
        rprp_dof = nDim + nRot
        self.K_rprp = flat_array[0 : rprp_dof**2].reshape((rprp_dof, rprp_dof))

        self.K_pp = []
        self.K_prp = []
        self.K_rpp = []

        offset = rprp_dof**2
        for idx in range(nActive):
            n_nodes = nSlavesPerActive[idx]
            k_pp_size = (n_nodes * nDim) ** 2
            k_prp_size = (n_nodes * nDim) * rprp_dof

            self.K_pp.append(flat_array[offset : offset + k_pp_size].reshape((n_nodes * nDim, n_nodes * nDim)))

            p_rp_start = offset + k_pp_size
            self.K_prp.append(flat_array[p_rp_start : p_rp_start + k_prp_size].reshape((n_nodes * nDim, rprp_dof)))

            rp_p_start = p_rp_start + k_prp_size
            self.K_rpp.append(flat_array[rp_p_start : rp_p_start + k_prp_size].reshape((rprp_dof, n_nodes * nDim)))

            offset += k_pp_size + 2 * k_prp_size


class DiscreteRigidBodyPenaltyContact(MPMConstraintBase):
    """
    Implicit Penalty contact constraint for discrete rigid bodies.

    Mathematical Formulation
    ------------------------
    This class enforces normal contact penalty constraints between a collection of background particles
    and a discrete rigid body. The rigid body's kinematics are fully described by its Reference Point (RP)
    translation and rotation DOFs.

    1. Kinematics & Gap Function:
       For a penetrating particle 'p' at coordinate x_p and the closest surface point x_s,
       the gap function 'g' is defined as:
         g = -d_0 - n_s · (Δu_p - Δu_rp - Δθ_rp × r_s)
       where:
         - d_0 is the signed projection distance at the start of the increment (negative = penetration).
         - n_s is the outward unit normal vector of the rigid surface.
         - Δu_p is the interpolated displacement increment of the particle: Δu_p = ∑ N_i Δu_i.
         - Δu_rp, Δθ_rp are the displacement and rotation increments of the Reference Point.
         - r_s is the relative position vector of the contact point: r_s = x_s0 - rp_initial.

       Contact is active when g >= 0.

    2. Normal Force & Residual Vector:
       The contact force vector acting on the particle is:
         f_p = f_n · c = -f_n · n_s
       where:
         - f_n = k_n · g is the normal force magnitude (with penalty parameter k_n).
         - c = -n_s is the contact direction vector.

       The conjugate reaction force and moment acting on the Reference Point are:
         f_rp = -f_p = f_n · n_s
         τ_rp = r_s × f_p = -f_n · (r_s × n_s)

       The contribution to the global external load vector PExt for the active DOFs U is:
         P_contact = f_n · ∂g/∂U = f_n · w
       where the gradient vector w is:
         w = [ -N_i · n_s,  n_s,  n_s × r_s ]

    3. Tangent Stiffness Matrix:
       The Jacobian of the contact force vector with respect to the displacement increments gives
       the tangent stiffness matrix:
         K = ∂(f_n · w)/∂U = K_mat + K_geo

       - Material Stiffness (Penalty part):
         K_mat = k_n · (w ⊗ w)

       - Geometric Stiffness (Faceted Surface assumption):
         Since the rigid surface is represented by flat elements, the normal vector n_s does not change
         with respect to particle sliding (∂n_s/∂u_p = 0). It only changes due to rigid body rotation:
           δn_s = δθ_rp × n_s = -n_s_hat · δθ_rp
         where n_s_hat is the skew-symmetric cross-product matrix of n_s.

         Differentiating the forces and moments yields the exact faceted geometric stiffness terms:
           a) Particle force coupling with RP rotation:
              ∂f_p/∂θ_rp = -f_n · n_s_hat
           b) RP force coupling with RP rotation:
              ∂f_rp/∂θ_rp = f_n · n_s_hat
           c) RP torque coupling with RP rotation (torque-rotation geometric stiffness):
              ∂τ_rp/∂θ_rp = f_p_hat · r_s_hat
              where f_p_hat and r_s_hat are the skew-symmetric matrices of f_p and r_s.
    """

    def __init__(
        self,
        name: str,
        particles,
        model,
        rigidBody,
        penaltyParameter: float = 1e5,
        proximityFactor: float = 2.0,
    ):
        self._name = name
        self._particles = list(particles)
        self.rigidBody = rigidBody
        self._penaltyParameter = penaltyParameter
        self.proximityFactor = proximityFactor

        self._domainSize = model.domainSize
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, self._domainSize)

        self._candidates = []
        self._nodes = []
        self._fieldsOnNodes = []
        self.isActive = False
        self.reactionForce = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return self._fieldsOnNodes

    @property
    def nDof(self) -> int:
        nDim = self._domainSize
        nRot = 3
        n_cand_nodes = len(self._nodes) - 1 if self._nodes else 0
        return n_cand_nodes * nDim + (nDim + nRot)

    @property
    def scalarVariables(self) -> list:
        return list()

    @property
    def active(self) -> bool:
        return self.isActive

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list):
        pass

    def updateConnectivity(self, model):
        """Dynamic proximity check at start of step to find candidate particles."""
        coords = np.array([p.getCenterCoordinates() for p in self._particles])
        u_rp, R, rp_initial = self.rigidBody.getCurrentKinematics()
        dists, _ = self.rigidBody.querySurface(coords, proximity_factor=self.proximityFactor)

        active_mask = dists < self.proximityFactor
        new_candidates = [p for i, p in enumerate(self._particles) if active_mask[i]]

        hasChanged = new_candidates != self._candidates
        self._candidates = new_candidates

        candidate_nodes = []
        seen = set()
        for p in self._candidates:
            for kf in p.kernelFunctions:
                if kf.node not in seen:
                    seen.add(kf.node)
                    candidate_nodes.append(kf.node)

        new_nodes = candidate_nodes + [self.rigidBody.rpNode]
        if new_nodes != self._nodes:
            hasChanged = True

        self._nodes = new_nodes
        self._fieldsOnNodes = [["displacement"] for _ in candidate_nodes] + [["displacement", "rotation"]]
        self.isActive = len(self._candidates) > 0
        return hasChanged

    def getVIJContributionSize(self) -> int:
        nDim = self._domainSize
        nRot = 3
        rprp_dof = nDim + nRot
        size = rprp_dof**2
        for p in self._candidates:
            n_nodes = len(p.kernelFunctions)
            size += (n_nodes * nDim) ** 2 + 2 * (n_nodes * nDim) * rprp_dof
        return size

    def _getLocalDofMapping(self):
        nDim = self._domainSize
        node_local_dofs = {}
        curr_local_dof = 0
        for i, node in enumerate(self._nodes):
            fields = self._fieldsOnNodes[i]
            dofs = []
            for field in fields:
                field_size = getFieldSize(field, nDim)
                dofs.extend(range(curr_local_dof, curr_local_dof + field_size))
                curr_local_dof += field_size
            node_local_dofs[node] = dofs
        return node_local_dofs

    def initializeVIJContribution(self, idcs: np.ndarray, I_: np.ndarray, J_: np.ndarray, offset: int) -> None:
        nDim = self._domainSize
        nRot = 3
        rprp_dof = nDim + nRot

        node_local_dofs = self._getLocalDofMapping()
        rp_dofs = node_local_dofs[self.rigidBody.rpNode]

        k = offset

        for ri in range(rprp_dof):
            for rj in range(rprp_dof):
                I_[k] = idcs[rp_dofs[ri]]
                J_[k] = idcs[rp_dofs[rj]]
                k += 1

        for p in self._candidates:
            p_nodes = [kf.node for kf in p.kernelFunctions]
            p_dofs = []
            for node in p_nodes:
                p_dofs.extend(node_local_dofs[node])
            n_p_dofs = len(p_dofs)

            for i in range(n_p_dofs):
                for j in range(n_p_dofs):
                    I_[k] = idcs[p_dofs[i]]
                    J_[k] = idcs[p_dofs[j]]
                    k += 1
            for i in range(n_p_dofs):
                for j in range(rprp_dof):
                    I_[k] = idcs[p_dofs[i]]
                    J_[k] = idcs[rp_dofs[j]]
                    k += 1
            for i in range(rprp_dof):
                for j in range(n_p_dofs):
                    I_[k] = idcs[rp_dofs[i]]
                    J_[k] = idcs[p_dofs[j]]
                    k += 1

    def shapeVIJContribution(self, flat_view: np.ndarray) -> DiscreteRigidContactStiffnessView:
        if isinstance(flat_view, DiscreteRigidContactStiffnessView):
            return flat_view
        nDim = self._domainSize
        nRot = 3
        nSlavesPerActive = [len(p.kernelFunctions) for p in self._candidates]
        return DiscreteRigidContactStiffnessView(
            flat_view,
            nDim=nDim,
            nRot=nRot,
            nSlavesPerActive=nSlavesPerActive,
            nActive=len(self._candidates),
        )

    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, K, timeStep: TimeStep):
        if not self.isActive:
            return

        K = self.shapeVIJContribution(K)

        self.reactionForce = 0.0
        nDim = self._domainSize
        nRot = 3

        node_local_dofs = self._getLocalDofMapping()
        rp_dofs = node_local_dofs[self.rigidBody.rpNode]

        # RP start-of-step configuration
        _, _, rp_initial = self.rigidBody.getCurrentKinematics()

        # RP current iteration trial increments
        delta_u_rp = dU[rp_dofs[0:nDim]]
        delta_theta_rp = dU[rp_dofs[nDim : nDim + nRot]]

        # Vectorized surface query
        coords = np.array([p.getCenterCoordinates() for p in self._candidates])
        dists, normals = self.rigidBody.querySurface(coords, proximity_factor=self.proximityFactor)

        for idx, p in enumerate(self._candidates):
            d0 = dists[idx]
            if d0 >= self.proximityFactor:
                continue

            n_s = normals[idx]

            # Contact point on surface at start of step
            x_s0 = coords[idx] - d0 * n_s
            r_s = x_s0 - rp_initial

            # Interpolate particle displacement increment
            N_vec = p.getInterpolationVector(coords[idx]).flatten()
            p_nodes = [kf.node for kf in p.kernelFunctions]
            p_dofs = []
            for node in p_nodes:
                p_dofs.extend(node_local_dofs[node])

            delta_u_p = np.zeros(nDim)
            for i, node in enumerate(p_nodes):
                local_node_dofs = node_local_dofs[node]
                delta_u_p += N_vec[i] * dU[local_node_dofs]

            # Compute gap function g
            c = -n_s
            delta_u_rel = delta_u_p - delta_u_rp - np.cross(delta_theta_rp, r_s)
            g = -d0 + np.dot(c, delta_u_rel)

            if g < 0:
                continue

            # Assemble residual gradient vector
            dg_dU = np.zeros(self.nDof)
            for i in range(len(p_nodes)):
                start_idx = i * nDim
                dg_dU[p_dofs[start_idx : start_idx + nDim]] = N_vec[i] * c

            dg_dU[rp_dofs[0:nDim]] = -c
            dg_dU[rp_dofs[nDim : nDim + nRot]] = np.cross(r_s, c)

            # Update residual
            f_n_mag = self._penaltyParameter * g
            PExt += f_n_mag * dg_dU
            self.reactionForce += f_n_mag

            # Tangent Stiffness Assembly
            K_mat = self._penaltyParameter * np.outer(dg_dU, dg_dU)
            K_geo = np.zeros((self.nDof, self.nDof))

            # Cross-product skew matrices for geometric terms
            f_p = f_n_mag * c
            n_s_hat = np.array([[0.0, -n_s[2], n_s[1]], [n_s[2], 0.0, -n_s[0]], [-n_s[1], n_s[0], 0.0]])
            f_p_hat = np.array([[0.0, -f_p[2], f_p[1]], [f_p[2], 0.0, -f_p[0]], [-f_p[1], f_p[0], 0.0]])
            r_s_hat = np.array([[0.0, -r_s[2], r_s[1]], [r_s[2], 0.0, -r_s[0]], [-r_s[1], r_s[0], 0.0]])

            # RP Rotation - RP Rotation geometric stiffness
            K_geo[np.ix_(rp_dofs[nDim : nDim + nRot], rp_dofs[nDim : nDim + nRot])] += f_p_hat @ r_s_hat

            # Force-rotation coupling geometric stiffness
            for i in range(len(p_nodes)):
                K_geo[np.ix_(p_dofs[i * nDim : (i + 1) * nDim], rp_dofs[nDim : nDim + nRot])] += (
                    -f_n_mag * N_vec[i] * n_s_hat
                )

            K_geo[np.ix_(rp_dofs[0:nDim], rp_dofs[nDim : nDim + nRot])] += f_n_mag * n_s_hat

            # Write to global sparse views
            K_total = K_mat + K_geo
            K.K_pp[idx] += K_total[np.ix_(p_dofs, p_dofs)]
            K.K_prp[idx] += K_total[np.ix_(p_dofs, rp_dofs)]
            K.K_rpp[idx] += K_total[np.ix_(rp_dofs, p_dofs)]
            K.K_rprp += K_total[np.ix_(rp_dofs, rp_dofs)]

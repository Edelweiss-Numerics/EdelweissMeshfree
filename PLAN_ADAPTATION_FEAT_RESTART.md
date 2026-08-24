# Plan: Adapting EdelweissMeshfree to EdelweissFE's `feat/restart` Stack

> **Context**: `EdelweissFE` is landing its consolidated stack onto `next_v26.11` (covering PR #64 AMR/MPC gate, PR #68 input-system registry overhaul, consolidated BlockAMG linear solvers, topology pipeline, and checkpoint/restart serialization). This document details the architectural impact and phase-by-phase adaptation plan for `EdelweissMeshfree`.

---

## 1. Upstream Architectural Drivers & Coupling Analysis

```
EdelweissFE Stack (next_v26.11):
 ├── PR #64: feat/amr-hanging-nodes & MPC condensation
 ├── PR #68: feat/input-system-registry (InputLanguage removal, schema dataclasses, generator/output APIs)
 ├── perf/linsolve-consolidated (BlockAMG, AMGCL lgmres outer solver, vectorized equilibration)
 ├── feat/topology-pipeline (Model modifiers, lifecycle hooks, entity ordering)
 └── feat/restart (HDF5 format versioning, element/constraint/timestepper checkpoint serialization)
```

### A. Input-System & OutputManager Overhaul (PR #68)
- **Removal of `InputLanguage` and `Module`**: The legacy dynamic keyword system is replaced by typed schema dataclasses registered in `edelweissfe.config.registry`.
- **`EnsightOutputManager`**: Constructor accepts typed `configuration=EnsightSchema(...)`. The legacy `updateDefinition()` method was removed in favor of `createPerNodeOutput()` and `createPerElementOutput()`.
- **Generators & Sections**: Mesh generators and sections transitioned to typed classes inheriting from `GeneratorBase`.

### B. Checkpoint & Restart Serialization (`feat/restart` & `feat/topology-pipeline`)
- **Format Versioning**: `FEModel` enforces `RESTART_FORMAT_VERSION = 2` on HDF5 root attributes.
- **Serialization Contract**:
  - `FEModel.writeRestart(f)` writes `nodeFields`, `scalarVariables`, `elements` (`getStateVars()`), `constraints` (`getRestartData()`), and `topologyHistory`.
  - `FEModel.readRestart(f)` validates format version, replays topology changes, restores `nodeFields`, `scalarVariables`, `elements` (`setStateVars()`), and `constraints` (`setRestartData()`).
- **TimeStepper State**: `AdaptiveTimeStepper.writeRestart()` / `readRestart()` snapshots generator progress and post-yield increment growth state (`incrementCounter`, `nPassedGoodIncrements`, `dT`, `increment`).
- **MPM Specifics**: `MPMModel` must serialize/deserialize `self.particles` and `self.materialPoints` without breaking when FE-specific datasets (`elements`, `topologyHistory`) are empty.

### C. Constraint & Solver Capability Contracts
- **Lifecycle Hooks**: `FEModel.advanceToTime()` calls `acceptLastState()` across `model.constraints` and `model.multiPointConstraints`.
- **Capability Validation**: Solvers enforce `supportsMPC` and capability validation via `validateModelCapabilities(model)`.
- **`MPMConstraintBase`**: Must implement `acceptLastState()`, `getRestartData()`, and `setRestartData()`.

### D. Linear Solver Modernization (`perf/linsolve-consolidated`)
- `BlockAMGSolver` now features native AMGCL `lgmres` outer solver with reset-on-increment support, vectorized diagonal equilibration, and threaded off-diagonal SpMV.
- `MPMDofManager` must conform to `DofVector` permutations and assembly patterns.

---

## 2. Phase-by-Phase Implementation Plan

### Phase 1: Environment & Cython Toolchain Alignment
- [ ] **Marmot Build**: Verify that the Marmot build in `$CONDA_PREFIX` includes all meshfree categories (`materials`, `particles`, `materialpoints`, `cells`, `cellelements`, `core`).
- [ ] **Clean In-Place Compilation**: In `EdelweissMeshfree`, run:
  ```bash
  python setup.py build_ext --inplace --force
  pip install -e .
  ```
- [ ] **Import Validation**: Verify that `marmotparticlewrapper`, `marmotcell`, `marmotcellelement`, and `marmotmeshfreeapproximation` import cleanly without unresolved symbols.

### Phase 2: Base Classes & Model Serialization (`MPMModel` & `MPMConstraintBase`)
- [ ] **`MPMModel.writeRestart(restartFile: h5py.File)`**:
  - Delegate to `super().writeRestart(restartFile)` to write `nodeFields`, `scalarVariables`, and time metadata.
  - Serialize `self.particles` to dataset group `f["particles"]` using `p.getRestartData()`.
  - Serialize `self.materialPoints` to dataset group `f["materialPoints"]`.
- [ ] **`MPMModel.readRestart(restartFile: h5py.File)`**:
  - Delegate to `super().readRestart(restartFile)` to restore field variables and time metadata.
  - Safely handle empty `f["elements"]` when no background FE elements exist.
  - Restore particle state variables via `p.readRestartData(f["particles"][str(p.number)])`.
- [ ] **`MPMConstraintBase` Lifecycle & State**:
  - Ensure `acceptLastState()`, `getRestartData() -> dict | None`, and `setRestartData(data: dict)` are declared in `MPMConstraintBase`.
  - Implement concrete state serialization for penalty and contact constraints (e.g. boundary friction, contact state).

### Phase 3: Input System & Output Manager Modernization
- [ ] **Merge `feat/input-system-registry`**: Bring `feat/input-system-registry` (commit `a8b8846`) up to date with the latest `next_v26.11` trunk.
- [ ] **Output Managers**: Ensure all MPM output managers (`EnsightOutputManager`) conform to `createPerNodeOutput()` / `createPerElementOutput()`.
- [ ] **Generators**: Audit `rectangulargridgenerator`, `rectangularmpgenerator`, `abqinputreader`, and `exodusinputreader` to ensure compatibility with `FEModel` topology changes window (`with model.topologyChanges():`).

### Phase 4: Solver & Linear Solver Integration
- [ ] **Solver Re-Alignment**: Audit [`nqs.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissMeshfree/edelweissmeshfree/solvers/nqs.py), [`nqsmarmotparallel.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissMeshfree/edelweissmeshfree/solvers/nqsmarmotparallel.py), [`nqsmparclength.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissMeshfree/edelweissmeshfree/solvers/nqsmparclength.py), and [`explicitmultiphysicssolver.py`](file:///home/matthias/constitutive_modeling/next_v2611/EdelweissMeshfree/edelweissmeshfree/solvers/explicitmultiphysicssolver.py).
- [ ] **Restart Output Integration**: Wire `RestartOutputManager` from `edelweissfe.outputmanagers.restart` into the solver step loop as an optional output manager, unifying checkpoint writing.
- [ ] **`BlockAMGSolver` Verification**: Ensure `python_linear_elastic_blockamg_test.py` runs with full solution-preserving convergence.

### Phase 5: Test Suite Verification & Gold File Audit
- [ ] **Unit & Integration Suite**: Run `pytest` across all ~64 examples in `examples/`.
- [ ] **Restart Verification**:
  - `examples/114_marmot_micropolar_snni_quad_restart_test/marmot_micropolar_snni_quad_restart_test.py`
  - `examples/115_marmot_micropolar_sqcni_vci_quad_indirect_control_gosford_test/`
  - `examples/127_indirect_control_predictor_test/`
- [ ] **Gold Solutions**: Verify all `gold.csv` outputs match within standard machine tolerances.

---

## 3. Risk Matrix & Mitigations

| Risk | Impact | Mitigation Strategy |
|---|---|---|
| **Restart Format Mismatch** | High (checkpoints unreadable across runs) | `MPMModel` delegates to `super()` while pinning `RESTART_FORMAT_VERSION = 2`. |
| **Cython ABI / Marmot Symbol Drift** | High (import errors at runtime) | Build Marmot with `*_MODULES="all"` and force-recompile Cython extensions in-place. |
| **OutputManager API Incompatibility** | Medium (examples failing on setup) | Already largely addressed in `feat/input-system-registry` (`a8b8846`); verify call sites across all test decks. |
| **DOF Permutation Mismatch in Solvers** | Medium (incorrect residual / stiffness assembly) | Use `DofManager.idcsOfFieldVariablesInDofVector` and `MPMDofManager` permutation patterns. |

# u-p-J feature verification tests

Fast, isolated regression tests for the mixed displacement–pressure–Jacobi (u-p-J)
meshfree features developed on the VMS / Nitsche feature branches. **One tiny test
per feature** (~0.3 s each, < 3 s ceiling), sharing one small base problem
(`_upj_base.py`: a 4×8, nearly-elastic, 2-increment plane-strain compression block —
the exact pressure is near-uniform, so "finite + compressive mean" is the sanity
signal).

These complement the large numerical **studies** (Cook's membrane = example 134,
plane-strain compression shear band = example 146), which exercise everything at once
and take many seconds. Here each feature is checked on its own so a regression is
localized immediately.

| test | feature |
|---|---|
| `upj_vms_mode0_test.py` | pressure-only OSS stabilization (default) |
| `upj_vms_mode1_test.py` | full VMS (resolved-scale momentum residual) |
| `upj_vms_mode2_test.py` | fully stabilized VMS (+ momentum & jacobi terms) |
| `upj_vms_mode3_test.py` | mode 2 + FD second-order material tangent |
| `upj_mortar_bc_test.py` | mortar weak Dirichlet BC |
| `upj_lagrange_bc_test.py` | Lagrange-multiplier weak Dirichlet BC |
| `upj_cwf_test.py` | consistent-weak-form boundary correction |
| `upj_kaware_test.py` | K-aware / bounded stabilization coefficient C |
| `upj_sqcnixsdi_test.py` | SQCNIxSDI u-p-J particle (sub-domain integration) |
| `upj_vci_test.py` | variationally consistent integration (order 1) |
| `upj_nitsche_bc_test.py` | Nitsche Dirichlet BC (**only on the `tom/feat/nitsche-dirichlet` branch**) |

Run all: `python -m pytest examples/150_upj_feature_tests/`
Run one: `python examples/150_upj_feature_tests/upj_cwf_test.py`

**Adding a test for a new feature:** copy one file, flip the relevant `run_sim` flag,
keep the `assert_sane_compression(foc)` check (add a feature-specific assertion if
cheap and robust). Keep it under 3 s.

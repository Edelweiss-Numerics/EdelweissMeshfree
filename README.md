<!-- [![documentation](https://github.com/EdelweissMPM/EdelweissMPM/actions/workflows/sphinx.yml/badge.svg)](https://edelweissfe.github.io/EdelweissMPM) -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# EdelweissMeshfree
## A light-weight, platform-independent, parallel meshfree module for EdelweissFE.

<p align="center">
<img width="400" height="400" src="./doc/source/borehole.gif">
</p>
<p align="center"><em>Implicit RKPM simulation of a borehole breakout using a gradient-enhanced micropolar damage–plasticity constitutive model.</em></p>

<!-- See the [documentation](https://edelweissfe.github.io/EdelweissMPM). -->

**EdelweissMeshfree** provides at an easy-to-understand yet efficient implementation of meshfree numerical methods for solving partial differential equations. The current release includes implementations of the **Material Point Method (MPM)** and the **Reproducing Kernel Particle Method (RKPM)**.

**Key features are:**

 * Python for non performance-critical routines
 * Cython for performance-critical routines
 * Parallelization
 * Modular system, which is easy to extend
 * Output to Paraview, Ensight, CSV, matplotlib
 * Interfaces to powerful direct and iterative linear solvers
 * Integration of the [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/) library, providing cells, material points, particles and constitutive model formulations

**Note:** The current public version of **EdelweissMeshfree** depends on the infrastructure of Marmot cells, material points, and particles; these components are required to run simulations.

## Installation

EdelweissMeshfree hard-depends on [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/)
(all cells, material points, particles, kernel functions, and approximations are
Marmot-backed Cython extensions) and on
[EdelweissFE](https://github.com/EdelweissFE/EdelweissFE), whose solver infrastructure
(DOF management, CSR assembly, linear solvers) it reuses.

Prerequisites, in order:

1. A conda environment with the free-threading Python build — the easiest way is
   EdelweissFE's bootstrap script, which sets up the environment and builds the complete
   stack (Eigen, autodiff, Fastor, AMGCL, Marmot, EdelweissFE):

   ```bash
   cd ../EdelweissFE
   bash scripts/bootstrap_stack.sh
   ```

   Alternatively, follow the manual "installation with Marmot" instructions in
   EdelweissFE's README.

2. Marmot must be installed into the active environment prefix. If it lives elsewhere,
   point the build to it via `MARMOT_INSTALL_DIR`.

Then install EdelweissMeshfree. The build dependencies (Cython, numpy) are already
provided by the conda environment, so disable pip's build isolation to compile against
them:

```bash
pip install --no-build-isolation .
```

Unlike EdelweissFE, all Cython extensions here are mandatory — a build failure indicates
a broken Marmot installation and aborts the install.

## Run tests

```bash
python -m pytest .
```

### Verifying free-threading

All Cython extensions declare themselves free-threading compatible. Verify that the GIL
stays disabled in your installation (a `RuntimeWarning: The global interpreter lock (GIL)
has been enabled to load module ...` on stderr indicates a stale or misconfigured build):

```bash
python -c "import sys; import edelweissmeshfree.solvers.base.parallelization, edelweissmeshfree.materialpoints.marmotmaterialpoint.mp; assert not sys._is_gil_enabled(), 'GIL was re-enabled!'; print('free-threading OK')"
```

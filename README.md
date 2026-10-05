<!-- [![documentation](https://github.com/EdelweissMPM/EdelweissMPM/actions/workflows/sphinx.yml/badge.svg)](https://edelweissfe.github.io/EdelweissMPM) -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/1095513890.svg)](https://doi.org/10.5281/zenodo.21113586)


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

1. EdelweissFE, installed with Marmot into its conda environment `edelweissfe`. Follow
   EdelweissFE's [installation instructions](https://github.com/Edelweiss-Numerics/EdelweissFE#installation),
   including the section *Optional: Marmot*. The environment is pinned by EdelweissFE's
   lock file and provides everything EdelweissMeshfree needs to build (compilers, Cython,
   numpy, Eigen, the free-threaded Python).

2. Marmot must be installed into the active environment prefix. If it lives elsewhere,
   point the build to it via `MARMOT_INSTALL_DIR`.

Then install EdelweissMeshfree. The build dependencies (Cython, numpy) are already
provided by the conda environment, so disable pip's build isolation to compile against
them:

```bash
conda activate edelweissfe
pip install --no-build-isolation -e .
```

The extensions are compiled with the same settings as EdelweissFE's, from
`edelweissfe.utils.extensionbuild`. In particular, they are compiled with `-march=native`
by default; set `EDELWEISSFE_ARCH_FLAGS` (e.g. `-march=x86-64-v3`, or empty for none) when
the build must run on other machines, e.g. in a container image.

Unlike EdelweissFE, all Cython extensions here are mandatory — a build failure indicates
a broken Marmot installation and aborts the install.

## Run tests

Run the test suite to verify the setup:
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

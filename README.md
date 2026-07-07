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

## Installation TL;DR

`EdelweissMeshfree` requires `EdelweissFE` (with `Marmot` support enabled) and the `Marmot` library to be installed first.

### Step 1: Pre-requisites & Dependencies
Ensure your conda/mamba environment is active and all required dependencies (`Eigen`, `autodiff`, `Fastor`, and optionally `amgcl`) are installed as detailed in the [EdelweissFE README](https://github.com/EdelweissFE/EdelweissFE).

### Step 2: Install Marmot
Clone and install the `Marmot` library:
```bash
git clone --branch master --recurse-submodules https://github.com/MAteRialMOdelingToolbox/Marmot/
cd Marmot && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
cd ../..
```

### Step 3: Install EdelweissFE
Clone and install `EdelweissFE` to link with the Marmot library:
```bash
git clone https://github.com/EdelweissFE/EdelweissFE.git
cd EdelweissFE
mamba install --file conda_requirements.txt
pip install -r pip_requirements.txt
pip install -v .
cd ..
```

### Step 4: Install EdelweissMeshfree
Install `EdelweissMeshfree`:
```bash
cd EdelweissMeshfree
python -m pip install .
```

## Run tests

Run the test suite to verify the setup:
```bash
python -m pytest .
```

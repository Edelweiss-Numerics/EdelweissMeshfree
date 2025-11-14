<!-- [![documentation](https://github.com/EdelweissMPM/EdelweissMPM/actions/workflows/sphinx.yml/badge.svg)](https://edelweissfe.github.io/EdelweissMPM) -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# EdelweissMeshfree
## A light-weight, platform-independent, parallel meshfree module for EdelweissFE.

<!-- <p align="center"> -->
<!--   <img width="512" height="512" src="./doc/source/borehole.gif"> -->
<!-- </p> -->

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

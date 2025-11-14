<!-- [![documentation](https://github.com/EdelweissMPM/EdelweissMPM/actions/workflows/sphinx.yml/badge.svg)](https://edelweissfe.github.io/EdelweissMPM) -->
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# EdelweissMeshfree
## A light-weight, platform-independent, parallel meshfree module for EdelweissFE.

<!-- <p align="center"> -->
<!--   <img width="512" height="512" src="./doc/source/borehole_damage_lowdilation.gif"> -->
<!-- </p> -->

<!-- See the [documentation](https://edelweissfe.github.io/EdelweissMPM). -->

EdelweissMeshfree aims at an easy to understand, yet efficient implementation of meshfree methods for solving partial differential equations. Current implementaions offer the **Material Point Method (MPM)** and the **Reproducing Kernel Particle Method (RKPM)**.
Some features are:

 * Python for non performance-critical routines
 * Cython for performance-critical routines
 * Parallelization
 * Modular system, which is easy to extend
 * Output to Paraview, Ensight, CSV, matplotlib
 * Interfaces to powerful direct and iterative linear solvers

EdelweissMeshfree makes use of the [Marmot](https://github.com/MAteRialMOdelingToolbox/Marmot/) library for cells, material points, particles and constitutive model formulations.

Please note that the current public EdelweissMeshfree requires Marmot cells, material points, and particles for being able to run simulations.

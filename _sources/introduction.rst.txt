Introduction
============

EdelweissMeshfree is a light-weight, platform-independent, parallel meshfree framework for
solid-mechanics simulations and coupled multi-physics problems.

The project focuses on providing a flexible and practical development environment for meshfree
methods, with strong support for the Material Point Method (MPM) and the Reproducing Kernel
Particle Method (RKPM). It is designed to bridge research and application needs by combining:

- Accessibility for rapid prototyping in Python,
- Performance-critical components implemented in Cython,
- Extensible modular architecture for adding new formulations and workflows,
- Integration with EdelweissFE and Marmot-based components.

In short, the scope of EdelweissMeshfree is to enable efficient nonlinear simulations up to
medium-sized problems while keeping method development and customization straightforward.

# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _          
# | ____|__| | ___| |_      _____(_)___ ___  
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| 
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ 
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/ 
# |  \/  | ___  ___| |__  / _|_ __ ___  ___  
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \ 
# | |  | |  __/\__ \ | | |  _| | |  __/  __/ 
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___| 
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
#
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------
"""
Particle-Kernel-Domains describe the potential interaction between a set of particles and a set of kernel kernel functions.
In meshfree methods, particles may convect through the spatial domain, and in general they may interact with different kernel functions at different times.
For certain simulations, not all particles should interact with all kernel functions.

Hence, we designate possible interactions between particles and kernel functions by the ParticleKernelDomain class.
"""

from edelweissmeshfree.meshfree.kernelfunctions.base.basemeshfreekernelfunction import (
    BaseMeshfreeKernelFunction,
)
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticleKernelDomain:
    """A class to describe the potential interaction between a set of particles and a set of kernel kernel functions.

    Parameters
    ----------
    particles
        The list of particles.
    kernelFunctions
        The list of kernel functions.
    """

    def __init__(self, particles: list[BaseParticle], meshfreeKernelFunctions: list[BaseMeshfreeKernelFunction]):
        self._particles = particles
        self._kernelFunctions = meshfreeKernelFunctions

    @property
    def particles(self) -> list[BaseParticle]:
        return self._particles

    @property
    def meshfreeKernelFunctions(self) -> list[BaseMeshfreeKernelFunction]:
        return self._kernelFunctions

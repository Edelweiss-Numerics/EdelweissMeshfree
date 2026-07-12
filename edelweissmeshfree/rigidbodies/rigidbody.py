from abc import ABC, abstractmethod


class RigidBody(ABC):
    """
    Abstract Base Class for all rigid bodies in the explicit and implicit solver framework.
    """

    rpNode = None

    @abstractmethod
    def updateKinematics(self, timeStep=None):
        """
        Update the kinematics of the rigid body according to its prescribed or computed motion.

        Parameters
        ----------
        timeStep : TimeStep, optional
            The current time step instance.
        """

    @abstractmethod
    def getCurrentKinematics(self):
        """
        Retrieve the current kinematic state of the rigid body.

        Returns
        -------
        tuple
            Typically returns (displacement, rotation, initial_position) or similar state.
        """

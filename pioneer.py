import numpy as np

class Pioneer:
    def __init__(self, sim, name='/PioneerP3DX'):
        """
        Initialize the Pioneer robot interface.

        :param sim: The Coppeliasim simulation object.
        :param name: The name of the robot in the scene (default: '/PioneerP3DX').
        """
        self.sim = sim
        self.name = name
        
        # Get handles for the robot components
        self.handle = sim.getObject(name)
        self.left_motor = sim.getObject(name + '/leftMotor')
        self.right_motor = sim.getObject(name + '/rightMotor')
        
        # Pioneer P3DX Physical Parameters
        self.wheel_radius = 0.0975  # meters
        self.wheel_separation = 0.33  # meters (approximate for P3DX)

    def set_speed(self, linear_speed: float, angular_speed: float):
        """
        Set the linear and angular speed of the robot.

        :param linear_speed: Linear speed in m/s (forward velocity).
        :param angular_speed: Angular speed in rad/s (rotational velocity).
        """
        # Differential drive kinematics
        # v_l = v - (omega * L) / 2
        # v_r = v + (omega * L) / 2
        v_left = linear_speed - (angular_speed * self.wheel_separation) / 2.0
        v_right = linear_speed + (angular_speed * self.wheel_separation) / 2.0
        
        # Convert linear wheel velocities to joint angular velocities
        # omega_wheel = v_wheel / R
        w_left = v_left / self.wheel_radius
        w_right = v_right / self.wheel_radius
        
        self.sim.setJointTargetVelocity(self.left_motor, w_left)
        self.sim.setJointTargetVelocity(self.right_motor, w_right)

    def get_pose(self):
        """
        Get the robot's current position and orientation.

        :return: A tuple containing (x, y, theta) where theta is the yaw angle in radians.
        """
        # Get position relative to the world (-1)
        position = self.sim.getObjectPosition(self.handle, -1)
        
        # Get orientation relative to the world (-1)
        # Returns Euler angles (alpha, beta, gamma). For standard Z-up, gamma is yaw.
        orientation = self.sim.getObjectOrientation(self.handle, -1)
        
        x = position[0]
        y = position[1]
        theta = orientation[2]  # Yaw
        
        return x, y, theta

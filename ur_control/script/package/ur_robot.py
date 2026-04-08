import sys
import time
import yaml
import numpy as np
from enum import Enum
from scipy.spatial.transform import Rotation

# UR RTDE Libraries
import rtde_control
import rtde_receive
import rtde_io


# [Optional] If you have a gripper driver, import it here
# from robotiq_gripper import RobotiqGripper

class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


class URRobot:
    def __init__(self, config: dict):
        print("Initializing UR Robot...")

        self.ip = config.get("robot_ip", "192.168.1.102")  # Default IP if not in config

        try:
            # Initialize RTDE interfaces
            self.rtde_c = rtde_control.RTDEControlInterface(self.ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.ip)
            self.rtde_io = rtde_io.RTDEIOInterface(self.ip)
            print("UR Robot connected.")
        except Exception as e:
            print(f"Error connecting to UR robot at {self.ip}: {e}")
            sys.exit(1)

        # UR specific settings
        self.tcp_acceleration = config.get("acceleration", 0.3)
        self.tcp_velocity = config.get("velocity", 0.3)
        self.joint_acceleration = config.get("acceleration", 0.3)
        self.joint_velocity = config.get("velocity", 0.3)

        # Initialize Gripper (Example for Robotiq, adjust as needed)
        # self.gripper = RobotiqGripper(self.ip)
        # self.gripper.activate()

    # ==========================#
    # Properties               #
    # ==========================#
    @property
    def end_effector_position(self) -> np.ndarray:
        """Returns [x, y, z] in meters"""
        return np.array(self.rtde_r.getActualTCPPose()[:3])

    @property
    def end_effector_orientation(self) -> np.ndarray:
        """Returns quaternion [x, y, z, w]"""
        # UR returns axis-angle [rx, ry, rz], convert to quaternion
        rot_vec = self.rtde_r.getActualTCPPose()[3:]
        r = Rotation.from_rotvec(rot_vec)
        return r.as_quat()

    @property
    def current_joint_positions(self) -> np.ndarray:
        """Returns joint angles in radians"""
        return np.array(self.rtde_r.getActualQ())

    @property
    def is_steady(self) -> bool:
        return self.rtde_r.getRuntimeState() == 2  # 2 means running

    # ==========================#
    # Helper Utils             #
    # ==========================#
    def _quat_to_rotvec(self, quat):
        """Helper to convert [x,y,z,w] quaternion to [rx,ry,rz] for UR"""
        return Rotation.from_quat(quat).as_rotvec().tolist()

    # ==========================#
    # Move functions           #
    # ==========================#
    def move_to_joint_position(
            self,
            joint_positions: list,
            rel_vel: float = None,
            asynch: bool = False,
    ):
        """Move to joint angles (radians)"""
        vel = rel_vel if rel_vel else self.joint_velocity
        acc = self.joint_acceleration

        # moveJ(q, speed, acceleration, asynchronous)
        self.rtde_c.moveJ(joint_positions, vel, acc, asynch)

    def move_abs(
            self,
            goal_pos: np.ndarray,
            rel_vel: float = None,
            goal_ori: np.ndarray = None,  # Quaternion [x,y,z,w]
            asynch: bool = False,
    ):
        """Move linearly in Cartesian space (moveL)"""
        vel = rel_vel if rel_vel else self.tcp_velocity
        acc = self.tcp_acceleration

        # Prepare target pose [x, y, z, rx, ry, rz]
        target_pose = list(goal_pos)

        if goal_ori is not None:
            # Use provided orientation
            target_rotvec = self._quat_to_rotvec(goal_ori)
        else:
            # Keep current orientation
            target_rotvec = self.rtde_r.getActualTCPPose()[3:]

        target_pose.extend(target_rotvec)

        # moveL(pose, speed, acceleration, asynchronous)
        self.rtde_c.moveL(target_pose, vel, acc, asynch)

    def move_rel(
            self,
            direction: np.ndarray,
            rel_vel: float = None,
            goal_ori: np.ndarray = None,
            asynch: bool = False,
    ):
        """Relative movement in Cartesian space"""
        current_pose = self.rtde_r.getActualTCPPose()
        current_pos = np.array(current_pose[:3])

        # Calculate new position
        new_pos = current_pos + direction

        # Calculate new orientation (if provided)
        if goal_ori is not None:
            # If a specific target orientation is given
            self.move_abs(new_pos, rel_vel, goal_ori, asynch)
        else:
            # If no orientation change, just translate
            # Note: For strict relative orientation, logic is more complex,
            # but usually "keep current orientation" is desired.
            target_pose = list(new_pos) + current_pose[3:]
            vel = rel_vel if rel_vel else self.tcp_velocity
            self.rtde_c.moveL(target_pose, vel, self.tcp_acceleration, asynch)

    def move_in_axis_rel(
            self,
            axis: Axis,
            dist: float,
            rel_vel: float = None,
            asynch: bool = False,
    ):
        direction = np.zeros(3)
        direction[axis.value] = dist
        self.move_rel(direction, rel_vel, asynch=asynch)

    def stop(self):
        self.rtde_c.stopJ(2.0)

    # ==========================#
    # Gripper functions         #
    # ==========================#
    # NOTE: You need to implement the specific logic for your gripper here.
    # Below is a generic interface.

    def move_gripper(self, width: float, speed: float = None, force: float = None):
        """
        Move gripper to width (0-100% or 0-0.08m depending on gripper).
        Example implementation for Robotiq 2F-85 usually maps 0-255.
        """
        # Example using digital output if it's a simple pneumatic gripper
        # if width > 0.5:
        #     self.rtde_io.setStandardDigitalOut(0, True)
        # else:
        #     self.rtde_io.setStandardDigitalOut(0, False)
        print(f"Moving gripper to {width} (Implement me!)")
        pass

    def open_gripper(self):
        print("Opening gripper...")
        self.move_gripper(1.0)  # Assuming 1.0 is open

    def close_gripper(self):
        print("Closing gripper...")
        self.move_gripper(0.0)  # Assuming 0.0 is closed

    # ==========================#
    # Utility / Test           #
    # ==========================#
    def disconnect(self):
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()


# ==========================#
# Test Functions           #
# ==========================#

def test_connection(config: dict):
    robot = URRobot(config)
    print(f"Current Joint Pos: {robot.current_joint_positions}")
    print(f"Current TCP Pos: {robot.end_effector_position}")
    print(f"Current TCP Ori (Quat): {robot.end_effector_orientation}")
    robot.disconnect()


def test_motion(config: dict):
    robot = URRobot(config)
    print("Moving relative Z +0.05m...")
    robot.move_rel(np.array([0, 0, 0.05]), rel_vel=0.1)
    time.sleep(1)
    print("Moving back...")
    robot.move_rel(np.array([0, 0, -0.05]), rel_vel=0.1)
    robot.disconnect()


if __name__ == "__main__":
    # Create a dummy config for testing if file doesn't exist
    config = {
        "robot_ip": "192.168.1.102",  # CHANGE ME
        "velocity": 0.1,
        "acceleration": 0.1
    }

    # Try loading from file if available
    try:
        with open("calibration_config.yaml", "r") as f:
            file_config = yaml.safe_load(f)
            if file_config:
                config.update(file_config)
    except FileNotFoundError:
        pass

    test_connection(config)
    # Uncomment to test actual movement (BE CAREFUL)
    # test_motion(config)
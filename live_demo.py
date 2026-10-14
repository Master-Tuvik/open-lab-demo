import json
import numpy as np
from environment import Environment, LocationType
from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from building_blocks import Building_Blocks
from visualizer import Visualize_UR
from Control_robot import BaseRobot
from numpy import pi
import time
from Experiment import Experiment

def draw_two_robots():
    ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
    ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)
    env = Environment(ur_params=ur_params_right)
    right_arm_rotation = [0, 0, -np.pi/2] 
    left_arm_rotation = [0, 0, np.pi/2]
    transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT],ur_rotation=right_arm_rotation)
    transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT],ur_rotation=left_arm_rotation)
    visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                transform_left_arm=transform_left_arm)
    right_meeting_conf = [
                    -2.9934641040971934,
                    -2.930924293268491,
                    0.8826038548093044,
                    2.048320438459186,
                    0.1481285494926033,
                    -3.1415926535897927
                ]
    right_meeting_conf[0] += np.pi/2
    left_meeting_conf = [
                    2.8495422749821753,
                    -1.0553575187730182,
                    0.8826038548093021,
                    0.17275366396371578,
                    2.849542274982176,
                    -2.0366847201866515e-16
                ]
    left_meeting_conf[0] -= np.pi/2
    visualizer.draw_two_robots(conf_left=left_meeting_conf,conf_right=right_meeting_conf)

def connect_move_robots():
    left_arm_ip = "192.168.0.17"  # TODO
    right_arm_ip = "192.168.0.10"  # TODO
    robot = BaseRobot(left_arm_ip, right_arm_ip)
    home_config = [0, -pi / 2, 0, -pi / 2, 0, 0]
    robot.robot_left.move_home()
    robot.robot_right.move_home()
    time.sleep(2)
    p1 = np.deg2rad([9.51, -118.24, -97.15, -56.06, 90.18, 9.58])
    p2 = np.deg2rad([-106.02, -133.07, -94, 29.18, 16.48, -66.73])
    path = [home_config, p1, p2]
    print("Testing movement for right arm from", path[1], "to", path[2])
    speed = 0.5
    acceleration = 0.5
    blend_radius = 0.05
    path_with_params = [
        [*target_config, speed, acceleration, blend_radius] for target_config in path
    ]
    path_with_params[-1][-1] = 0
    robot.robot_right.moveJ(path_with_params, asynchronous=False) # Default is false, this means the function will block until the movement has completed.
    # https://sdurobotics.gitlab.io/ur_rtde/api/api.html#_CPPv4N7ur_rtde20RTDEControlInterface5moveJERKNSt6vectorINSt6vectorIdEEEEb

def run_json(json_path):
    left_arm_ip = "192.168.0.17"  # TODO
    right_arm_ip = "192.168.0.10"  # TODO
    robot = BaseRobot(left_arm_ip, right_arm_ip)
    home_config = [0, -pi / 2, 0, -pi / 2, 0, 0]
    robot.robot_left.move_home()
    robot.robot_right.move_home()
    time.sleep(2)

    with open(json_path, 'r') as openfile:
            # Reading from json file
            steps = json.load(openfile)
            for step in steps:
                # iterate over the step elements
                for i in range(len(step["active_id"])):
                    print(step["description"][i])
                    # which arm are we moving?
                    arm_id = step["active_id"][i]
                    robot.set_active_robot(arm_id)
                    # first, check gripper pre status
                    robot.gripper_action(robot.active_robot, step["gripper_pre"][i])
                    # now move according to the path
                    curr_conf = step["path"][i][0]
                    if step["command"][i] == "move":
                        # robot.move([curr_conf, conf])
                        robot.move_path(step["path"][i],False) # Default is false, this means the function will block until the movement has completed.
                        # super(robot_interface.RobotInterfaceWithGripper, robot.active_robot).move_path([curr_conf, conf])
                        # super(robot_interface.RobotInterfaceWithGripper, robot.active_robot).getInverseKinematics()
                    elif step["command"][i] == "movel":
                        relative_pose = step["path"][i]
                        robot.moveL_relative(relative_pose)
                    # lastly, check gripper post status
                    robot.gripper_action(robot.active_robot, step["gripper_post"][i])

def create_json():
    exp1 = Experiment()
    exp1.plan_experiment()
if __name__ == '__main__':
    # draw_two_robots()
    # connect_move_robots()
    # run_json("plan.json")
    create_json()
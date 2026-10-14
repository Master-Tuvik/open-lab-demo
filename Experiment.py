import json
import time
from enum import Enum
import numpy as np
from matplotlib.sankey import RIGHT

from environment import Environment
from inverse_kinematics import IKType

from kinematics import UR5e_PARAMS, UR5e_without_camera_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR

import inverse_kinematics

from environment import LocationType


def log(msg):
    written_log = f"STEP: {msg}"
    print(written_log)
    dir_path = r"./outputs/"
    with open(dir_path + 'output.txt', 'a') as file:
        file.write(f"{written_log}\n")


def get_cube_approach(cube, bb, arm_location):
    cube_approaches = inverse_kinematics.get_IK_for_manipulator(
        points=[cube], bb=bb, active_arm=arm_location, type=IKType.CUBE_APPROACH
    )[0]
    return cube_approaches[0].reshape(-1)

class Gripper(str, Enum):
    OPEN = "OPEN",
    CLOSE = "CLOSE"
    STAY = "STAY"

def get_shifted_cubes(cubes_in_original_area_pre_shift, cubes_already_moved_pre_shift, env):
    cubes_already_moved = cubes_in_original_area = []
    for cube in cubes_in_original_area_pre_shift:
        cubes_in_original_area.append(cube + env.cube_area_corner[LocationType.RIGHT])
    for cube in cubes_already_moved_pre_shift:
        cubes_already_moved.append(cube + env.cube_area_corner[LocationType.LEFT])
    return [*cubes_already_moved, *cubes_in_original_area]


def update_environment(env, active_arm, static_arm_conf, cubes_positions):
    env.set_active_arm(active_arm)
    env.update_obstacles(cubes_positions, static_arm_conf)


class Experiment:
    def __init__(self, cubes=None):
        # environment params
        self.cubes = cubes
        self.right_arm_base_delta = 0  # deviation from the first link 0 angle
        self.left_arm_base_delta = 0  # deviation from the first link 0 angle
        self.right_arm_meeting_safety = None
        self.left_arm_meeting_safety = None

        # tunable params
        self.max_itr = 1000
        self.max_step_size = 0.5
        self.goal_bias = 0.05
        self.resolution = 0.1
        # start confs
        self.right_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        self.left_arm_home = np.deg2rad([0, -90, 0, -90, 0, 0])
        # result dict
        self.experiment_result = []

    def push_step_info_into_single_cube_passing_data(self, description, active_id, command, static_conf, path, cubes, gripper_pre, gripper_post):
        self.experiment_result[-1]["description"].append(description)
        self.experiment_result[-1]["active_id"].append(active_id)
        self.experiment_result[-1]["command"].append(command)
        self.experiment_result[-1]["static"].append(static_conf)
        self.experiment_result[-1]["path"].append(path)
        self.experiment_result[-1]["cubes"].append(cubes)
        self.experiment_result[-1]["gripper_pre"].append(gripper_pre)
        self.experiment_result[-1]["gripper_post"].append(gripper_post)

    def plan_single_arm(self, planner, start_conf, goal_conf, description, active_id, command, static_arm_conf, cubes_real,
                            gripper_pre, gripper_post):
        path, cost = planner.find_path(start_conf=start_conf,
                                       goal_conf=goal_conf,
                                       manipulator=active_id)
        # create the arm plan
        self.push_step_info_into_single_cube_passing_data(description,
                                                          active_id,
                                                          command,
                                                          static_arm_conf.tolist(),
                                                          [path_element.tolist() for path_element in path],
                                                          [list(cube) for cube in cubes_real],
                                                          gripper_pre,
                                                          gripper_post)

    def plan_single_cube_passing(self, cube_i, cubes,
                                 left_arm_start, right_arm_start,
                                 env, bb, planner, left_arm_transform, right_arm_transform,):
        # add a new step entry
        single_cube_passing_info = {
            "description": [],  # text to be displayed on the animation
            "active_id": [],  # active arm id
            "command": [],
            "static": [],  # static arm conf
            "path": [],  # active arm path
            "cubes": [],  # coordinates of cubes on the board at the given timestep
            "gripper_pre": [],  # active arm pre path gripper action (OPEN/CLOSE/STAY)
            "gripper_post": []  # active arm pre path gripper action (OPEN/CLOSE/STAY)
        }
        self.experiment_result.append(single_cube_passing_info)
        # fix obstacles and update env
        ###############################################################################
        # set variables
        description = "right_arm => [start -> cube pickup], left_arm static"
        active_arm = LocationType.RIGHT
        # start planning
        log(msg=description)
        # fix obstacles and update env
        cubes_already_moved_pre_shift = cubes[0:cube_i]
        cubes_in_original_area_pre_shift = cubes[cube_i:]
        cubes_real = get_shifted_cubes(cubes_in_original_area_pre_shift, cubes_already_moved_pre_shift, env)

        update_environment(env, active_arm, left_arm_start, cubes_real)

        cube_approach = get_cube_approach(cubes[cube_i], bb, active_arm)
        # plan the path
        self.plan_single_arm(planner, right_arm_start, cube_approach, description, active_arm, "move",
                                 left_arm_start, cubes_real, Gripper.OPEN, Gripper.STAY)
        ###############################################################################

        self.push_step_info_into_single_cube_passing_data("picking up a cube: go down",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          list(self.left_arm_home),
                                                          [0, 0, -0.14],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.CLOSE)

        ###############################################################################

        self.push_step_info_into_single_cube_passing_data("picking up a cube: go up",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          list(self.left_arm_home),
                                                          [0, 0, 0.14],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.STAY)

        ###############################################################################
        # set variables
        description = "right_arm => [cube -> meeting], left_arm static"
        active_arm = LocationType.RIGHT
        # start planning
        log(msg=description)
        # fix obstacles and update env
        cubes_already_moved_pre_shift = cubes[0:cube_i]
        cubes_in_original_area_pre_shift = cubes[cube_i + 1:]
        cubes_real = get_shifted_cubes(cubes_in_original_area_pre_shift, cubes_already_moved_pre_shift, env)

        update_environment(env, active_arm, left_arm_start, cubes_real)
        # plan the path
        self.plan_single_arm(planner, cube_approach, self.right_arm_meeting_safety, description, active_arm, "move",
                                 left_arm_start, cubes_real, Gripper.STAY, Gripper.STAY)
        ###############################################################################
        description = "right_arm static, left_arm => [start -> meeting]"
        active_arm = LocationType.LEFT
        static_arm_conf = self.right_arm_meeting_safety

        log(msg=description)
        # fix obstacles and update env
        update_environment(env, active_arm, self.right_arm_meeting_safety, cubes_real)
        # plan the path
        self.plan_single_arm(planner, left_arm_start, self.left_arm_meeting_safety, description, active_arm, "move",
                                 static_arm_conf, cubes_real, Gripper.STAY, Gripper.OPEN)
        ###############################################################################
        # rotate the left arm
        left_arm_meeting_safety_rotated = np.copy(self.left_arm_meeting_safety)
        if left_arm_meeting_safety_rotated[5] > np.pi / 2:
            left_arm_meeting_safety_rotated[5] -= np.pi / 2
        else:
            left_arm_meeting_safety_rotated[5] += np.pi / 2

        self.push_step_info_into_single_cube_passing_data("prepare for meeting: rotate left arm",
                                                          LocationType.LEFT,
                                                          "move",
                                                          list(static_arm_conf),
                                                          [list(self.left_arm_meeting_safety),
                                                           list(left_arm_meeting_safety_rotated)],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.STAY)
        ###############################################################################
        log(msg="passing the cube.")
        # Plan experiment

        self.push_step_info_into_single_cube_passing_data("passing the cube: right arm approaches",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          list(self.left_arm_home),
                                                          [-0.08, 0, 0],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.STAY)

        self.push_step_info_into_single_cube_passing_data("passing the cube: left arm approaches",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          list(self.left_arm_home),
                                                          [-0.08, 0, 0],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.CLOSE)

        self.push_step_info_into_single_cube_passing_data("passing the cube: right arm goes away",
                                                          LocationType.RIGHT,
                                                          "movel",
                                                          list(self.left_arm_home),
                                                          [0.08, 0, 0],
                                                          [],
                                                          Gripper.OPEN,
                                                          Gripper.STAY)

        self.push_step_info_into_single_cube_passing_data("passing the cube: left arm goes away",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          list(self.left_arm_home),
                                                          [0.08, 0, 0],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.STAY)

        self.push_step_info_into_single_cube_passing_data("leave for meeting: rotate left arm back",
                                                          LocationType.LEFT,
                                                          "move",
                                                          list(static_arm_conf),
                                                          [list(left_arm_meeting_safety_rotated),
                                                           list(self.left_arm_meeting_safety)],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.STAY)

        ###############################################################################
        description = "right_arm static, left_arm => [meeting -> cube put-down]"
        active_arm = LocationType.LEFT
        static_arm_conf = self.right_arm_meeting_safety

        log(msg=description)
        # fix obstacles and update env
        update_environment(env, active_arm, static_arm_conf, cubes_real)
        left_arm_cube_putdown = get_cube_approach(cubes[cube_i], bb, active_arm)
        # plan the path
        self.plan_single_arm(planner, self.left_arm_meeting_safety, left_arm_cube_putdown, description, active_arm,
                             "move", static_arm_conf, cubes_real, Gripper.STAY, Gripper.STAY)

        ###############################################################################
        self.push_step_info_into_single_cube_passing_data("dropping a cube: go down",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          list(self.right_arm_home),
                                                          [0,0, -0.25],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.OPEN)

        self.push_step_info_into_single_cube_passing_data("dropping a cube: go up",
                                                          LocationType.LEFT,
                                                          "movel",
                                                          list(self.right_arm_home),
                                                          [0,0, 0.25],
                                                          [],
                                                          Gripper.STAY,
                                                          Gripper.STAY)

        return left_arm_cube_putdown, self.right_arm_meeting_safety


    def plan_experiment(self):
        start_time = time.time()

        exp_id = 2
        ur_params_right = UR5e_PARAMS(inflation_factor=1.0)
        ur_params_left = UR5e_without_camera_PARAMS(inflation_factor=1.0)

        env = Environment(ur_params=ur_params_right)

        transform_right_arm = Transform(ur_params=ur_params_right, ur_location=env.arm_base_location[LocationType.RIGHT])
        transform_left_arm = Transform(ur_params=ur_params_left, ur_location=env.arm_base_location[LocationType.LEFT])

        env.arm_transforms[LocationType.RIGHT] = transform_right_arm
        env.arm_transforms[LocationType.LEFT] = transform_left_arm

        bb = Building_Blocks(env=env,
                             resolution=self.resolution,
                             p_bias=self.goal_bias, )

        rrt_star_planner = RRT_STAR(max_step_size=self.max_step_size,
                                    max_itr=self.max_itr,
                                    bb=bb)
        visualizer = Visualize_UR(ur_params_right, env=env, transform_right_arm=transform_right_arm,
                                  transform_left_arm=transform_left_arm)
        # cubes
        if self.cubes is None:
            self.cubes = self.get_cubes_for_experiment(exp_id)

        log(msg="calculate meeting point for the test.")

        # get meeting point confs for both arms
        meeting_conf_right_arm_regular = meeting_conf_left_arm_regular = []
        meet_height = 0.5
        safety_distance = 0.05

        meeting_point = bb.get_meeting_point_for_arms(meet_height)
        meeting_point[1] -= safety_distance
        update_environment(env, LocationType.RIGHT, self.left_arm_home, self.cubes)
        meeting_confs1 = inverse_kinematics.get_IK_for_manipulator(
            points=[meeting_point], bb=bb, active_arm=LocationType.RIGHT
        )[0]
        if len(meeting_confs1) == 0:
            print(f"Problem with the right arm at {meet_height}")
            return
        for i in range(len(meeting_confs1)):
            possible_conf1 = meeting_confs1[i]
            meeting_conf_right_arm_regular_local = possible_conf1.reshape(-1)

            meeting_point[1] += 2 * safety_distance
            meeting_point[0] += 0
            update_environment(env, LocationType.LEFT, meeting_conf_right_arm_regular_local, self.cubes)

            meeting_confs2 = inverse_kinematics.get_IK_for_manipulator(
                points=[meeting_point], bb=bb, active_arm=LocationType.LEFT
            )[0]
            if len(meeting_confs2) == 0:
                continue
            meeting_conf_left_arm_regular = meeting_confs2[0].reshape(-1)
            meeting_conf_right_arm_regular = meeting_conf_right_arm_regular_local
            break
        if len(meeting_conf_left_arm_regular) == 0:
            print("no left_arm solution")
            return
        meeting_point[1] -= safety_distance
        meeting_point[0] -= 0

        self.right_arm_meeting_safety = meeting_conf_right_arm_regular
        self.left_arm_meeting_safety = meeting_conf_left_arm_regular

        log(msg="start planning the experiment.")
        left_arm_start = self.left_arm_home
        right_arm_start = self.right_arm_home
        for i in range(len(self.cubes)):
            left_arm_start, right_arm_start = (
                self.plan_single_cube_passing(i, self.cubes, left_arm_start, right_arm_start,
                                              env, bb, rrt_star_planner, left_arm_start, right_arm_start))
        t2 = time.time()
        print(f"It took t={t2 - start_time} seconds")
        # save the experiment to data:
        # Serializing json
        json_object = json.dumps(self.experiment_result, indent=4)
        # Writing to sample.json
        dir_path = r"./outputs/"
        with open(dir_path + "plan.json", "w") as outfile:
            outfile.write(json_object)
        # show the experiment then export it to a GIF
        # self.show_all_experiment(visualizer, dir_path + "plan.json")
        # visualizer.animate_by_pngs()

    def get_cubes_for_experiment(self, experiment_id):
        cube_side = 0.04
        cubes = []
        if experiment_id == 1:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            cubes.append([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
        elif experiment_id == 2:
            x_min = 0.0
            x_max = 0.4
            y_min = 0.0
            y_max = 0.4
            dx = x_max - x_min
            dy = y_max - y_min
            x_slice = dx / 4.0
            y_slice = dy / 2.0
            # row 1: cube 1
            cubes.append([x_min + 0.5 * x_slice, y_min + 1.5 * y_slice, cube_side / 2.0])
            # row 1: cube 2
            cubes.append([x_min + 0.5 * x_slice, y_min + 0.5 * y_slice, cube_side / 2.0])
        return cubes

    def show_all_experiment(self, visualizer, experiment_json):
        # Opening JSON file
        with open(experiment_json, 'r') as openfile:
            # Reading from json file
            steps = json.load(openfile)
            for step in steps:
                # iterate over the step dicts
                for i in range(len(step["active_id"])):
                    # display the movement
                    if step["command"][i] == "movel":
                        continue
                    path = [np.array(path_conf) for path_conf in step["path"][i]]
                    if step["description"][i] == "passing the cube.":
                        visualizer.show_stage_moving_single_arm(step["active_id"][i], path, np.array(step["static"][i]), step["cubes"][i],
                                                            step["description"][i], True)
                    else:
                        visualizer.show_stage_moving_single_arm(step["active_id"][i], path, np.array(step["static"][i]), step["cubes"][i],
                                                            step["description"][i])
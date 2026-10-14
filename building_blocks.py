import math
import queue
import random
import time
import threading
import multiprocessing
import timeit

import numpy as np

import environment
from environment import LocationType

class Building_Blocks(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''

    def __init__(self, env, resolution=0.1, p_bias=0.05, special_bias=False):
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.07])
        # testing variables
        self.t_curr = 0
        self.special_bias = special_bias

        # self.TWO_PI = 2 * math.pi

    def sample(self, goal_conf, bias_to_goal) -> np.array:
        """
        sample random configuration
        @param bias_to_goal:  bias towards goal? (boolean)
        @param goal_conf - the goal configuration
        """
        # hint - use self.ur_params.mechamical_limits
        p = np.random.rand()
        if p <= self.p_bias and bias_to_goal:
            return np.array(goal_conf), True
        joints = self.env.ur_params.mechamical_limits
        conf = [np.random.uniform(joints[joint][0], joints[joint][1]) for joint in joints.keys()]
        return np.array(conf), False

    def get_meeting_point_for_arms(self, height):
        middle_x = (self.env.arm_base_location[LocationType.LEFT][0] + self.env.arm_base_location[LocationType.RIGHT][0]) / 2.0
        middle_y = (self.env.arm_base_location[LocationType.LEFT][1] + self.env.arm_base_location[LocationType.RIGHT][1]) / 2.0
        return [middle_x, middle_y, height]

    def is_in_collision(self, conf) -> bool:
        """check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration 
        """
        # hint: use self.transform.conf2sphere_coords(), self.ur_params.sphere_radius, self.env.obstacles
        global_sphere_coords = self.env.arm_transforms[self.env.active_arm].conf2sphere_coords(conf)
        # arm - arm collision
        # iterate over all links, assuming the links are arranged by their physical order
        global_sphere_coords_list = global_sphere_coords.keys()
        first_link_id = -1
        for first_link in global_sphere_coords_list:
            first_link_id += 1
            first_link_spheres = global_sphere_coords[first_link]
            # compare it to different links
            second_link_id = -1
            for second_link in global_sphere_coords_list:
                second_link_id += 1
                if second_link_id <= first_link_id:
                    continue
                if second_link_id == first_link_id + 1:
                    # don't compare spheres on neighboring links
                    # the official assumption (as stated by the staff) is that neighbors cant collide
                    continue
                if first_link_id >= 3 and second_link_id <= first_link_id:
                    # dont compare the end with the beginning, it cant happen.
                    continue
                # if first_link_id >=2 and second_link_id == 5:
                #     continue
                second_link_spheres = global_sphere_coords[second_link]
                for i in range(0, len(first_link_spheres)):
                    for j in range(0, len(second_link_spheres)):
                        sphere1 = first_link_spheres[i]
                        sphere2 = second_link_spheres[j]
                        if (math.sqrt((sphere1[0] - sphere2[0]) ** 2 + (sphere1[1] - sphere2[1]) ** 2
                                      + (sphere1[2] - sphere2[2]) ** 2) <=
                                self.env.ur_params.sphere_radius[first_link] + self.env.ur_params.sphere_radius[
                                    second_link]):
                            return True
            # arm - obstacle collision
            for sphere in first_link_spheres:
                if first_link_id > 0 and sphere[2] <= self.env.ur_params.sphere_radius[first_link]:
                    # don't check floor collision for the shoulder. it's always 90 degrees to the floor.
                    # print("Floor-collision: Link {0} with the floor".format(first_link))
                    return True
                # walls
                right_limit = self.env.tables[LocationType.RIGHT][1][0]
                top_limit = self.env.tables[LocationType.LEFT][0][1]
                if first_link_id > 0 and sphere[1] >= right_limit:
                    # don't check floor collision for the shoulder. it's always 90 degrees to the floor.
                    # print("wall-right collision: Link {0} with the floor".format(first_link))
                    return True
                if first_link_id > 0 and sphere[0] >= top_limit:
                    # don't check floor collision for the shoulder. it's always 90 degrees to the floor.
                    # print("wall-top collision: Link {0} with the floor".format(first_link))
                    return True
                # window collision? Can this be ignored for my experiment?
                # obstacle collision?
                for sphere_obs in self.env.obstacles:
                    dist = math.sqrt((sphere[0] - sphere_obs[0]) ** 2 +
                                     (sphere[1] - sphere_obs[1]) ** 2 + (sphere[2] - sphere_obs[2]) ** 2)
                    if dist <= self.env.ur_params.sphere_radius[first_link] + self.env.radius:
                        return True
        return False

    def get_max_extension(self, prev_conf, current_conf):
        min_check_value = 3
        max_angle_diff = np.max(np.abs(prev_conf - current_conf))
        # generate the confs
        number_of_configurations_to_check = math.ceil(max((max_angle_diff / self.resolution), min_check_value))
        confs_to_check = {}
        # initiate the conf dict with lists (that will contain the confs)
        for j in range(0, number_of_configurations_to_check):
            confs_to_check[j] = []
        # populate the conf dict
        for i in range(0, len(prev_conf)):
            angles_to_check = np.linspace(prev_conf[i], current_conf[i], num=number_of_configurations_to_check,
                                          endpoint=True)
            # insert the i-th link checking-angles into the conf dict
            for j in range(0, number_of_configurations_to_check):
                confs_to_check[j].append(angles_to_check[j])
        # check all the confs for collisions
        biggest_i = 0
        farthest_conf = None
        for i in range(0, number_of_configurations_to_check):
            # self.fix_conf(confs_to_check[i])
            if not self.is_in_collision(confs_to_check[i]):
                biggest_i = i
                farthest_conf = confs_to_check[i]
        if farthest_conf is not None:
            return np.array(farthest_conf)
        if biggest_i > 0:
            return np.round(np.linalg.norm(prev_conf - current_conf) / biggest_i, 2)
        return 0

    def local_planner(self, prev_conf, current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        # hint: use self.is_in_collision()
        min_check_value = 3
        # go over the links, getting the largest angle diff between confs
        max_angle_diff = 0
        max_angle_diff = np.max(np.abs(prev_conf - current_conf))
        # generate the confs
        number_of_configurations_to_check = math.ceil(max((max_angle_diff / self.resolution), min_check_value))
        confs_to_check = {}
        # initiate the conf dict with lists (that will contain the confs)
        for j in range(0, number_of_configurations_to_check):
            confs_to_check[j] = []
        # populate the conf dict
        for i in range(0, len(prev_conf)):
            angles_to_check = np.linspace(prev_conf[i], current_conf[i], num=number_of_configurations_to_check,
                                          endpoint=True)
            # insert the i-th link checking-angles into the conf dict
            for j in range(0, number_of_configurations_to_check):
                confs_to_check[j].append(angles_to_check[j])
        # check all the confs for collisions
        threads = []
        results = []  # List to store thread results
        for i in range(0, number_of_configurations_to_check):
            # self.fix_conf(confs_to_check[i])
            if self.is_in_collision(confs_to_check[i]):
                # collision
                # print("Path-collision: path between confs is in collision".format())
                return False
        return True


    def edge_cost(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        sum = 0
        try:
            for i in range(len(self.cost_weights)):
                sum += math.sqrt(self.cost_weights[i] * (conf1[i] - conf2[i]) ** 2)
                # d_i = conf1[i] - conf2[i]
                ## wrapping around for S^1
                # min_diff_sqr = min((d_i-2*np.pi)**2,(d_i+2*np.pi)**2, d_i**2)
                # sum += (self.cost_weights[i] * d_i)**2
        except:
            pass
        return math.sqrt(sum)

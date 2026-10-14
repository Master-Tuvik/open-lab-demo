import numpy as np
import time

import environment
from RRTTree import RRTTree

class RRT_STAR(object):
    def __init__(self, max_step_size, max_itr, bb):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.bb = bb
        self.tree = RRTTree(bb)
        # testing variables
        self.t_curr = 0
        self.itr_no_goal_limit = 250
        self.sample_rotation = 0.1
        # self.TWO_PI = 2 * math.pi
        self.last_cost = -1
        self.last_ratio = 1

    def reset_params(self, start_conf):
        self.tree = RRTTree(self.bb)
        self.tree.AddVertex(start_conf)  # the start conf is real, the rest are relative
        self.last_cost = -1
        self.last_ratio = 1

    def find_path(self, start_conf, goal_conf, manipulator):
        if self.bb.is_in_collision(start_conf) or self.bb.is_in_collision(goal_conf):
            print ("bad start conf")
            return [], 0
        if self.bb.is_in_collision(goal_conf):
            print ("bad goal conf")
            return [], 0
        # print (self.bb.local_planner(start_conf, goal_conf))
        """Implement RRT-STAR"""
        print(f'[manipulator:{manipulator.name}] Planning from {start_conf} to {goal_conf}, step={self.max_step_size}')
        i = 1
        self.t_curr = time.time()
        plan = []
        self.reset_params(start_conf)
        goal_id = -1
        reset_count = 1
        bias_to_goal = True
        while i < self.max_itr or goal_id == -1:
            if i > self.itr_no_goal_limit and goal_id == -1 and reset_count < 100:
                print(f"Reset #{reset_count}")
                reset_count += 1
                # self.tree = RRTTree(self.bb)
                # self.tree.AddVertex(start_conf)
                self.reset_params(start_conf)
                i = 1
                # min_dist_to_goal = np.infty
                # min_state = None
                # return [], 0
            if i % 500 == 0:
                print(i, len(self.tree.vertices))#, min_dist_to_goal, min_state)
                if goal_id > 0:
                    new_cost = self.tree.vertices[goal_id].cost
                    print(f"Current cost: {new_cost}")
                    ratio = new_cost / self.last_cost
                    if self.last_cost > 0:
                        print(f"ratio: {ratio * 100.0}%")
                    if self.last_ratio != 1 and ratio == 1:
                        print(self.last_ratio, ratio)
                        break
                    self.last_ratio = ratio
                    self.last_cost = new_cost
            rand_state, goal_sampled = self.bb.sample(goal_conf, bias_to_goal)  # create a random state
            near_id, nearest_vertex = self.tree.GetNearestVertex(rand_state)
            p = np.random.rand()
            if p < self.sample_rotation:
                # try to provide a random state just by rotating
                near_state = nearest_vertex.state
                i_rnd = np.random.randint(6)
                angle_rand = np.random.uniform(low=-np.pi/8, high=np.pi/8)
                new_state = np.copy(near_state)
                new_state[i_rnd] = new_state[i_rnd] + angle_rand
            else:
                max_ext = 0
                # if goal_sampled: # and self.bb.local_planner(nearest_vertex.state, goal_conf):
                if goal_sampled:
                    # real_goal = self.bb.get_real_state_in_hypertorus(nearest_vertex.state, goal_conf)
                    # new_state = self.bb.get_max_extension(nearest_vertex.state, goal_conf)
                    new_state = self.extend(nearest_vertex.state, goal_conf)
                #     try to extend it all the way, if theres no collision
                # if max_ext > 0:
                #     new_state = self.extend(nearest_vertex.state, real_rand_state, max_ext)
                else:
                    # theres collision? extend by step_size
                    # real_rand_state = self.bb.get_real_state_in_hypertorus(nearest_vertex.state, rand_state)
                    new_state = self.extend(nearest_vertex.state, rand_state)
                    # new_state = self.bb.get_max_extension(nearest_vertex.state, real_rand_state)
            if len(new_state) == 0:
                print(f'ha? goal_id={goal_id}')
                continue
            if self.bb.local_planner(nearest_vertex.state, new_state):
                new_id = self.tree.AddVertex(new_state)
                self.tree.AddEdge(near_id, new_id, self.bb.edge_cost(nearest_vertex.state, new_state))
                k_value = self.get_k_num(i) + 1
                kNN_ids = []
                if k_value == len(self.tree.vertices):
                    kNN_ids = self.tree.vertices.keys()
                else:
                    kNN_ids, _ = self.tree.GetKNN(new_state, k_value)
                # self.test_time("pre-rewire")
                for kNN_nearest_id in kNN_ids:
                    self.rewire(kNN_nearest_id, new_id, 0)
                for kNN_nearest_id in kNN_ids:
                    self.rewire(new_id, kNN_nearest_id, 1)
                if goal_id == -1 and goal_conf[0] == new_state[0] and goal_conf[1] == new_state[1]:
                    print(f"Found the goal! in [{i}] samples. Improving the path now!")
                    goal_id = new_id
                    break #DEBUG PURPOSES
            i += 1
        if goal_id == -1:
            return [], 0
        path, cost = self.get_shortest_path(goal_id)
        return path, cost

    def extend(self, x_near, x_random, custom_ext = -1)-> np.array:
        '''
        Implement the Extend method
        @param x_near - Nearest Neighbor
        @param x_random - random sampled configuration
        return the extended configuration
        '''
        direction = np.array(x_random - x_near)
        norm = np.linalg.norm(direction)
        if norm == 0:
            print(f'rand={x_random},near={x_near}')
            return []
        if self.max_step_size > norm:
            # out of bounds
            return x_random
        local_step_size = self.max_step_size
        if custom_ext > 0:
            local_step_size = custom_ext
        direction_with_step = (direction / np.linalg.norm(direction)) * local_step_size
        new_state = x_near + direction_with_step
        # for val in new_state:
        #     if val > np.pi:
        #         pass
        return new_state

    def rewire(self, x_potential_parent_id, x_child_id, s) -> None:
        '''
        Implement the rewire method
        @param x_potential_parent_id - candidte to become a parent
        @param x_child_id - the id of the child vertex
        return None
        '''
        if x_child_id == x_potential_parent_id:
            return
        child = self.tree.vertices[x_child_id]
        parent = self.tree.vertices[x_potential_parent_id]
        edge_cost = self.bb.edge_cost(child.state, parent.state)
        new_cost = parent.cost + edge_cost
        if new_cost < child.cost:
            if self.bb.local_planner(child.state, parent.state):
                old_parent_id = self.tree.edges[x_child_id]
                self.tree.AddEdge(x_potential_parent_id, x_child_id, edge_cost)
                if x_child_id in self.tree.edges.keys():
                    old_parent = self.tree.vertices[old_parent_id]
                    old_parent.children.remove(child)
                self.propagate(child)

    # fix the cost of nodes, starting from this node
    def propagate(self, start):
        if len(start.children) > 0:
            for child in start.children:
                child.cost = start.cost + self.bb.edge_cost(start.state, child.state)
                self.propagate(child)

    def get_shortest_path(self, dest):
        '''
        Returns the path and cost from some vertex to Tree's root
        @param dest - the id of some vertex
        return the shortest path and the cost
        '''
        prev_state = np.array(self.tree.vertices[dest].state)
        path = [prev_state]
        current_id = dest
        cost = 0
        while current_id in self.tree.edges.keys():
            current_id = self.tree.edges[current_id]
            new_state = np.array(self.tree.vertices[current_id].state)
            path.insert(0, new_state)
            cost += self.bb.edge_cost(prev_state,  new_state)
        return path, cost
    
    def get_k_num(self, i):
        '''
        Determines the number of K nearest neighbors for each iteration
        '''
        if i < 300:
            k_num = 1
        elif 300 <= i < 600:
            k_num = 3
        elif 600 <= i < 1000:
            k_num=5
        elif 1000 <= i < 1500:
            k_num=6
        else:
            k_num = 7
        return k_num


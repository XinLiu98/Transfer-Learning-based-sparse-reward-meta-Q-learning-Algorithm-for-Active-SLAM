#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import numpy as np
import math
from math import pi
import random
import time
import torch

import sys
sys.path.append('/home/gzz/MQL_mujoco/rlkit/envs')
import agent
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState

goal_model_1_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '',  'Target_1', 'model.sdf')

from . import register_env
@register_env('gazebo-dir')
class GazeboDirEnv():
    def __init__(self, is_training=True, task={}, n_tasks=2, randomize_tasks=False):
        # Simulation 仿真设置
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)

        self.goal_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)

        self.pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        goal_model_position = [-4, 4]
        self.tasks = [{'target_1': target_1} for target_1 in goal_model_position]
        self._task = task
        self._goal_dir = task.get('target_1', 1)
        self._goal = self._goal_dir
        super(GazeboDirEnv, self).__init__()

        # Agent 智能体
        threshold_goal = 0.5
        threshold_collision = 0.27
        self.agent_1 = agent.Agent(num=1, threshold_goal=threshold_goal, threshold_collision=threshold_collision, flag_of_social=False)

        rospy.init_node('GazeboDirEnv')
        time.sleep(1.0)

    def step(self, action):

        self.agent_1.set_action([action[0], action[1]])
        # self.agent_1.set_action(action)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/unpause_physics service call failed")

        time.sleep(0.1)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/pause_physics service call failed")

        time.sleep(0.1)

        t1 = time.time()

        # observation, done, arrive = self.get_agent_state(1)
        observation, done, arrive,linear_speed, angular_speed = self.get_agent_state(1)
        t2 = time.time()

        # print("Step Time : {}  ms".format(round(1000*(t2-t1),2)))

        #reward_1, flag_ego_safety_1, flag_social_safety_1 = self.get_agent_reward(1)
        reward, flag_ego_safety_1, flag_social_safety_1, reward_goal, reward_collision= self.get_agent_reward(1)
        infos = dict(reward_forward=reward_goal,
                     reward_ctrl=reward_collision, task=self._task)

        #return state_all_1, reward_1, done_1 or arrive_1
        return (observation, reward, done, arrive, infos, linear_speed, angular_speed, reward_goal, reward_collision)

    def sample_tasks(self, num_tasks):
        goal_model_position = Pose()
        mode = np.random.uniform(0., 4.0, size=(num_tasks,))
        if mode < 1:
            temp_x = 3
            temp_y = 0 + (6.0 * random.random()) - 3.0
        elif mode >= 1 and mode < 2:
            temp_x = -3
            temp_y = (6.0 * random.random()) - 3.0
        elif mode >= 2 and mode < 3:
            temp_x = (6.0 * random.random()) - 3.0
            temp_y = -3
        elif mode >= 3 and mode < 4:
            temp_x = (6.0 * random.random()) - 3.0
            temp_y = 3

        goal_model_position.position.x, goal_model_position.position.y = temp_x, temp_y
        print('env-sample_tasks')
        tasks = [{'target_1': target_1} for target_1 in goal_model_position]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        print('env-reset_task')
        self._task = self.tasks[idx]
        self._goal_dir = self._task['target_1']
        self._goal = self._goal_dir
        self.reset()

    def seed(self, seed_num=0):  # this is so that you can get consistent results
        #pass  # optionally, you could add: random.seed(random_num)
        #TODO 0307
        random.seed(0)
        # print('env173',random.seed())
        return
    # 根据num获取agent状态
    def get_agent_state(self, num):

        done = False
        arrive = False

        if num == 1:
            observation, done, arrive, linear_speed, angular_speed = self.agent_1.get_state()
            # print('e181',observation.shape)
            # if observation.shape==[1,60]:
            #     self.get_agent_reset()

        return observation, done, arrive, linear_speed, angular_speed

    # 根据num获取agent reward
    def get_agent_reward(self, num):
        if num == 1:
            reward, flag_ego_safety, flag_social_safety,reward_goal, reward_collision = self.agent_1._compute_reward()
        return reward, flag_ego_safety, flag_social_safety, reward_goal, reward_collision

    # reset all agent
    def get_agent_reset(self):
        self.agent_1.reset()

    # env step

    def reset(self):
        print("=============================>  Env Reset <=============================   ", self.agent_1.cumulated_steps)
        sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
        # Reset the env
        print(1)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
            print(2)
        except (rospy.ServiceException) as e:
            print("gazebo/pause_physics service call failed")
        
        time.sleep(0.1)
        
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            self.del_model('target_1')
            print(3)
        except (rospy.ServiceException) as e:
            print("gazebo/delete_model service call failed")

        time.sleep(0.1)

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
            print(4)
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        time.sleep(0.05)
        # Reset robot
        self.robot_ModelState = ModelState()
        self.robot_ModelState.model_name = 'mobile_base'
        self.robot_ModelState.reference_frame = 'map'
        print(5)

        self.robot_ModelState.pose.position.x = (1.0 * random.random()) - 0.5
        self.robot_ModelState.pose.position.y = (1.0 * random.random()) - 0.5
        self.robot_ModelState.pose.position.z = 0.0
        print(6)

        # yaw = - pi / 2.0 / 2.0
        # mode = random.random() * 4
        # mode = 0.5
        # print('!!!!!!!!mode', mode)
        # if mode < 1:
        #     yaw = 0/ 2.0
        #     # print('!!!!!!!!!yaw', yaw)
        # elif mode >= 1 and mode < 2:
        #     yaw = -pi / 2.0
        # elif mode >= 2 and mode < 3:
        #     yaw = - pi / 2.0 / 2.0
        # elif mode >= 3 and mode < 4:
        #     yaw = pi / 2.0 / 2.0
        yaw = (6.28 * random.random()) - 3.14
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        self.robot_ModelState.pose.orientation.x = 0.0
        self.robot_ModelState.pose.orientation.y = 0.0
        self.robot_ModelState.pose.orientation.z = sin_yaw
        self.robot_ModelState.pose.orientation.w = cos_yaw
        print(7)

        self.robot_ModelState.twist.linear.x = 0.0
        self.robot_ModelState.twist.angular.z = 0.0

        self.pub.publish(self.robot_ModelState)
        time.sleep(0.05)

        self.get_agent_reset()

        # Build the targets
        # 1
        # Build the target
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_1_dir, "r").read()
            target_1 = SpawnModel
            target_1.model_name = 'target_1'  # the same with sdf name
            target_1.model_xml = goal_urdf
            goal_model_position = Pose()
            print(8)
            
            mode = random.random() * 4
            if mode < 1:
                temp_x = 3
                temp_y = 0 + (6.0 * random.random()) - 3.0
                # print('!!!!!!!x',temp_x)
                # print('!!!!!!!y', temp_y)
            elif mode >= 1 and mode < 2:
                temp_x = -3
                temp_y = (6.0 * random.random()) - 3.0
            elif mode >= 2 and mode < 3:
                temp_x = (6.0 * random.random()) - 3.0
                temp_y = -3
            elif mode >= 3 and mode < 4:
                temp_x = (6.0 * random.random()) - 3.0
                temp_y = 3

            #tasks = [{'target_1': target_1} for target_1 in SpawnModel]

            goal_model_position.position.x, goal_model_position.position.y = temp_x, temp_y
            self.agent_1.update_target((temp_x, temp_y))
            self.goal_model(target_1.model_name, target_1.model_xml, 'namespace', goal_model_position, 'world')
            # print('=============    reset  Target reset for agent_1!')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target_1")
        
        time.sleep(0.5)

        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
            print(9)
        except (rospy.ServiceException) as e:
            print("gazebo/unpause_physics service call failed")


        time.sleep(0.2)
        observation, done, arrive, linear_speed, angular_speed = self.get_agent_state(1)

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
            print(10)
        except (rospy.ServiceException) as e:
            print("gazebo/pause_physics service call failed")

        # print("=============================>  Env Reset <=============================")
        return observation
        # , tasks


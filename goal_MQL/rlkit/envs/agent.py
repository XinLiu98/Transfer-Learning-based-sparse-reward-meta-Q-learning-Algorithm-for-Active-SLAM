# -*- coding: utf-8 -*- 

# 单个机器人控制函数


import math
import time

import sys
#TODO 1
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
from sensor_msgs.msg import LaserScan

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist, Point, Pose

from nav_msgs.msg import Path
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

IMAGE_W = 160
IMAGE_H = 80
MAX_LASER_RANGE = 3.0

# reward parameter
r_arrive = 100
r_collision = -50
Cr = 100   # judge arrival
Cd = 0.5 #0.354/2   # = r when testing  # compute reward if no collision and arrival
Cp = -0.05  # time step penalty   # time step penalty

def Transformation(lidar_input, current_x, current_y, current_yaw, target_x, target_y, target_yaw):

    lidar_current = np.asarray(lidar_input)

    index_xy = np.linspace(0,360,360)
    x_current = lidar_current * np.sin(index_xy / 360.0 * math.pi).astype(np.float64)
    y_current = lidar_current * np.cos((1- index_xy / 360.0) * math.pi).astype(np.float64)
    z_current = np.zeros_like(x_current)
    ones = np.ones_like(x_current)
    coordinates_current = np.stack([x_current, y_current, z_current, ones], axis=0)

    current_reference_x = current_x
    current_reference_y = current_y
    current_reference_yaw = current_yaw
    target_reference_x = target_x
    target_reference_y = target_y
    target_reference_yaw = target_yaw
    T_target_relative_to_world = np.array([[np.cos(target_reference_yaw), -np.sin(target_reference_yaw), 0, target_reference_x],
                                           [np.sin(target_reference_yaw),  np.cos(target_reference_yaw), 0, target_reference_y],
                                           [                             0,                               0, 1,                  0],
                                           [                             0,                               0, 0,                  1]],dtype=object)

    T_current_relative_to_world = np.array([[np.cos(current_reference_yaw), -np.sin(current_reference_yaw), 0, current_reference_x],
                                            [np.sin(current_reference_yaw),  np.cos(current_reference_yaw), 0, current_reference_y],
                                            [                              0,                                 0, 1,                   0],
                                            [                              0,                                 0, 0,                   1]],dtype=object)
    T_target_relative_to_world = T_target_relative_to_world.astype(np.float)
    #print(T_target_relative_to_world.dtype)
    T_world_relative_to_target = np.linalg.inv(T_target_relative_to_world)
    T_current_relative_to_target = T_world_relative_to_target.dot(T_current_relative_to_world)

    coordinates_target = T_current_relative_to_target.dot(coordinates_current)

    x_target = coordinates_target[0]
    y_target = coordinates_target[1]

    # 转到图像坐标系下
    image_x = np.floor(IMAGE_H - 1.0 - x_target / MAX_LASER_RANGE * IMAGE_H).astype(np.int)
    image_y = np.floor(IMAGE_H - 1.0 - y_target / MAX_LASER_RANGE * IMAGE_H).astype(np.int)

    image_x[(image_x < 0) | (image_x > (IMAGE_H - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0
    image_y[(image_y < 0) | (image_y > (IMAGE_W - 1)) | (lidar_current > (MAX_LASER_RANGE - 0.2))] = 0

    # TEST
    # image_test = np.zeros((IMAGE_H, IMAGE_W))
    # image_test[image_x, image_y] = 1

    # image_test[0, :] = 0
    # image_test[:, 0] = 0

    # cv2.imshow('image_test', image_test)
    # cv2.waitKey(10)


    return image_x, image_y

class Agent(object):
    def __init__(self, num, threshold_goal, threshold_collision, flag_of_social=False):

        # Info 
        self.num = num
        self.robot_name = 'robot_' + str(num)

        self.social = flag_of_social
        self.range = 0

        # Goal position 目标位置
        self.goal = [0.0, 0.0]
        self.local_goal = [0.0, 0.0]
        self.delta_local_goal = [0.0, 0.0]
        self.final_local_goal = [0.0, 0.0] # 要考虑yaw角的
        self.agent_position_1 = [0.0, 0.0]
        self.agent_position_2 = [0.0, 0.0]
        self.agent_position_3 = [0.0, 0.0]
        self.agent_position_4 = [0.0, 0.0]

        # Global Goal
        self.goal_pose = PoseStamped() 
        self.goal_pose.header.frame_id = "map"
        self.goal_pose.pose.position.x = self.goal[0]
        self.goal_pose.pose.position.y = self.goal[1]
        self.goal_pose.pose.position.z = 0.0
        self.goal_pose.pose.orientation.x = 0
        self.goal_pose.pose.orientation.y = 0
        self.goal_pose.pose.orientation.z = 0
        self.goal_pose.pose.orientation.w = 1

        # Local Goal
        self.local_goal_pose = PoseStamped() 
        self.local_goal_pose.header.frame_id = self.robot_name + "/base_footprint"
        self.local_goal_pose.pose.position.x = self.local_goal[0]
        self.local_goal_pose.pose.position.y = self.local_goal[1]
        self.local_goal_pose.pose.position.z = 0.0
        self.local_goal_pose.pose.orientation.x = 0
        self.local_goal_pose.pose.orientation.y = 0
        self.local_goal_pose.pose.orientation.z = 0
        self.local_goal_pose.pose.orientation.w = 1

        # State agent状态
        self.odom = Odometry()
        self.global_yaw = 0 # 全局yaw角，用于计算局部目标点位置


        self.observe = [] # 最新lidar输入
        self.observe_image = [] # 40帧雷达图像

        self.reward_distance = 0.0

        final_observation_multi_array = np.zeros((10,363))
        # final_observation_multi_array = np.zeros((1, 363))
        self.final_observation_multi = final_observation_multi_array.tolist()


        # 任务状态
        self.done_collision = False
        self.done_reached_goal = False
        self.final_goal_distance = 20.0
        self.last_final_goal_distance = 20.0
        self.initial_final_goal_distance = 20.0
        self.reward = 0
        self.success = False
        self.success_episodes = 0


        self.cumulated_steps = 0.0
        self.cumulated_steps_goal = 0.0
        self.cumulated_reward = 0.0

        self.flag_of_new_goal = 1

        self.save_time = 0
        self.time_lidar = 0
        self.time_odom = 0

        # 任务参数
        self.min_laser_value = 0.13 #0.21
        self.max_laser_value = MAX_LASER_RANGE
        self.min_range = threshold_collision
        self.reached_goal_threshold = threshold_goal

        
        # Publisher & Subscriber
        self.pub_cmd_vel = rospy.Publisher( '/cmd_vel', Twist, queue_size=10)
        self.pub_goal = rospy.Publisher('/goal', PoseStamped, queue_size=1)
        self.pub_local_goal = rospy.Publisher('/local_goal', PoseStamped, queue_size=10)
        
        rospy.Subscriber('/scan', LaserScan, self.callback_laser, queue_size=1)

        rospy.Subscriber('/odom', Odometry, self.callback_odometry_1, queue_size=1)



    # 重置机器人状态，主要是一些状态记录
    def reset(self):
        self.observe = []
        self.observe_2 = []
        self.observe_image = []
        self.reward_distance = 0.0

        final_observation_multi_array = np.zeros((1,363))
        self.final_observation_multi = final_observation_multi_array.tolist()


        self.goal = [0.0, 0.0]
        self.local_goal = [0.0, 0.0]
        self.delta_local_goal = [0.0, 0.0]
        self.final_local_goal = [0.0, 0.0]
        self.agent_position_1 = [0.0, 0.0]


        # Global Goal
        self.goal_pose = PoseStamped() 
        self.goal_pose.header.frame_id = "map"
        self.goal_pose.pose.position.x = self.goal[0]
        self.goal_pose.pose.position.y = self.goal[1]
        self.goal_pose.pose.position.z = 0.0
        self.goal_pose.pose.orientation.x = 0
        self.goal_pose.pose.orientation.y = 0
        self.goal_pose.pose.orientation.z = 0
        self.goal_pose.pose.orientation.w = 1

        # Local Goal
        self.local_goal_pose = PoseStamped() 
        self.local_goal_pose.header.frame_id = self.robot_name + "/base_footprint"
        self.local_goal_pose.pose.position.x = self.local_goal[0]
        self.local_goal_pose.pose.position.y = self.local_goal[1]
        self.local_goal_pose.pose.position.z = 0.0
        self.local_goal_pose.pose.orientation.x = 0
        self.local_goal_pose.pose.orientation.y = 0
        self.local_goal_pose.pose.orientation.z = 0
        self.local_goal_pose.pose.orientation.w = 1

        self.odom = Odometry()
        self.x = 0
        self.y = 0
        self.heading = 0
        self.x_2 = 0
        self.y_2 = 0
        self.heading_2 = 0

        self.done_collision = False
        self.done_reached_goal = False
        self.flag_of_new_goal = 1

        self.final_goal_distance = 20.0
        self.last_final_goal_distance = 20.0
        self.initial_final_goal_distance = 20.0


        self.cumulated_steps = 0.0
        self.cumulated_steps_goal = 0.0
        self.cumulated_reward = 0.0

        self.time_lidar = 0
        self.time_odom = 0

        self.reward = 0
        self.success = False


    # goal_pose 全局目标点保存
    def update_target(self, pose):
        # print("update_target  :   ", pose)
        self.goal = pose

        self.goal_pose.pose.position.x = pose[0]
        self.goal_pose.pose.position.y = pose[1]
        
        self.pub_goal.publish(self.goal_pose)

        self.done_reached_goal = False
        self.flag_of_new_goal = 1
    # laser callback 雷达回调函数: 获取雷达信息，得到 self.observe
    def callback_laser(self, data):
        
        t1 = time.time()
        self.observe = []
        self.reward_distance = self.max_laser_value * 1
        self.time_lidar += 1
        self.range = 0
        #print(len(data.ranges))

        for i in range(len(data.ranges)):

            # 数据获取（降采样）
            # if i % 2 == 0:

            if data.ranges[i] == float('Inf'):
                self.observe.append(self.max_laser_value)
            elif np.isnan(data.ranges[i]):
                self.observe.append(self.max_laser_value)
            else:
                self.observe.append(data.ranges[i])

            if (self.reward_distance > data.ranges[i] > 0):
                self.reward_distance = data.ranges[i]
                self.range = i

        t2 = time.time()
        # print("Lidar Time : {}  ms".format(round(1000*(t2-t1),2)))

    # odom callback 里程计回调函数
    def callback_odometry_1(self, odom):
        
        self.agent_position_1[0] = odom.pose.pose.position.x
        self.agent_position_1[1] = odom.pose.pose.position.y
        
        if self.num == 1:
            self.odom = odom
            orientation = odom.pose.pose.orientation
            temp_my_x, temp_my_y, temp_my_z, temp_my_w = orientation.x, orientation.y, orientation.z, orientation.w
            temp_atan2_y = 2.0 * (temp_my_w * temp_my_z + temp_my_x * temp_my_y)
            temp_atan2_x = 1.0 - 2.0 * (temp_my_y * temp_my_y + temp_my_z * temp_my_z)
            self.global_yaw = math.atan2(temp_atan2_y , temp_atan2_x)

    # 获取当前状态： lidar_image & local_goal & current_vel

    def get_state(self):
        """
        观测获取
        """
        # 目标完成判断
        # state ~ final_goal_distance
        self.success = False
        self.final_goal_distance = math.hypot(self.goal_pose.pose.position.x - self.odom.pose.pose.position.x, self.goal_pose.pose.position.y - self.odom.pose.pose.position.y)
        #print('final_goal_distance',self.final_goal_distance)

        if self.final_goal_distance < Cd: #self.reached_goal_threshold:threshold_goal=0.5

            self.done_reached_goal = True
            self.cumulated_steps_goal = 0


        if self.flag_of_new_goal == 1:
            self.flag_of_new_goal = 0
            self.last_final_goal_distance = self.final_goal_distance * 1
            self.initial_final_goal_distance = self.final_goal_distance * 1
            #print(self.flag_of_new_goal)

        laser_scan = self.observe[:] # 当前雷达
        #print('a322', len(laser_scan))
        # final_observation = [] # 完整state
        # final_observation = laser_scan
        #
        # obstacle_min_range = round(min(laser_scan), 2)
        # obstacle_angle = np.argmin(laser_scan)
        #print('final_observation', len(final_observation))

        # 碰撞判断
        # if self.odom.pose.pose.position.x > 3.78 or self.odom.pose.pose.position.x < -3.78 or self.odom.pose.pose.position.y > 3.78 or self.odom.pose.pose.position.y < -3.78:
        #     self.done_collision = True
        #     self.done = True

        if (self.min_range > self.reward_distance > 0):
            self.done_collision = True

        #print('done_collision',self.done_collision)

        # state ~ local_goal_direction
        self.local_goal = [self.goal_pose.pose.position.x, self.goal_pose.pose.position.y]
        self.local_goal_pose.pose.position.x = self.local_goal[0]
        self.local_goal_pose.pose.position.y = self.local_goal[1]

        self.delta_local_goal[0] = self.local_goal[0] - self.odom.pose.pose.position.x
        self.delta_local_goal[1] = self.local_goal[1] - self.odom.pose.pose.position.y
    
        distance_local_goal = math.sqrt((self.delta_local_goal[0])**2 + (self.delta_local_goal[1])**2) 
        theta_for_original_coordinate = math.atan2(self.delta_local_goal[1], self.delta_local_goal[0])
        
        theta_for_new_coordinate = theta_for_original_coordinate - self.global_yaw
        
        self.final_local_goal[0] = distance_local_goal * math.cos(theta_for_new_coordinate)
        self.final_local_goal[1] = distance_local_goal * math.sin(theta_for_new_coordinate)

        local_goal_normolize = math.sqrt((self.final_local_goal[0])**2 + (self.final_local_goal[1])**2)
        if local_goal_normolize >= 1:
            self.final_local_goal[0] = self.final_local_goal[0] / local_goal_normolize 
            self.final_local_goal[1] = self.final_local_goal[1] / local_goal_normolize 
        else:
            self.final_local_goal[0] = self.final_local_goal[0]
            self.final_local_goal[1] = self.final_local_goal[1]

        self.local_goal_pose.pose.position.x = self.final_local_goal[0]
        self.local_goal_pose.pose.position.y = self.final_local_goal[1]
        #print('local_goal_pose',self.local_goal_pose)

        self.pub_local_goal.publish(self.local_goal_pose)
        self.pub_goal.publish(self.goal_pose)

        # ego
        final_observation = laser_scan[:]

        linear_velocity_denoise = round(self.odom.twist.twist.linear.x * 2, 3)
        if linear_velocity_denoise < 0:
            linear_velocity_denoise = 0.0
        if linear_velocity_denoise > 1.0:
            linear_velocity_denoise = 1.0
        final_observation.append(linear_velocity_denoise)  # 0~1

        linear_angular_denoise = round(self.odom.twist.twist.angular.z, 3)
        if linear_angular_denoise < 0:
            linear_angular_denoise = 0.0
        if linear_angular_denoise > 1.0:
            linear_angular_denoise = 1.0
        final_observation.append(linear_angular_denoise) # -1.5 ~ 1.5 => 0~1

        goal_x = round(self.goal_pose.pose.position.x, 3)
        final_observation.append(goal_x)
        goal_y = round(self.goal_pose.pose.position.y, 3)
        final_observation.append(goal_y)

        final_observation.append(self.odom.pose.pose.position.x)
        final_observation.append(self.odom.pose.pose.position.y)

        final_observation.append(self.global_yaw)

        goal_distance = round(self.final_goal_distance / 25.0, 3)
        if goal_distance < 0:
            goal_distance = 0.0
        if goal_distance > 1.0:
            goal_distance = 1.0
        final_observation.append(goal_distance)  # 0 ~ 25 => 0~1

        goal_direction = round(((math.atan2(self.final_local_goal[1], self.final_local_goal[0]) / math.pi) + 1.) / 2, 3)
        if goal_direction < 0:
            goal_direction = 0.0
        if goal_direction > 1.0:
            goal_direction = 1.0
        final_observation.append(goal_direction)  # -pi ~ pi => 0~1

        obstacle_min_range = self.reward_distance #round(min(list(laser_scan)), 2)
        obstacle_angle = self.range
        final_observation.append(obstacle_min_range)
        final_observation.append(obstacle_angle)

        self.final_observation_multi.pop(0)
        self.final_observation_multi.append(final_observation[:])

        final_observation_all = np.asarray(self.final_observation_multi[:])
        observation_output = final_observation_all

        # print('obs',observation_output.shape)
        linear_speed = self.odom.twist.twist.linear.x
        angular_speed = self.odom.twist.twist.angular.z
        return observation_output,self.done_collision, self.done_reached_goal, linear_speed, angular_speed

    # 执行action
    def set_action(self, action):

        # # 执行决策指令
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x = (action[0] + 1) * 0.2 + 0.1 #np.clip(action[0], 0.1, 0.5)
        cmd_vel_value.angular.z = action[1]
        print('cmd_vel_value.linear.x',cmd_vel_value.linear.x)
        print('cmd_vel_value.angular.z', cmd_vel_value.angular.z)

        self.pub_cmd_vel.publish(cmd_vel_value)
        

    def _compute_reward(self):

        # reward parameter
        r_arrive = 100
        r_collision = -50
        Cr = 100   # judge arrival
        Cd = 0.5 #0.354 / 2  # = r when testing  # compute reward if no collision and arrival
        Cp = -0.05  # time step penalty   # time step penalty
        self.reward = 0
        reward_rotation = 0
        reward_velocity = 0
        reward_collision = 0

        distance_local_goal = math.sqrt((self.delta_local_goal[0]) ** 2 + (self.delta_local_goal[1]) ** 2)
        theta_for_original_coordinate = math.atan2(self.delta_local_goal[1], self.delta_local_goal[0])

        theta_for_new_coordinate = theta_for_original_coordinate - self.global_yaw
        d = self.final_goal_distance
        alpha = theta_for_new_coordinate

        if d < Cd:
            self.done_reached_goal = True
            self.reward = 100
            self.success = True
            self.success_episodes += 1
            print(self.success_episodes, "arrival!!!!!!!!!!!!!!")

        if not (self.done_reached_goal or self.done_collision):
            delta_d = self.last_final_goal_distance - self.final_goal_distance
            print('delta_d', delta_d)
            self.last_final_goal_distance = self.final_goal_distance * 1

            if abs(self.odom.twist.twist.angular.z) > 0.5:
                reward_rotation = -0.3 * abs(self.odom.twist.twist.angular.z) #0.1--0.3

                # VELOCITY REWARD
            if abs(self.odom.twist.twist.linear.x) < 0.3:
                reward_velocity = -0.1 * (0.5 - abs(self.odom.twist.twist.linear.x)) / 0.3

            # if self.reward_distance < 1:
            #     reward_collision = - 0.1 * (1 - (self.reward_distance - 0.25) / 1.25)
            reward = Cr * delta_d + Cp + reward_rotation + reward_velocity #+reward_collision
            self.reward += reward

        if self.done_collision:

            self.reward = r_collision
            print("collision!")
        print('reward',self.reward)
        self.cumulated_steps += 1
        rospy.logwarn("Cumulated_steps=" + str(self.cumulated_steps))

        return self.reward
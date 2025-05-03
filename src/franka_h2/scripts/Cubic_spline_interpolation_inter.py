#!/usr/bin/env python3

from __future__ import print_function 
import rospy 
import actionlib 
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # 新增标准消息

import math
from spatialmath.base import *
from spatialmath import SE3
import spatialmath.base.symbolic as sym
import numpy as np

import roboticstoolbox as rtb
from sensor_msgs.msg import JointState
from franka_h2.msg import TrajectoryData

# cal torque
from urdf_parser_py.urdf import URDF
import rospkg
import os

from fk import cal_fk
import time
from ik_geo import franka_IK_EE
from CSI import ManualCubicSpline

class CSI_complex_solver():
    def __init__(self):
        # 初始化标准轨迹动作客户端
        self.arm_client = actionlib.SimpleActionClient(
            'panda_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction
        )
        self.arm_client.wait_for_server()

        # 获取包路径
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('franka_h2')

        # 构建URDF文件路径
        self.urdf_path = os.path.join(pkg_path, 'urdf', 'panda_robot_gazebo.urdf')

        # 加载URDF并解析惯性参数
        self.robot = URDF.from_xml_file(self.urdf_path)
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.inertia_params = self._load_inertia_params()
        self.prev_velocities = None
        self.prev_time = None

        # 自定义轨迹的发布器
        self.traj_pub = rospy.Publisher('/trajectory_data', TrajectoryData, queue_size=10)
        # 标准轨迹的发布器（可选，根据需求选择动作或话题）
        self.std_traj_pub = rospy.Publisher('/joint_trajectory', JointTrajectory, queue_size=10)

        # 获取当前关节状态
        self.current_joint_positions = None
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        rospy.wait_for_message('/joint_states', JointState)

    def joint_state_callback(self, msg):
        self.current_joint_positions = msg.position
    
    def _load_inertia_params(self):
        """从URDF中提取完整的惯性参数（质量、转动惯量矩阵、质心位置）"""
        inertia_params = {}
        for name in self.joint_names:
            for joint in self.robot.joints:
                if joint.name == name:
                    child_link = self.robot.link_map[joint.child]
                    if child_link.inertial:
                        # 提取质量
                        mass = child_link.inertial.mass
                        
                        # 提取完整转动惯量矩阵
                        inertia = child_link.inertial.inertia
                        inertia_matrix = [
                            [inertia.ixx, inertia.ixy, inertia.ixz],
                            [inertia.ixy, inertia.iyy, inertia.iyz],
                            [inertia.ixz, inertia.iyz, inertia.izz]
                        ]
                        
                        # 提取质心位置（相对于关节坐标系）
                        origin = child_link.inertial.origin
                        com_position = origin.xyz if origin else [0,0,0]
                        
                        inertia_params[name] = {
                            'mass': mass,
                            'inertia_matrix': inertia_matrix,
                            'com_position': com_position
                        }
                    break
        return inertia_params
    
    # def generate_trajectory(self, q_waypoints, t_waypoints, freq=50):
    #     """生成多中间点三次样条轨迹"""
        
    #     # 时间参数设置
    #     duration = t_waypoints[-1]
    #     num_points = int(duration * freq)
    #     t = np.linspace(0, duration, num_points)

    #     # 计算各关节参数
    #     positions = []
    #     velocities = []
    #     accelerations = []

    #     # 为每个关节创建样条函数
    #     n_joints = len(q_waypoints[0])
    #     splines = []
    #     for j in range(n_joints):
    #         # 提取当前关节的路径点
    #         q = [point[j] for point in q_waypoints]
    #         # 创建三次样条插值器
    #         from scipy.interpolate import CubicSpline
    #         cs = CubicSpline(t_waypoints, q, bc_type='natural')
    #         splines.append(cs)

    #     # 生成轨迹点
    #     for ti in t:
    #         point = JointTrajectoryPoint()
    #         pos = []
    #         vel = []
    #         acc = []
    #         for j in range(n_joints):
    #             # 计算各关节参数
    #             cs = splines[j]
    #             pos.append(cs(ti))
    #             vel.append(cs(ti, 1))  # 一阶导数
    #             acc.append(cs(ti, 2))  # 二阶导数
            
    #         positions.append(pos)
    #         velocities.append(vel)
    #         accelerations.append(acc)

    #     return positions, velocities, t
    
    
    def calculate_gravity_torques(self):
        """计算各关节重力矩（返回字典形式）"""
        g = 9.81  # 重力加速度
        gravity_torques = {}  # 使用字典存储结果
        
        for joint_name in self.joint_names:
            # 获取关节参数
            params = self.inertia_params.get(joint_name)
            if not params:
                rospy.logwarn(f"未找到关节 {joint_name} 的惯性参数")
                continue
                
            # 解包参数
            mass = params['mass']
            com = np.array(params['com_position'])
            
            # 基坐标系重力矢量（可根据实际坐标系方向调整）
            base_gravity = np.array([0, 0, -g])  
            
            # 简化的坐标系转换（假设关节坐标系与基坐标系对齐）
            joint_gravity = base_gravity  # 实际需用变换矩阵转换
            
            # 计算力矩 τ = r × F
            torque_vector = np.cross(com, mass * joint_gravity)
            
            # 存储Z轴分量（假设关节绕Z轴旋转）
            gravity_torques[joint_name] = torque_vector[2]  # 单位：Nm
            
        return gravity_torques

    def move_to_goal(self, goal_positions, duration=10.0):
        if self.current_joint_positions is None:
            rospy.logerr("未获取到当前关节状态！")
            return
        
        duration_single = duration / 4

        num_points = int(duration* 50)
        timestamps = np.linspace(0, duration, num_points)

        gravity_torques = self.calculate_gravity_torques()

        g = 9.81

        # 存储插值数据
        all_positions = []
        all_velocities = []

        t_waypoints = np.linspace(0, duration, 5)
        
        for joint_idx in range(len(goal_positions)):
            # 为不同关节设置不同的中间轨迹点
            if joint_idx == 0:
                q_waypoints = []
                start = self.current_joint_positions[joint_idx]
                goal = goal_positions[joint_idx]
                q_waypoints.append(start)
                q_waypoints.append(((goal-start)/5+start))
                q_waypoints.append(((goal-start)/4+start))
                q_waypoints.append(((goal-start)/3+start))
                q_waypoints.append(goal)
                print(q_waypoints)
                spline1 = ManualCubicSpline(t_waypoints, q_waypoints)
                t_samples = np.linspace(t_waypoints[0], t_waypoints[-1], num_points)
                positions = spline1.evaluate(t_samples, 0)
                velocities = spline1.evaluate(t_samples, 1)
                all_positions.append(positions)
                all_velocities.append(velocities)
            elif joint_idx == 1:
                q_waypoints = []
                start = self.current_joint_positions[joint_idx]
                goal = goal_positions[joint_idx]
                q_waypoints.append(start)
                q_waypoints.append(((goal-start)/2+start))
                q_waypoints.append(((goal-start)/1.5+start))
                q_waypoints.append(((goal-start)/1.2+start))
                q_waypoints.append(goal)
                print(q_waypoints)
                spline2 = ManualCubicSpline(t_waypoints, q_waypoints)
                t_samples = np.linspace(t_waypoints[0], t_waypoints[-1], num_points)
                positions = spline2.evaluate(t_samples, 0)
                velocities = spline2.evaluate(t_samples, 1)
                all_positions.append(positions)
                all_velocities.append(velocities)
            elif joint_idx == 2:
                q_waypoints = []
                start = self.current_joint_positions[joint_idx]
                goal = goal_positions[joint_idx]
                q_waypoints.append(start)
                q_waypoints.append(((goal-start)/3+start))
                q_waypoints.append(((goal-start)/2.5+start))
                q_waypoints.append(((goal-start)/1.5+start))
                q_waypoints.append(goal)
                print(q_waypoints)
                spline3 = ManualCubicSpline(t_waypoints, q_waypoints)
                t_samples = np.linspace(t_waypoints[0], t_waypoints[-1], num_points)
                positions = spline3.evaluate(t_samples, 0)
                velocities = spline3.evaluate(t_samples, 1)
                all_positions.append(positions)
                all_velocities.append(velocities)
            elif joint_idx == 3:
                q_waypoints = []
                start = self.current_joint_positions[joint_idx]
                goal = goal_positions[joint_idx]
                q_waypoints.append(start)
                q_waypoints.append(((goal-start)/2+start))
                q_waypoints.append(((goal-start)/1.5+start))
                q_waypoints.append(((goal-start)/1.2+start))
                q_waypoints.append(goal)
                print(q_waypoints)
                spline4 = ManualCubicSpline(t_waypoints, q_waypoints)
                t_samples = np.linspace(t_waypoints[0], t_waypoints[-1], num_points)
                positions = spline4.evaluate(t_samples, 0)
                velocities = spline4.evaluate(t_samples, 1)
                all_positions.append(positions)
                all_velocities.append(velocities)
            elif joint_idx == 4:
                q_waypoints = []
                start = self.current_joint_positions[joint_idx]
                goal = goal_positions[joint_idx]
                q_waypoints.append(start)
                q_waypoints.append(((goal-start)/4+start))
                q_waypoints.append(((goal-start)/3+start))
                q_waypoints.append(((goal-start)/1.2+start))
                q_waypoints.append(goal)
                print(q_waypoints)
                spline5 = ManualCubicSpline(t_waypoints, q_waypoints)
                t_samples = np.linspace(t_waypoints[0], t_waypoints[-1], num_points)
                positions = spline5.evaluate(t_samples, 0)
                velocities = spline5.evaluate(t_samples, 1)
                all_positions.append(positions)
                all_velocities.append(velocities)
            elif joint_idx == 5:
                q_waypoints = []
                start = self.current_joint_positions[joint_idx]
                goal = goal_positions[joint_idx]
                q_waypoints.append(start)
                q_waypoints.append(((goal-start)/3+start))
                q_waypoints.append(((goal-start)/2+start))
                q_waypoints.append(((goal-start)/1.5+start))
                q_waypoints.append(goal)
                print(q_waypoints)
                spline6 = ManualCubicSpline(t_waypoints, q_waypoints)
                t_samples = np.linspace(t_waypoints[0], t_waypoints[-1], num_points)
                positions = spline6.evaluate(t_samples, 0)
                velocities = spline6.evaluate(t_samples, 1)
                all_positions.append(positions)
                all_velocities.append(velocities)
            elif joint_idx == 6:
                q_waypoints = []
                start = self.current_joint_positions[joint_idx]
                goal = goal_positions[joint_idx]
                q_waypoints.append(start)
                q_waypoints.append(((goal-start)/2+start))
                q_waypoints.append(((goal-start)/1.5+start))
                q_waypoints.append(((goal-start)/1.2+start))
                q_waypoints.append(goal)
                print(q_waypoints)
                spline7 = ManualCubicSpline(t_waypoints, q_waypoints)
                t_samples = np.linspace(t_waypoints[0], t_waypoints[-1], num_points)
                positions = spline7.evaluate(t_samples, 0)
                velocities = spline7.evaluate(t_samples, 1)
                all_positions.append(positions)
                all_velocities.append(velocities)

        # ================== 新增：构建标准JointTrajectory ==================
        std_traj_msg = JointTrajectory()
        std_traj_msg.joint_names = self.joint_names
        for i in range(num_points):
            point = JointTrajectoryPoint()
            point.positions = [all_positions[j][i] for j in range(len(self.joint_names))]
            point.velocities = [all_velocities[j][i] for j in range(len(self.joint_names))]
            point.time_from_start = rospy.Duration(timestamps[i])
            std_traj_msg.points.append(point)

        # 方式一：通过Action发送给控制器
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = std_traj_msg
        self.arm_client.send_goal(goal)

        # 方式二：直接发布到话题（如果控制器订阅话题）
        # self.std_traj_pub.publish(std_traj_msg)

        # ================== 原有的自定义轨迹发布 ==================
        for i in range(num_points):
            target_msg = TrajectoryData()
            target_msg.joint_names = self.joint_names
            target_msg.positions = [all_positions[j][i] for j in range(len(self.joint_names))]
            current_velocities = [all_velocities[j][i] for j in range(len(self.joint_names))]
            target_msg.velocities = current_velocities

            # 计算角加速度
            current_time = timestamps[i]
            if self.prev_velocities and i > 0:
                dt = current_time - self.prev_time
                accelerations = [(v - pv)/dt for v, pv in zip(current_velocities, self.prev_velocities)]
            else:
                accelerations = [0.0]*len(self.joint_names)
            
            target_msg.acc = accelerations  # 直接赋值计算结果

            

            # 力矩计算部分
            for name in self.joint_names:  # 按joint_names顺序遍历
                # 获取当前关节参数
                params = self.inertia_params[name]
                
                # 获取对应的加速度（按索引匹配顺序）
                j = self.joint_names.index(name)
                acc = accelerations[j]
                
                # 获取当前关节重力矩
                grav = gravity_torques[name]
                
                # 计算惯性力矩 + 重力补偿
                inertia_torque = np.dot(params['inertia_matrix'], [0, 0, acc])[2]  # 取绕Z轴分量
                total_torque = inertia_torque + grav
                
                target_msg.torques.append(total_torque)

            # 计算力矩（τ = Iα 的简化模型）
            # target_msg.torques = [self.inertia_params[name] * acc for name, acc 
            #                     in zip(self.joint_names, accelerations)]
            
            

            # 更新时间记录
            self.prev_velocities = current_velocities
            self.prev_time = current_time
            
            # 发布时间戳和消息
            target_msg.stamp = rospy.Time.now() + rospy.Duration(current_time)
            self.traj_pub.publish(target_msg)
            rospy.sleep(1.0 / 50)

        rospy.loginfo("运动完成！")

panda = rtb.models.URDF.Panda()
# print(panda)

T = panda.fkine(panda.qz, end='panda_hand')
#print(T)

# x = float(input("Enter X Co-ordinate: "))
# y = float(input("Enter Y Co-ordinate: "))
# z = float(input("Enter Z Co-ordinate: "))

# point = SE3(x,y,z)
# point_sol = panda.ikine_LM(point)
# print("IK Solution: ",point_sol.q)

# 定义目标位置和旋转
x = 0.5
y = 0.2
z = 0.25
R_custom = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
# R_custom = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ])

# 创建目标位姿
# T_goal = SE3.Rt(R_custom, [x, y, z])

T_goal = cal_fk([2.22601256 , 1.2024973 , -2.72513796 ,-2.21730977, -2.19911507 , 0.1795953,  0])
# print(T_goal)

# 逆运动学解算  
# point_sol = panda.ikine_LM(T_goal)
# print("IK Solution: ", point_sol.q)
# T_actual = panda.fkine(point_sol.q)
# print(T_actual)

# Test current joint angles
q_current = np.zeros(7)

# Test q7 value
q7_test = 0.0

print("Target position:")
print(T_goal)
start_time = time.perf_counter()
solutions = franka_IK_EE(T_goal, q7_test, q_current)
print("ik Solution :", solutions[0])
T_actual = cal_fk(solutions[0])
print("Actual position:")
print(T_actual)

# 记录结束时间
end_time = time.perf_counter()
# 计算并存储耗时（转换为毫秒）
elapsed_ms = (end_time - start_time) * 1000
print(f"耗时: {elapsed_ms:.4f} ms")


if __name__ == '__main__':

  try:

    rospy.init_node('arm_interpolation_controller')

    solve_ik = CSI_complex_solver()
    #move_robot_arm([point_sol[0] , point_sol[1] , point_sol[2] , point_sol[3] , point_sol[4]])
    # solve_ik.move_robot_arm(point_sol.q)

    print(solutions[0])

    solve_ik.move_to_goal(solutions[0], duration=10.0)
    
    print("Robotic arm has successfully reached the goal!")
     
  except rospy.ROSInterruptException:
    print("Program interrupted before completion.", file=sym.stderr)














































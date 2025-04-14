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

class LM_ik_solver():
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
        """从URDF中提取关节对应的惯性矩"""
        inertia_dict = {}
        for name in self.joint_names:
            for joint in self.robot.joints:
                if joint.name == name:
                    child_link = self.robot.link_map[joint.child]
                    if child_link.inertial:
                        # 获取惯性矩阵（这里简化为绕关节轴的转动惯量）
                        inertia = child_link.inertial.inertia
                        # 假设关节绕Z轴旋转，取izz分量
                        inertia_dict[name] = inertia.izz
                    break
        return inertia_dict

    def quintic_interpolation(self, start, goal, duration, freq=50):
        """五次多项式插值"""
        num_points = int(duration * freq)
        t = np.linspace(0, duration, num_points)
        positions = []
        velocities = []
        
        for ti in t:
            # 五次多项式系数计算（此处简化）
            pos = start + (goal - start) * (10*(ti/duration)**3 - 15*(ti/duration)**4 + 6*(ti/duration)**5)
            vel = (goal - start) * (30*(ti/duration)**2 - 60*(ti/duration)**3 + 30*(ti/duration)**4) / duration
            positions.append(pos)
            velocities.append(vel)
        
        return positions, velocities, t
    
    def trapezoidal_interpolation(self, start, goal, duration, freq=50):
        """梯形速度曲线插值（三段式）"""
        num_points = int(duration * freq)
        t = np.linspace(0, duration, num_points)
        positions = []
        velocities = []
        
        delta = goal - start  # 总位移
        t_acc = t_dec = duration / 3.0  # 加速/减速段时间（各占1/3）
        t_const = duration - t_acc - t_dec  # 匀速段时间
        
        # 计算运动参数
        if abs(duration) > 1e-6:  # 防止除以0
            a = (9.0 * delta) / (2.0 * duration**2)  # 加速度计算
            v_max = a * t_acc  # 最大速度
        else:
            a = 0.0
            v_max = 0.0
        
        for ti in t:
            if ti < t_acc:  # 加速段
                phase_pos = 0.5 * a * ti**2
                phase_vel = a * ti
            elif ti < t_acc + t_const:  # 匀速段
                phase_pos = 0.5 * a * t_acc**2 + v_max * (ti - t_acc)
                phase_vel = v_max
            else:  # 减速段
                t_dec_phase = ti - (t_acc + t_const)
                phase_pos = (0.5 * a * t_acc**2 + 
                            v_max * t_const + 
                            v_max * t_dec_phase - 
                            0.5 * a * t_dec_phase**2)
                phase_vel = v_max - a * t_dec_phase
            
            # 计算绝对位置和速度
            positions.append(start + phase_pos)
            velocities.append(phase_vel)
        
        return positions, velocities, t

    def move_to_goal(self, goal_positions, duration=5.0):
        if self.current_joint_positions is None:
            rospy.logerr("未获取到当前关节状态！")
            return

        num_points = int(duration * 50)
        timestamps = np.linspace(0, duration, num_points)

        # 存储插值数据
        all_positions = []
        all_velocities = []
        for joint_idx in range(len(goal_positions)):
            start = self.current_joint_positions[joint_idx]
            goal = goal_positions[joint_idx]
            positions, velocities, _ = self.quintic_interpolation(start, goal, duration, 50)
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
            
            # 计算力矩（τ = Iα 的简化模型）
            target_msg.torques = [self.inertia_params[name] * acc for name, acc 
                                in zip(self.joint_names, accelerations)]

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
z = 0.5
R_custom = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
# R_custom = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ])

# 创建目标位姿
T_goal = SE3.Rt(R_custom, [x, y, z])

# 逆运动学解算  
point_sol = panda.ikine_LM(T_goal)
print("IK Solution: ", point_sol.q)


if __name__ == '__main__':

  try:

    rospy.init_node('arm_interpolation_controller')

    solve_ik = LM_ik_solver()
    #move_robot_arm([point_sol[0] , point_sol[1] , point_sol[2] , point_sol[3] , point_sol[4]])
    # solve_ik.move_robot_arm(point_sol.q)

    solve_ik.move_to_goal(point_sol.q, duration=5.0)
    
    print("Robotic arm has successfully reached the goal!")
     
  except rospy.ROSInterruptException:
    print("Program interrupted before completion.", file=sym.stderr)


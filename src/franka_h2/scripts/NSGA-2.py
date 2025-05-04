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
import random
from collections import defaultdict

class CSI_solver():
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
    
    def generate_trajectory(self, q_start, q_end, duration, freq = 50):
        """生成三次样条轨迹"""
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        
        # 时间参数设置
        # sample_rate = freq  # Hz
        # dt = 1.0/sample_rate
        num_points = int(duration * freq)
        t = np.linspace(0, duration, num_points)

        # 计算各关节参数
        positions = []
        velocities = []
        accelerations = []
    
        for ti in t:
            # t_p = t[i]
            point = JointTrajectoryPoint()
            
            
            # for j in range(7):
            # 三次样条计算
            a0 = q_start
            a1 = 0.0
            a2 = 3*(q_end-q_start)/(duration**2)
            a3 = -2*(q_end-q_start)/(duration**3)
            
            pos = a0 + a1*ti + a2*ti**2 + a3*ti**3
            vel = a1 + 2*a2*ti + 3*a3*ti**2
            acc = 2*a2 + 6*a3*ti
            
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
            
        return positions, velocities, t
    
    def generate_optimized_trajectory(self, q_start, q_end, duration, freq=50):
        """使用NSGA-II优化生成轨迹"""
        # NSGA-II参数设置
        population_size = 50
        generations = 100
        crossover_prob = 0.9
        mutation_prob = 0.1
        variable_ranges = [(-0.5, 0.5)] * 7  # 各关节初始速度范围（需根据实际情况调整）

        # 目标函数计算
        def evaluate(individual):
            """计算单个个体的双目标适应度"""
            total_smoothness = 0.0
            total_energy = 0.0
            T = duration

            for j in range(7):
                # 获取关节参数
                joint_name = self.joint_names[j]
                params = self.inertia_params[joint_name]
                I_j = params['inertia_matrix'][2][2]  # 绕Z轴的转动惯量
                g_j = self.calculate_gravity_torques()[joint_name]

                # 轨迹参数计算
                delta_q = q_end[j] - q_start[j]
                v_start = individual[j]
                
                # 三次多项式系数计算
                a2 = (3*delta_q - 2*v_start*T) / T**2
                a3 = (-2*(delta_q - v_start*T/2)) / T**3

                # 计算平滑性指标（加速度平方积分）
                integral_acc_sq = 4*a2**2*T + 12*a2*a3*T**2 + 12*a3**2*T**3
                total_smoothness += integral_acc_sq

                # 计算能耗指标（力矩平方积分）
                integral_tau_sq = (I_j**2 * integral_acc_sq) + (2*I_j*g_j*(-v_start)) + (g_j**2 * T)
                total_energy += integral_tau_sq

            return (total_smoothness, total_energy)

        # 遗传算子
        def crossover(p1, p2):
            """模拟二进制交叉"""
            child1, child2 = [], []
            for i in range(7):
                beta = np.random.uniform(0.9, 1.1)
                child1.append(0.5*((1+beta)*p1[i] + (1-beta)*p2[i]))
                child2.append(0.5*((1-beta)*p1[i] + (1+beta)*p2[i]))
            return child1, child2

        def mutate(ind):
            """多项式变异"""
            mutated = ind.copy()
            for i in range(7):
                if random.random() < mutation_prob:
                    mutated[i] += np.random.normal(0, 0.1)
                    mutated[i] = np.clip(mutated[i], *variable_ranges[i])
            return mutated

        # 初始化种群
        population = [[np.random.uniform(variable_ranges[j][0], 
                          variable_ranges[j][1]) 
         for j in range(7)]
        for _ in range(population_size)]

        # 进化循环
        for _ in range(generations):
            # 评估适应度
            fitness = [evaluate(ind) for ind in population]

            # 非支配排序
            fronts = []
            dominated_count = defaultdict(int)
            dominated_set = defaultdict(list)
            ranks = {}
            
            # 构建支配关系
            for i, f1 in enumerate(fitness):
                for j, f2 in enumerate(fitness):
                    if f1[0] < f2[0] and f1[1] < f2[1]:
                        dominated_set[i].append(j)
                        dominated_count[j] += 1

            # 构建前沿
            current_front = [i for i in range(population_size) if dominated_count[i] == 0]
            fronts.append(current_front)
            
            # 后续前沿
            next_front = []
            while current_front:
                for i in current_front:
                    for j in dominated_set[i]:
                        dominated_count[j] -= 1
                        if dominated_count[j] == 0:
                            next_front.append(j)
                fronts.append(next_front)
                current_front = next_front
                next_front = []

            # 选择操作（简单截断选择）
            selected = []
            for front in fronts:
                if len(selected) + len(front) <= population_size:
                    selected.extend(front)
                else:
                    remaining = population_size - len(selected)
                    selected += front[:remaining]
                    break

            # 生成新种群
            new_population = [population[i] for i in selected]
            while len(new_population) < population_size:
                # 选择父代
                p1, p2 = random.choices(selected, k=2)
                # 交叉
                if random.random() < crossover_prob:
                    c1, c2 = crossover(population[p1], population[p2])
                    new_population.extend([c1, c2])
                # 变异
                else:
                    new_population.append(mutate(population[p1]))
                    new_population.append(mutate(population[p2]))
            population = new_population[:population_size]

        # 选择最优解（取第一个前沿的第一个解）
        best_ind = population[0]

        # 生成轨迹数据
        num_points = int(duration * freq)
        t = np.linspace(0, duration, num_points)
        positions = []
        velocities = []
        
        for j in range(7):
            v_start = best_ind[j]
            delta_q = q_end[j] - q_start[j]
            a2 = (3*delta_q - 2*v_start*duration) / duration**2
            a3 = (-2*(delta_q - v_start*duration/2)) / duration**3
            
            joint_pos = []
            joint_vel = []
            for ti in t:
                pos = q_start[j] + v_start*ti + a2*ti**2 + a3*ti**3
                vel = v_start + 2*a2*ti + 3*a3*ti**2
                joint_pos.append(pos)
                joint_vel.append(vel)
            positions.append(joint_pos)
            velocities.append(joint_vel)

        # 转置为时间主维度
        return np.array(positions).T.tolist(), np.array(velocities).T.tolist(), t
    
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

        num_points = int(duration * 50)
        timestamps = np.linspace(0, duration, num_points)

        gravity_torques = self.calculate_gravity_torques()

        g = 9.81

        # 存储插值数据
        all_positions = []
        all_velocities = []
        # for joint_idx in range(len(goal_positions)):
        #     start = self.current_joint_positions[joint_idx]
        #     goal = goal_positions[joint_idx]
        #     positions, velocities, _ = self.generate_optimized_trajectory(start, goal, duration, 50)
        #     all_positions.append(positions)
        #     all_velocities.append(velocities)
            # trajectory = self.generate_trajectory(start, goal, duration)
        all_positions, all_velocities, t = self.generate_optimized_trajectory(self.current_joint_positions, goal_positions, duration, 50)
        # all_positions.append(positions)
        # all_velocities.append(velocities)

        # ================== 新增：构建标准JointTrajectory ==================
        std_traj_msg = JointTrajectory()
        std_traj_msg.joint_names = self.joint_names
        print(len(all_positions))
        print(len(all_velocities))
        # for i in range(num_points):
        #     point = JointTrajectoryPoint()
        #     point.positions = [all_positions[j][i] for j in range(len(self.joint_names))]
        #     point.velocities = [all_velocities[j][i] for j in range(len(self.joint_names))]
        #     point.time_from_start = rospy.Duration(timestamps[i])
        #     std_traj_msg.points.append(point)
        for i in range(len(t)):
            point = JointTrajectoryPoint()
            point.positions = [all_positions[i][j] for j in range(7)]
            point.velocities = [all_velocities[i][j] for j in range(7)]
            point.time_from_start = rospy.Duration(t[i])
            std_traj_msg.points.append(point)

        # 方式一：通过Action发送给控制器
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = std_traj_msg
        self.arm_client.send_goal(goal)
        self.arm_client.wait_for_result()

        rospy.loginfo("优化轨迹执行完成！")

        # 方式二：直接发布到话题（如果控制器订阅话题）
        # self.std_traj_pub.publish(std_traj_msg)

        # ================== 原有的自定义轨迹发布 ==================
        for i in range(len(t)):
            target_msg = TrajectoryData()
            target_msg.joint_names = self.joint_names
            target_msg.positions = [all_positions[i][j] for j in range(7)]
            current_velocities = [all_velocities[i][j] for j in range(7)]
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

# 定义目标位置和旋转
x = 0.5
y = 0.2
z = 0.25
R_custom = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

T_goal = cal_fk([2.22601256 , 1.2024973 , -2.72513796 ,-2.21730977, -2.19911507 , 0.1795953,  0])
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

    solve_ik = CSI_solver()
    #move_robot_arm([point_sol[0] , point_sol[1] , point_sol[2] , point_sol[3] , point_sol[4]])
    # solve_ik.move_robot_arm(point_sol.q)

    print(solutions[0])

    solve_ik.move_to_goal(solutions[0], duration=10.0)
    
    print("Robotic arm has successfully reached the goal!")
     
  except rospy.ROSInterruptException:
    print("Program interrupted before completion.", file=sym.stderr)



#!/usr/bin/env python3

from __future__ import print_function 
import rospy 
import actionlib 
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal 
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np
from sensor_msgs.msg import JointState
from franka_h2.msg import TrajectoryData
import random
from scipy.spatial import KDTree

from fk import cal_fk
import time
from ik_geo import franka_IK_EE 

from urdf_parser_py.urdf import URDF
import rospkg
import os

class BiRRTPlanner:
    def __init__(self):
        # 初始化与原CSI_solver相同的ROS接口
        self.arm_client = actionlib.SimpleActionClient(
            'panda_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction
        )
        self.arm_client.wait_for_server()
        
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 
                           'joint5', 'joint6', 'joint7']
        
        # RRT参数
        self.max_iterations = 10000
        self.step_size = 0.1  # 步长
        self.goal_bias = 0.1  # 朝目标采样的概率
        self.connection_radius = 0.5  # 连接半径
        
        # 关节限位（与原代码相同）
        self.joint_limits = [
            [-2.8973, 2.8973],  # joint1
            [-1.7628, 1.7628],  # joint2
            [-2.8973, 2.8973],  # joint3
            [-3.0718, -0.0698], # joint4
            [-2.8973, 2.8973],  # joint5
            [-0.0175, 3.7525],  # joint6
            [-2.8973, 2.8973]   # joint7
        ]
        
        # 轨迹发布器
        self.traj_pub = rospy.Publisher('/trajectory_data', TrajectoryData, queue_size=10)
        
        # 获取当前关节状态
        self.current_joint_positions = None
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        rospy.wait_for_message('/joint_states', JointState)

        # 自定义轨迹的发布器
        self.traj_pub = rospy.Publisher('/trajectory_data', TrajectoryData, queue_size=10)

        # 获取包路径
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('franka_h2')

        # 构建URDF文件路径
        self.urdf_path = os.path.join(pkg_path, 'urdf', 'panda_robot_gazebo.urdf')

        # 加载URDF并解析惯性参数
        self.robot = URDF.from_xml_file(self.urdf_path)

        self.inertia_params = self._load_inertia_params()
    
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
        
    def joint_state_callback(self, msg):
        self.current_joint_positions = msg.position[:7]
    
    def random_config(self):
        """生成随机构型"""
        config = []
        for limit in self.joint_limits:
            config.append(random.uniform(limit[0], limit[1]))
        return np.array(config)
    
    def distance(self, q1, q2):
        """计算两个构型之间的距离"""
        return np.linalg.norm(np.array(q1) - np.array(q2))
    
    def interpolate(self, q1, q2, step_size):
        """在两个构型之间插值"""
        dist = self.distance(q1, q2)
        if dist < step_size:
            return q2
        else:
            ratio = step_size / dist
            return q1 + ratio * (q2 - q1)
    
    def is_valid_config(self, q):
        """检查构型是否有效（在关节限位内）"""
        for i, (joint_pos, limit) in enumerate(zip(q, self.joint_limits)):
            if joint_pos < limit[0] or joint_pos > limit[1]:
                return False
        return True
    
    def build_rrt_trees(self, start, goal):
        """构建双向RRT树"""
        start_tree = [{'config': start, 'parent': None}]
        goal_tree = [{'config': goal, 'parent': None}]
        
        for _ in range(self.max_iterations):
            # 随机采样或朝目标采样
            if random.random() < self.goal_bias:
                rand_config = goal if len(start_tree) % 2 == 0 else start
            else:
                rand_config = self.random_config()
            
            # 扩展两棵树
            if len(start_tree) % 2 == 0:
                new_node = self.extend_tree(start_tree, rand_config)
                if new_node is not None:
                    # 尝试连接到目标树
                    success, path = self.try_connect(new_node, goal_tree)
                    if success:
                        return self.extract_path(start_tree, new_node, goal_tree, path)
            else:
                new_node = self.extend_tree(goal_tree, rand_config)
                if new_node is not None:
                    # 尝试连接到起始树
                    success, path = self.try_connect(new_node, start_tree)
                    if success:
                        path = self.extract_path(start_tree, path, goal_tree, new_node)
                        return path[::-1]  # 反转路径
        
        return None
    
    def extend_tree(self, tree, target_config):
        """扩展RRT树"""
        # 找到最近节点
        nearest_idx = min(range(len(tree)), 
                         key=lambda i: self.distance(tree[i]['config'], target_config))
        nearest_node = tree[nearest_idx]
        
        # 朝目标方向扩展
        new_config = self.interpolate(nearest_node['config'], target_config, self.step_size)
        
        # 检查新构型是否有效
        if self.is_valid_config(new_config):
            new_node = {'config': new_config, 'parent': nearest_idx}
            tree.append(new_node)
            return new_node
        return None
    
    def try_connect(self, node, tree):
        """尝试连接到另一棵树"""
        nearest_idx = min(range(len(tree)), 
                        key=lambda i: self.distance(tree[i]['config'], node['config']))
        
        if self.distance(node['config'], tree[nearest_idx]['config']) < self.connection_radius:
            return True, tree[nearest_idx]  # 返回节点对象而不是索引
        return False, None

    def extract_path(self, start_tree, start_node, goal_tree, goal_node):
        """提取路径"""
        path = []
        
        # 从起始树提取路径
        current = start_node
        while current is not None:
            path.append(current['config'])
            if current['parent'] is None:
                break
            current = start_tree[current['parent']]
        path = path[::-1]  # 反转路径
        
        # 从目标树提取路径
        current = goal_node  # goal_node现在是节点对象而不是索引
        while current is not None:
            path.append(current['config'])
            if current['parent'] is None:
                break
            current = goal_tree[current['parent']]
        
        return path
    
    def smooth_path(self, path, num_iterations=100):
        """路径平滑"""
        if len(path) <= 3:  # 如果路径点太少，直接返回原路径
            return path
        
        smoothed_path = list(path)  # 创建路径的副本
        
        for _ in range(num_iterations):
            if len(smoothed_path) <= 3:  # 如果平滑后路径太短，停止平滑
                break
                
            try:
                i = random.randint(0, len(smoothed_path)-3)
                j = random.randint(i+2, len(smoothed_path)-1)
                
                # 检查直接连接是否可行
                direct_path = np.linspace(smoothed_path[i], smoothed_path[j], num=10)
                if all(self.is_valid_config(q) for q in direct_path):
                    smoothed_path = smoothed_path[:i+1] + smoothed_path[j:]
            except ValueError:
                break  # 如果出现范围错误，退出平滑过程
        
        return smoothed_path
    
    def generate_trajectory(self, q_start, q_end, duration, freq=50):
        """生成五次多项式轨迹"""
        num_points = int(duration * freq)
        t = np.linspace(0, duration, num_points)
        
        positions = []
        velocities = []
        accelerations = []
        
        for ti in t:
            # 五次多项式系数
            a0 = q_start
            a1 = 0
            a2 = 0
            a3 = 10 * (q_end - q_start) / duration**3
            a4 = -15 * (q_end - q_start) / duration**4
            a5 = 6 * (q_end - q_start) / duration**5
            
            # 计算位置、速度和加速度
            pos = a0 + a1*ti + a2*ti**2 + a3*ti**3 + a4*ti**4 + a5*ti**5
            vel = a1 + 2*a2*ti + 3*a3*ti**2 + 4*a4*ti**3 + 5*a5*ti**4
            acc = 2*a2 + 6*a3*ti + 12*a4*ti**2 + 20*a5*ti**3
            
            positions.append(pos)
            velocities.append(vel)
            accelerations.append(acc)
        
        return positions, velocities, accelerations
    
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
        """主要执行函数"""
        if self.current_joint_positions is None:
            rospy.logerr("未获取到当前关节状态！")
            return
        
        # 使用Bi-RRT规划路径
        path = self.build_rrt_trees(
            np.array(self.current_joint_positions), 
            np.array(goal_positions)
        )
        
        if path is None:
            rospy.logerr("无法找到有效路径！")
            return
        
        # 路径平滑
        smoothed_path = self.smooth_path(path)
        
        # 为每段路径生成轨迹
        total_points = []
        total_velocities = []
        segment_duration = duration / (len(smoothed_path) - 1)
        
        for i in range(len(smoothed_path) - 1):
            positions, velocities, _ = self.generate_trajectory(
                smoothed_path[i], 
                smoothed_path[i+1], 
                segment_duration
            )
            total_points.extend(positions)
            total_velocities.extend(velocities)
        
        # 构建轨迹消息
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names

        timestamps = np.linspace(0, duration, len(total_points))
        
        for i in range(len(total_points)):
            point = JointTrajectoryPoint()
            point.positions = [total_points[j][i] for j in range(len(self.joint_names))]
            point.velocities = [total_velocities[j][i] for j in range(len(self.joint_names))]
            point.time_from_start = rospy.Duration(timestamps[i])
            traj_msg.points.append(point)
        
        # 发送轨迹
        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj_msg
        self.arm_client.send_goal(goal)
        
        # 等待执行完成
        self.arm_client.wait_for_result(rospy.Duration(duration + 5.0))
        
        rospy.loginfo("运动规划执行完成！")

        gravity_torques = self.calculate_gravity_torques()

        g = 9.81

        for i in range(len(total_points)):
            target_msg = TrajectoryData()
            target_msg.joint_names = self.joint_names
            target_msg.positions = [total_points[j][i] for j in range(len(self.joint_names))]
            current_velocities = [total_velocities[j][i] for j in range(len(self.joint_names))]
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

if __name__ == '__main__':
    try:
        rospy.init_node('birrt_planner')
        planner = BiRRTPlanner()
        
        T_goal = cal_fk([2.22601256 , 1.2024973 , -2.72513796 ,-2.21730977, -2.19911507 , 0.1795953,  0])

        q_current = np.zeros(7)

        # Test q7 value
        q7_test = 0.0

        start_time = time.perf_counter()
        solutions = franka_IK_EE(T_goal, q7_test, q_current)

        # 示例目标位置
        goal_positions = [0.5, 0.5, 0.0, -1.0, 0.0, 1.5, 0.0]

        sol = list(solutions[0])

        planner.move_to_goal(solutions[0], duration=10.0)
        
    except rospy.ROSInterruptException:
        pass
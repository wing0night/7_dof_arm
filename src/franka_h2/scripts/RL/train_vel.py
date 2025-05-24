#!/usr/bin/env python3
import torch

from PPOAgent import PPOAgent

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


from threading import Lock

from gazebo_msgs.msg import ModelStates, LinkStates
from gazebo_msgs.srv import GetModelState, GetLinkState, GetJointProperties
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from std_srvs.srv import Empty

from tqdm import tqdm

from fk import cal_fk
from extract_pose import extract_pose

# 获取包路径
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('franka_h2')

class RobotEnv:
    """ROS机械臂环境封装类"""
    def __init__(self):
        self.position_lock = Lock()  # 关节状态访问锁
        
        # ROS初始化
        rospy.init_node('rl_arm_control')
        
        self.arm_client = actionlib.SimpleActionClient(
            'panda_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction
        )
        self.arm_client.wait_for_server()

        
        # 构建URDF文件路径
        self.urdf_path = os.path.join(pkg_path, 'urdf', 'panda_robot_gazebo.urdf')

        # 添加机器人模型作为类成员
        self.panda = rtb.models.URDF.Panda()
        
        # 添加状态标志
        self.state_updated = False
        self.last_update_time = rospy.Time.now()
        
        # 状态参数
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 
                           'joint5', 'joint6', 'joint7']
        self.state_dim = 7 * 3 + 7  # 7关节(位置+速度+力矩) + 末端位置&三轴角
        # 后续删掉力矩，换成目标末端位置&三轴角
        self.action_dim = 7
        # 初始化状态存储
        # 状态存储
        self.current_state = None
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_joint_efforts = None
        self.current_ee_pos = None
        self.axis_angle = None
        self.position_ee = None
        self.position_ee_tar = None
        self.axis_angle_tar = None

        self.max_joint_velocity = 3.14
        
        # 订阅者
        self.state_sub = rospy.Subscriber(
            '/joint_states', 
            JointState
        )
        # 控制运动的消息的发布
        self.cmd_pub = rospy.Publisher('/joint_trajectory', JointTrajectory, queue_size=10)
        # 这里可以是自定义消息的发布器用来绘图
        self.traj_pub = rospy.Publisher('/trajectory_data', TrajectoryData, queue_size=10)

        # 等待第一次状态更新
        # rospy.loginfo("等待接收第一帧关节状态...")
        # while not self.state_updated and not rospy.is_shutdown():
        #     rospy.sleep(0.1)
        # rospy.loginfo("成功接收到关节状态！")

        self.velocity_pub = rospy.Publisher(
            '/arm_controller/command',  # 示例topic
            Float64MultiArray, 
            queue_size=1
        )
        self.dt = 3.0 # step中的时间间隔
        
        
        self.target_pos = np.array([-0.0760815, 0.16729992, 0.76910605])  # 目标末端位置
        # 定义目标位置和旋转
        self.x = -0.0760815
        self.y = 0.16729992
        self.z = 0.76910605
        self.R_custom = np.array(
            [[-0.50410872, -0.13489031 ,-0.85304103 ],
 [-0.81089773 , 0.41381235 , 0.41376831  ],
 [ 0.29718558,  0.90031325 ,-0.31798865  ]
        ])
        
        # 创建4x4单位矩阵
        # self.T_tar = np.eye(4)
        # # 填入旋转部分 (左上3x3)
        # self.T_tar[:3, :3] = self.R_custom
        # # 填入平移部分 (前三行第四列)
        # self.T_tar[:3, 3] = [self.x, self.y, self.z]
        self.T_tar = cal_fk([2.22601256 , 1.2024973 , -2.72513796 ,-2.21730977, -2.19911507 , 0.1795953,  0])
        
        # 加载URDF并解析惯性参数
        self.robot = URDF.from_xml_file(self.urdf_path)
        self.inertia_params = self._load_inertia_params()
        
        # 关节限位示例（需根据实际URDF修改）
        self.joint_limits = [
            [-2.8973, 2.8973],  # joint1
            [-1.7628, 1.7628],  # joint2
            [-2.8973, 2.8973],  # joint3
            [-3.0718, -0.0698], # joint4
            [-2.8973, 2.8973],  # joint5
            [-0.0175, 3.7525],  # joint6
            [-2.8973, 2.8973]   # joint7
        ]
        # self.joint_limits = [
        #     [-25, 25],  # joint1
        #     [-15, 15],  # joint2
        #     [-25, 25],  # joint3
        #     [-25, 25], # joint4
        #     [-25, 25],  # joint5
        #     [-25, 25],  # joint6
        #     [-25, 25]   # joint7
        # ]

        # 添加Gazebo服务客户端
        rospy.loginfo("等待Gazebo服务...")
        try:
            rospy.wait_for_service('/gazebo/get_model_state', timeout=5.0)
            rospy.wait_for_service('/gazebo/get_link_state', timeout=5.0)
            rospy.wait_for_service('/gazebo/get_joint_properties', timeout=5.0)
            
            self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            self.get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
            self.get_joint_properties = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
            
            rospy.loginfo("Gazebo服务连接成功")
        except rospy.ROSException as e:
            rospy.logerr(f"连接Gazebo服务失败: {str(e)}")
            raise
        
        # 订阅Gazebo的状态话题
        self.model_states_sub = rospy.Subscriber(
            '/gazebo/model_states', 
            ModelStates, 
            self.gazebo_model_callback
        )
        
        self.link_states_sub = rospy.Subscriber(
            '/gazebo/link_states', 
            LinkStates, 
            self.gazebo_link_callback
        )
        
        # 存储Gazebo状态
        self.gazebo_model_state = None
        self.gazebo_link_states = None

        self.prev_velocities = None
        self.prev_time = None
    
    def gazebo_model_callback(self, msg):
        """处理Gazebo模型状态更新"""
        try:
            # 找到panda机器人的索引
            if 'panda' in msg.name:
                idx = msg.name.index('panda')
                self.gazebo_model_state = {
                    'pose': msg.pose[idx],
                    'twist': msg.twist[idx]
                }
        except Exception as e:
            rospy.logerr(f"处理Gazebo模型状态时出错: {str(e)}")
    
    def gazebo_link_callback(self, msg):
        """处理Gazebo链接状态更新"""
        try:
            self.gazebo_link_states = {}
            for i, name in enumerate(msg.name):
                if 'panda' in name:
                    self.gazebo_link_states[name] = {
                        'pose': msg.pose[i],
                        'twist': msg.twist[i]
                    }
        except Exception as e:
            rospy.logerr(f"处理Gazebo链接状态时出错: {str(e)}")
    
    def get_gazebo_joint_state(self, joint_name):
        """获取指定关节的状态"""
        try:
            # 获取关节属性
            resp = self.get_joint_properties(f"{joint_name}")
            # print(resp)
            if resp.success:
                # print(f"获取关节 {joint_name} 状态成功：", resp)
                return {
                    'position': resp.position[0],
                    'rate': resp.rate[0],
                    # 'effort': resp.effort[0]
                }
            else:
                rospy.logwarn(f"获取关节 {joint_name} 状态失败")
                return None
        except Exception as e:
            rospy.logerr(f"调用Gazebo服务失败: {str(e)}")
            return None
    
    def get_all_joint_states_from_gazebo(self):
        """从Gazebo获取所有关节状态"""
        joint_states = {}
        for joint_name in self.joint_names:
            state = self.get_gazebo_joint_state(joint_name)
            if state:
                joint_states[joint_name] = state
        return joint_states
    
    def calculate_joint_efforts(self, positions, velocities, prev_velocities=None, dt=0.02):
        """计算关节力矩
        Args:
            positions: 当前关节位置
            velocities: 当前关节速度
            prev_velocities: 上一时刻关节速度（用于计算加速度）
            dt: 时间间隔（默认0.02s，对应50Hz）
        """
        try:
            efforts = []
            
            # 计算加速度
            if prev_velocities is not None:
                accelerations = [(v - pv)/dt for v, pv in zip(velocities, prev_velocities)]
            else:
                accelerations = [0.0] * len(velocities)
            
            # 计算重力矩
            g = 9.81  # 重力加速度
            
            for i, joint_name in enumerate(self.joint_names):
                # 获取关节参数
                params = self.inertia_params.get(joint_name)
                if not params:
                    rospy.logwarn(f"未找到关节 {joint_name} 的惯性参数")
                    efforts.append(0.0)
                    continue
                
                # 解包参数
                mass = params['mass']
                com = np.array(params['com_position'])
                inertia_matrix = np.array(params['inertia_matrix'])
                
                # 1. 计算重力矩
                # 基坐标系重力矢量
                base_gravity = np.array([0, 0, -g])
                # 简化的重力矩计算（假设关节坐标系与基坐标系对齐）
                gravity_torque = np.cross(com, mass * base_gravity)[2]  # 取Z轴分量
                
                # 2. 计算惯性力矩
                # 使用转动惯量矩阵的Z轴分量（因为关节绕Z轴转动）
                inertia = inertia_matrix[2][2]
                inertial_torque = inertia * accelerations[i]
                
                # 3. 计算科里奥利力和离心力（简化模型）
                # 使用当前速度的平方来近似
                coriolis_centrifugal = 0.1 * mass * velocities[i]**2  # 系数0.1是一个经验值，可以调整
                
                # 4. 计算摩擦力（简化模型）
                # 使用粘性摩擦系数
                viscous_friction = 0.1 * velocities[i]  # 粘性摩擦系数为0.1，可以调整
                # 库仑摩擦（符号函数）
                coulomb_friction = 0.05 * np.sign(velocities[i])  # 库仑摩擦系数为0.05，可以调整
                
                # 合计所有力矩
                total_torque = (gravity_torque + 
                            inertial_torque + 
                            coriolis_centrifugal + 
                            viscous_friction + 
                            coulomb_friction)
                
                efforts.append(total_torque)
                
            return np.array(efforts)
            
        except Exception as e:
            rospy.logerr(f"计算关节力矩时出错: {str(e)}")
            return np.zeros(len(self.joint_names))
    
    def update_state_from_gazebo(self):
        """使用Gazebo数据更新机器人状态"""
        try:
            # 获取所有关节状态
            joint_states = self.get_all_joint_states_from_gazebo()
            if not joint_states:
                return False
            
            # 更新状态
            with self.position_lock:
                # 更新关节位置
                positions = []
                velocities = []
                efforts = []
                
                for joint_name in self.joint_names:
                    if joint_name in joint_states:
                        state = joint_states[joint_name]
                        positions.append(state['position'])
                        velocities.append(state['rate'])
                        # efforts.append(state['effort'])
                
                
                positions = np.array(positions)
                velocities = np.array(velocities)
                
                # 计算力矩
                efforts = self.calculate_joint_efforts(
                    positions,
                    velocities,
                    self.current_joint_velocities,  # 使用上一时刻的速度
                    dt=0.02  # 假设50Hz的更新频率
                )
                


                self.current_joint_positions = np.array(positions)
                self.current_joint_velocities = np.array(velocities)
                self.current_joint_efforts = np.array(efforts)
                
                # 计算末端执行器位置
                # T = self.panda.fkine(self.current_joint_positions)
                # self.current_ee_pos = np.array([T.t[0], T.t[1], T.t[2]])
                # 获得当前末端位姿和三轴角表示
                T_ee = cal_fk(self.current_joint_positions)
                position_ee, axis_angle = extract_pose(T_ee)
                self.position_ee = np.array(position_ee)
                self.axis_angle = np.array(axis_angle)
                
                position_ee_tar, axis_angle_tar = extract_pose(self.T_tar)
                self.position_ee_tar = np.array(position_ee_tar)
                self.axis_angle_tar = np.array(axis_angle_tar)
                
                # 更新完整状态向量
                self.current_state = np.concatenate([
                    self.current_joint_positions, # [0:7]
                    self.current_joint_velocities, # [7:14]
                    self.position_ee, # [14:17]
                    self.axis_angle, # [17:21]
                    self.position_ee_tar, # [21:24]
                    self.axis_angle_tar # [24:28]
                ])
                
                self.state_updated = True
                self.last_update_time = rospy.Time.now()
                
            return True
            
        except Exception as e:
            rospy.logerr(f"从Gazebo更新状态时出错: {str(e)}")
            return False
    
    def _handle_action_done(self, status, result):
        # with self.position_lock:
        self.current_joint_positions = result.actual.positions
        
    def state_cb(self, msg):
        """改进的状态回调函数"""
        try:
            # 提取关节状态
            joint_pos = np.array(msg.position[:7])
            joint_vel = np.array(msg.velocity[:7] if msg.velocity else [0.0]*7)
            joint_eff = np.array(msg.effort[:7] if msg.effort else [0.0]*7)
            
            # 计算末端执行器位置
            T = self.panda.fkine(joint_pos)
            end_effector_pos = np.array([T.t[0], T.t[1], T.t[2]])
            
            # 使用锁保护状态更新
            # with self.position_lock:
            # 更新各个状态组件
            self.current_joint_positions = joint_pos
            self.current_joint_velocities = joint_vel
            self.current_joint_efforts = joint_eff
            self.current_ee_pos = end_effector_pos
            
            # 构建完整状态向量
            self.current_state = np.concatenate([
                joint_pos, joint_vel, joint_eff, end_effector_pos
            ])
            
            # 更新状态标志
            self.state_updated = True
            self.last_update_time = rospy.Time.now()
            
            rospy.logdebug(f"状态更新成功: pos={joint_pos}")
            
        except Exception as e:
            rospy.logerr(f"状态回调处理错误: {str(e)}")
    
    def wait_for_state_update(self, timeout=1.0):
        """等待状态更新的辅助函数"""
        start_time = rospy.Time.now()
        rate = rospy.Rate(50)  # 50Hz检查频率
        
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                return False
                
            # with self.position_lock:
            if self.state_updated:
                self.state_updated = False  # 重置标志
                return True
            
            rate.sleep()
        
        return False
    
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
        
    def calculate_reward(self, action):
        """多目标奖励计算"""

        # 获取新状态
        # 直接从Gazebo获取最新状态
        if not self.update_state_from_gazebo():
            rospy.logwarn("无法从Gazebo更新状态")
            return None, 0, True, {}

        # 验证状态是否有效
        if self.current_state is None:
            rospy.logwarn("计算奖励时状态为空")
            return 0.0
        
        # 获取新状态并验证变化
        new_state = self.current_state.copy()
        
        # 位置误差奖励
        pos_error = np.linalg.norm(new_state[14:17] - self.target_pos)
        reward_pos = -2.0 * np.exp(2 * pos_error)
        
        # 平滑度奖励
        joint_vel = new_state[7:14]
        joint_acc = np.diff(joint_vel)
        # reward_smooth = -0.5 * (np.sum(joint_vel**2) - 0.2 * np.sum(joint_acc**2))
        reward_smooth = -5 * (np.sum(joint_vel**2) - 0.2 * np.sum(joint_acc**2))
        
        # # 能量消耗奖励
        joint_effort = new_state[14:21]
        reward_energy = -0.2 * np.sum((joint_effort**2))
        
        # 额外奖励
        reward_extra = 10.0 if pos_error < 0.005 else 10 / (1 + 5 * pos_error)
        
        total_reward = reward_pos + reward_smooth + reward_energy + reward_extra
        # total_reward = reward_pos + reward_extra
        # print(f"Total Reward: {total_reward}\n")
        print(f"位置误差奖励: {reward_pos:.4f}, 平滑度奖励: {reward_smooth:.4f}, 能量消耗奖励: {reward_energy:.4f}, 额外奖励: {reward_extra:.4f}")
        
        return total_reward
    
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
    
    def _scale_action(self, action):
        """将归一化动作映射到物理速度范围"""
        # 示例：若policy输出层用tanh激活
        return action * self.max_joint_velocity  # max_joint_velocity=3.14
    
    def step(self, action):
        """执行动作并获取下一个状态"""
        try:
            # 1. 获取当前状态
            if not self.update_state_from_gazebo():
                rospy.logwarn("无法从Gazebo更新状态")
                return None, 0, True, {}
                
            current_positions = self.current_joint_positions.copy()
            
            # 2. 将速度动作转换为位置目标
            # 使用当前速度和时间步长计算目标位置
            scaled_vel = self._scale_action(action)  # 缩放速度到合理范围
            target_positions = current_positions + scaled_vel * self.dt
            
            # 3. 检查关节限位
            for i, (pos, limit) in enumerate(zip(target_positions, self.joint_limits)):
                target_positions[i] = max(min(pos, limit[1]), limit[0])
            
            # 4. 生成轨迹
            duration = self.dt  # 使用与step相同的时间
            num_points = int(duration * 50)  # 50Hz的采样频率
            timestamps = np.linspace(0, duration, num_points)
            
            # 存储插值数据
            all_positions = []
            all_velocities = []
            for joint_idx in range(len(target_positions)):
                start = current_positions[joint_idx]
                goal = target_positions[joint_idx]
                positions, velocities, _ = self.quintic_interpolation(
                    start, 
                    goal, 
                    duration, 
                    freq=50
                )
                all_positions.append(positions)
                all_velocities.append(velocities)
            
            # 5. 构建轨迹消息
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names
            
            for i in range(num_points):
                point = JointTrajectoryPoint()
                point.positions = [all_positions[j][i] for j in range(len(self.joint_names))]
                point.velocities = [all_velocities[j][i] for j in range(len(self.joint_names))]
                point.time_from_start = rospy.Duration(timestamps[i])
                traj_msg.points.append(point)
            
            # 6. 发送轨迹
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = traj_msg
            self.arm_client.send_goal(goal)
            
            # 7. 等待执行完成
            success = self.arm_client.wait_for_result(rospy.Duration(duration + 0.5))
            if not success:
                rospy.logwarn("轨迹执行超时")
                return None, 0, True, {}
            
            # 8. 发布轨迹数据用于可视化
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
                
                target_msg.acc = accelerations
                
                # 更新力矩
                target_msg.torques = self.current_joint_efforts.tolist()
                
                # 更新时间记录
                self.prev_velocities = current_velocities
                self.prev_time = current_time
                
                # 发布时间戳和消息
                target_msg.stamp = rospy.Time.now() + rospy.Duration(current_time)
                self.traj_pub.publish(target_msg)
                rospy.sleep(1.0 / 50)
            
            # 9. 获取新状态
            if not self.update_state_from_gazebo():
                rospy.logwarn("无法从Gazebo更新状态")
                return None, 0, True, {}
            
            new_state = self.current_state.copy()
            
            # 10. 计算奖励
            reward = self.calculate_reward(action)
            
            # 11. 终止判断
            done = True
            
            return new_state, reward, done, {}
            
        except Exception as e:
            rospy.logerr(f"执行步骤时出错: {str(e)}")
            return None, 0, True, {}
    
    def reset(self):
        self.update_state_from_gazebo()
        """重置环境到初始状态"""
        if self.current_joint_positions is None:
            rospy.logerr("未获取到当前关节状态！")
            return
        # goal_positions = [2.22601256 , 1.2024973 , -2.72513796 ,-2.21730977, -2.19911507 , 0.1795953,  0]
        goal_positions = np.array([0, 0, 0, -1.7, 0, 0, 0])  # 目标关节位置
        duration = 5.0  # 重置时间

        num_points = int(duration * 50)
        timestamps = np.linspace(0, duration, num_points)

        # gravity_torques = self.calculate_gravity_torques()

        g = 9.81

        # 存储插值数据
        all_positions = []
        all_velocities = []
        for joint_idx in range(len(goal_positions)):
            start = self.current_joint_positions[joint_idx]
            goal = goal_positions[joint_idx]
            positions, velocities, _ = self.quintic_interpolation(start, goal, duration, 50)
            all_positions.append(positions)
            all_velocities.append(velocities)
            # trajectory = self.generate_trajectory(start, goal, duration)

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

            index = 0

            # 力矩计算部分
            for name in self.joint_names:  # 按joint_names顺序遍历
                # 获取当前关节参数
                params = self.inertia_params[name]
                
                # 获取对应的加速度（按索引匹配顺序）
                j = self.joint_names.index(name)
                acc = accelerations[j]
                
                # 获取当前关节重力矩
                # grav = gravity_torques[name]
                
                # # 计算惯性力矩 + 重力补偿
                # inertia_torque = np.dot(params['inertia_matrix'], [0, 0, acc])[2]  # 取绕Z轴分量
                # total_torque = inertia_torque + grav
                
                target_msg.torques.append(self.current_joint_efforts[index])
                index += 1

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
        self.update_state_from_gazebo()
        return self.current_state
    
    def _verify_robot_state(self):
        """验证机器人状态"""
        try:
            # 发布一个小的测试动作
            test_action = np.array([1, 0.0, 0.0, 0, 0.0, 0.0, 0.0])
            initial_state = self.current_state[:7].copy()
            
            # 执行测试动作
            self.step(test_action)
            
            # 验证状态是否改变
            if np.allclose(self.current_state[:7], initial_state):
                rospy.logerr("机器人状态未响应测试动作！")
                return False
            print("机器人状态验证成功！")
                
            return True
            
        except Exception as e:
            rospy.logerr(f"状态验证失败: {str(e)}")
            return False
    
    def plot_training_rewards(self, rewards, save_path):
        """
        绘制训练奖励曲线
        Args:
            episode_rewards: 每个episode的奖励列表
            save_path: 图片保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 绘制原始奖励曲线
            episodes = range(1, len(rewards) + 1)
            plt.plot(episodes, rewards, 'b-', alpha=0.3, label='origin reward')
            
            # 计算移动平均
            window_size = 5
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
                plt.plot(range(window_size, len(rewards) + 1), 
                        moving_avg, 'r-', label=f'{window_size}move avg')
            
            # 添加统计信息
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            max_reward = np.max(rewards)
            min_reward = np.min(rewards)
            
            # 设置图表属性
            plt.title('Reward Curve')
            plt.xlabel('Round')
            plt.ylabel('Total Reward')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            # 添加统计信息文本框
            stats_text = f'info:\n' \
                        f'mean reward: {mean_reward:.2f}\n' \
                        f'std reward: {std_reward:.2f}\n' \
                        f'max reward: {max_reward:.2f}\n' \
                        f'min reward: {min_reward:.2f}'
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 保存图片（如果指定了保存路径）
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                rospy.loginfo(f"训练曲线已保存到: {save_path}")
            
            # 显示图表
            plt.show()
            
        except Exception as e:
            rospy.logerr(f"绘制训练曲线时出错: {str(e)}")

def train():
    """主训练循环"""
    # 初始化环境和智能体
    env = RobotEnv()
    agent = PPOAgent(env.state_dim, env.action_dim)
    
    # 验证机器人状态
    # if not env._verify_robot_state():
    #     rospy.logerr("机器人状态验证失败，请检查控制器和传感器！")
    #     return
    
    # 训练参数
    max_episodes = 50
    episode_rewards = []

    # 直接迭代 progress_bar 对象
    progress_bar = tqdm(range(max_episodes), desc="Training", unit="episode")

    for ep in progress_bar:  # 关键修改：直接迭代进度条对象
        steps = 0
        if ep % 10 == 0 and agent.episode_length > agent.min_ep_length:
            agent.episode_length = int(agent.episode_length * agent.decay_rate)
        
        # 重置环境并确保获得有效状态
        state = env.reset()
        if state is None:
            rospy.logerr("无法获取初始状态，跳过此回合")
            continue
        print(f"Episode {ep} - 初始状态: {state}")
            
        episode_reward = 0
        memory = []
        
        for step in range(agent.episode_length):
            try:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = agent.get_action(state_tensor, training=True)
                action = action.detach().squeeze(0).numpy()
                
                next_state, reward, done, _ = env.step(action)
                
                if next_state is None:
                    rospy.logwarn("获取到无效的next_state，结束当前回合")
                    break
                
                memory.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                
                if done:
                    break
                
                if len(memory) >= agent.batch_size:
                    agent.update_policy(*zip(*memory))
                    memory = []
                
                steps += 1
            
            except Exception as e:
                rospy.logerr(f"训练步骤出错: {str(e)}")
                break
        
        episode_rewards.append(episode_reward)
        
        # 自动更新进度条（通过迭代 progress_bar 对象实现）
        progress_bar.set_postfix({
            "ep_reward": episode_reward,
            "steps": steps
        })

    progress_bar.close()
    
    # 保存模型
    try:
        pth_path = os.path.join(pkg_path, 'scripts/RL/pth', 'arm_ppo.pth')
        torch.save(agent.policy.state_dict(), pth_path)
        print(f"模型已保存到: {pth_path}")

        # 绘制并保存训练曲线
        plot_path = os.path.join(pkg_path, 'scripts/RL/plots', 'training_rewards.png')
        # 确保保存目录存在
        # os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        env.plot_training_rewards(episode_rewards, plot_path)
    except Exception as e:
        rospy.logerr(f"保存模型失败: {str(e)}")

def test():
    env = RobotEnv()
    env.reset()



if __name__ == "__main__":
    try:
        train()
        # test()
    except rospy.ROSInterruptException:
        pass

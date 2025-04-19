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

class RobotEnv:
    """ROS机械臂环境封装类"""
    def __init__(self):
        self.position_lock = Lock()  # 关节状态访问锁
        
        # ROS初始化
        rospy.init_node('rl_arm_control')
        
        # 初始化标准轨迹动作客户端
        self.arm_client = actionlib.SimpleActionClient(
            'panda_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction
        )
        self.arm_client.wait_for_server()
        self.arm_client.done_cb = self._handle_action_done  # 注册回调
        # 利用actionlib提供的接口对Gazebo中的机械臂进行控制并获取回调更新位置状态

        # 获取包路径
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('franka_h2')
        # 构建URDF文件路径
        self.urdf_path = os.path.join(pkg_path, 'urdf', 'panda_robot_gazebo.urdf')
        
        # 状态参数
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 
                           'joint5', 'joint6', 'joint7']
        self.state_dim = 7 * 3 + 3  # 7关节(位置+速度+力矩) + 末端位置
        self.action_dim = 7
        
        # 订阅者和发布者
        self.state_sub = rospy.Subscriber('/joint_states', JointState, self.state_cb)
        # 控制运动的消息的发布
        self.cmd_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
        # 这里可以是自定义消息的发布器用来绘图
        self.traj_pub = rospy.Publisher('/trajectory_data', TrajectoryData, queue_size=10)
        
        # 初始化状态存储
        self.current_state = None
        self.target_pos = np.array([0.5, 0.2, 0.5])  # 目标末端位置
        # 定义目标位置和旋转
        self.x = 0.5
        self.y = 0.2
        self.z = 0.5
        self.R_custom = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        
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
    
    def _handle_action_done(self, status, result):
        with self.position_lock:
            self.current_joint_positions = result.actual.positions
        
    def state_cb(self, msg):
        """状态回调函数"""
        # 解析关节状态：位置、速度、力矩
        joint_pos = np.array(msg.position[:7])  # 确保是7维
        joint_vel = np.array(msg.velocity[:7] if msg.velocity else [0.0]*7)  # 如果没有速度信息，用0填充
        joint_eff = np.array(msg.effort[:7] if msg.effort else [0.0]*7)  # 如果没有力矩信息，用0填充
        
        # 获取末端执行器位置（使用正运动学计算）
        # 创建Panda机器人模型
        panda = rtb.models.URDF.Panda()
        # 计算末端执行器位置
        T = panda.fkine(joint_pos)
        end_effector_pos = np.array([T.t[0], T.t[1], T.t[2]])
        
        # 组合状态向量，确保维度正确
        self.current_state = np.concatenate([
            joint_pos,     # 7维
            joint_vel,     # 7维
            joint_eff,     # 7维
            end_effector_pos  # 3维
        ])
    
        # 验证维度
        assert len(self.current_state) == self.state_dim, \
            f"状态维度不匹配：期望{self.state_dim}，实际{len(self.current_state)}"
    
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
        """多目标奖励计算（论文公式11-15）"""
        # 位置误差奖励（公式12）
        pos_error = np.linalg.norm(self.current_state[-3:] - self.target_pos)
        reward_pos = -1.0 * np.exp(2 * pos_error)
        
        # 平滑度奖励（公式13）
        joint_vel = self.current_state[7:14]
        joint_acc = np.diff(joint_vel)  # 简化的加速度计算
        reward_smooth = -0.5 * (np.sum(joint_vel**2) + 0.2 * np.sum(joint_acc**2))
        
        # 能量消耗奖励（公式14）
        joint_effort = self.current_state[14:21]
        reward_energy = -0.3 * np.sum((joint_effort**2))
        
        # 额外奖励（公式15）
        if pos_error < 0.005:
            reward_extra = 10.0
        else:
            reward_extra = 10 / (1 + 10 * pos_error)
            
        return reward_pos + reward_smooth + reward_energy + reward_extra
    
    def step(self, action):
        # 创建轨迹消息
        cmd_msg = JointTrajectory()
        cmd_msg.joint_names = self.joint_names
        
        # 获取当前状态作为起点，确保是float类型
        current_pos = [float(x) for x in self.current_state[:7]]
        
        # 创建短时间的轨迹（而不是单点）
        point1 = JointTrajectoryPoint()
        point1.positions = current_pos
        point1.velocities = [0.0] * 7
        point1.time_from_start = rospy.Duration(0.0)
        
        # 估算一个中间轨迹点
        point2 = JointTrajectoryPoint()
        
        # 正确处理numpy数组类型的action
        action = action.flatten()  # 确保是1维数组
        next_pos = (np.array(current_pos) + action * 0.1).tolist()  # 使用numpy计算后转换为list
        
        point2.positions = next_pos
        point2.velocities = action.tolist()  # 直接将numpy数组转换为list
        point2.time_from_start = rospy.Duration(0.1)
        
        cmd_msg.points = [point1, point2]
        
        # 使用actionlib发送和等待
        goal = FollowJointTrajectoryGoal(trajectory=cmd_msg)
        self.arm_client.send_goal(goal)
        self.arm_client.wait_for_result(rospy.Duration(0.2))
        
        return self.current_state.copy(), self.calculate_reward(action), False, {}
    
    def reset(self):
        """重置环境到初始状态"""
        try:
            # 创建初始轨迹消息
            cmd_msg = JointTrajectory()
            cmd_msg.joint_names = self.joint_names
            
            # 设置初始关节角度（确保使用float类型）
            initial_positions = [float(x) for x in [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]]
            
            # 创建轨迹点
            point = JointTrajectoryPoint()
            # 确保所有数值都是float类型
            point.positions = [float(pos) for pos in initial_positions]
            point.velocities = [float(0.0)] * 7  # 明确指定float类型
            point.time_from_start = rospy.Duration(1.0)
            
            cmd_msg.points.append(point)
            
            # 使用actionlib发送和等待
            goal = FollowJointTrajectoryGoal(trajectory=cmd_msg)
            self.arm_client.send_goal(goal)
            
            # 等待机械臂到达初始位置
            self.arm_client.wait_for_result(rospy.Duration(2.0))
            
            # 等待一小段时间确保状态更新
            rospy.sleep(0.1)
            
            # 确保我们有有效的状态
            while self.current_state is None and not rospy.is_shutdown():
                rospy.loginfo("等待初始状态更新...")
                rospy.sleep(0.1)
            
            return self.current_state.copy()
            
        except Exception as e:
            rospy.logerr(f"重置环境时出错: {str(e)}")
            return None

def train():
    """主训练循环"""
    # 初始化环境和智能体
    env = RobotEnv()
    agent = PPOAgent(env.state_dim, env.action_dim)
    
    # 训练参数
    max_episodes = 1000
    episode_rewards = []
    
    for ep in range(max_episodes):
        if ep % 10 == 0 and agent.episode_length > agent.min_ep_length:
            agent.episode_length = int(agent.episode_length * agent.decay_rate)
            
        state = env.reset()
        episode_reward = 0
        memory = []
        
        for step in range(agent.episode_length):
            # 确保状态是正确的格式
            state_tensor = torch.FloatTensor(state)
            if len(state_tensor.shape) == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # 获取动作
            action, _ = agent.policy(state_tensor)
            # 正确处理动作数据
            action = action.detach().squeeze(0).numpy()  # 确保是numpy数组并移除多余的维度
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            memory.append((state, action, reward, next_state, done))
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            
            if len(memory) >= agent.batch_size:
                agent.update_policy(*zip(*memory))
                memory = []
                
        episode_rewards.append(episode_reward)
        print(f"Episode {ep}, Reward: {episode_reward:.2f}, Length: {agent.episode_length}")
    
    torch.save(agent.policy.state_dict(), 'arm_ppo.pth')

if __name__ == "__main__":
    try:
        train()
    except rospy.ROSInterruptException:
        pass



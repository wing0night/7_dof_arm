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

# 获取包路径
rospack = rospkg.RosPack()
pkg_path = rospack.get_path('franka_h2')

class RobotEnv:
    """ROS机械臂环境封装类"""
    def __init__(self):
        self.position_lock = Lock()  # 关节状态访问锁
        
        # ROS初始化
        rospy.init_node('rl_arm_control')
        
        # # 初始化标准轨迹动作客户端
        # self.arm_client = actionlib.SimpleActionClient(
        #     'panda_controller/follow_joint_trajectory', 
        #     FollowJointTrajectoryAction
        # )
        # self.arm_client.wait_for_server()
        # self.arm_client.done_cb = self._handle_action_done  # 注册回调
        # # 利用actionlib提供的接口对Gazebo中的机械臂进行控制并获取回调更新位置状态
        self.arm_client = actionlib.SimpleActionClient(
            'panda_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction
        )
        if not self.arm_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("Action server not available!")
            raise RuntimeError("Action server not available!")

        
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
        self.state_dim = 7 * 3 + 3  # 7关节(位置+速度+力矩) + 末端位置
        self.action_dim = 7
        # 初始化状态存储
        # 状态存储
        self.current_state = None
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.current_joint_efforts = None
        self.current_ee_pos = None
        
        # 订阅者
        self.state_sub = rospy.Subscriber(
            '/joint_states', 
            JointState, 
            self.state_cb,
            queue_size=10
        )
        # 控制运动的消息的发布
        self.cmd_pub = rospy.Publisher('/joint_trajectory', JointTrajectory, queue_size=10)
        # 这里可以是自定义消息的发布器用来绘图
        self.traj_pub = rospy.Publisher('/trajectory_data', TrajectoryData, queue_size=10)

        # 等待第一次状态更新
        rospy.loginfo("等待接收第一帧关节状态...")
        while not self.state_updated and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("成功接收到关节状态！")
        
        
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
            with self.position_lock:
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
                
            with self.position_lock:
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
        # 验证状态是否有效
        if self.current_state is None:
            rospy.logwarn("计算奖励时状态为空")
            return 0.0
            
        # 打印当前状态各部分
        print("\nCurrent State Components:")
        print(f"Joint Positions: {self.current_state[:7]}")
        print(f"Joint Velocities: {self.current_state[7:14]}")
        print(f"Joint Efforts: {self.current_state[14:21]}")
        print(f"End Effector Position: {self.current_state[-3:]}")
        print(f"Target Position: {self.target_pos}")
        
        # 位置误差奖励
        pos_error = np.linalg.norm(self.current_state[-3:] - self.target_pos)
        reward_pos = -1.0 * np.exp(2 * pos_error)
        
        # 平滑度奖励
        joint_vel = self.current_state[7:14]
        joint_acc = np.diff(joint_vel)
        reward_smooth = -0.5 * (np.sum(joint_vel**2) + 0.2 * np.sum(joint_acc**2))
        
        # 能量消耗奖励
        joint_effort = self.current_state[14:21]
        reward_energy = -0.3 * np.sum((joint_effort**2))
        
        # 额外奖励
        reward_extra = 10.0 if pos_error < 0.005 else 10 / (1 + 10 * pos_error)
        
        # 打印详细的奖励计算过程
        print("\nReward Calculation Details:")
        print(f"Position Error: {pos_error}")
        print(f"Position Reward: {reward_pos}")
        print(f"Smoothness Reward: {reward_smooth}")
        print(f"Energy Reward: {reward_energy}")
        print(f"Extra Reward: {reward_extra}")
        
        total_reward = reward_pos + reward_smooth + reward_energy + reward_extra
        print(f"Total Reward: {total_reward}\n")
        
        return total_reward
    
    def generate_trajectory(self, current_pos, next_pos, duration=1.0, freq=50):
        """生成完整的轨迹"""
        num_points = int(duration * freq)
        t = np.linspace(0, duration, num_points)
        positions = []
        velocities = []
        
        for ti in t:
            # 五次多项式插值
            normalized_time = ti / duration
            s = 10 * normalized_time**3 - 15 * normalized_time**4 + 6 * normalized_time**5
            s_dot = (30 * normalized_time**2 - 60 * normalized_time**3 + 30 * normalized_time**4) / duration
            
            pos = []
            vel = []
            for i in range(len(current_pos)):
                delta = next_pos[i] - current_pos[i]
                pos.append(current_pos[i] + s * delta)
                vel.append(s_dot * delta)
            
            positions.append(pos)
            velocities.append(vel)
            
        return positions, velocities, t
    
    def step(self, action):
        """改进的step函数"""
        try:
            rospy.loginfo("开始执行step函数...")
            
            # 保存旧状态
            with self.position_lock:
                if self.current_state is None:
                    rospy.logwarn("当前状态为空，无法执行动作！")
                    return None, 0, True, {}
                old_state = self.current_state.copy()
                current_pos = self.current_joint_positions.copy()
                rospy.loginfo(f"当前关节位置: {current_pos}")
            
            # 计算目标位置
            action = action.flatten()
            next_pos = (np.array(current_pos) + action * 0.1).tolist()
            rospy.loginfo(f"动作: {action}")
            rospy.loginfo(f"计算的目标位置: {next_pos}")
            
            # 生成和发送轨迹
            rospy.loginfo("开始生成轨迹...")
            positions, velocities, timestamps = self.generate_trajectory(
                current_pos, next_pos, duration=1.0
            )
            rospy.loginfo(f"生成了 {len(positions)} 个轨迹点")
            
            # 创建并发送轨迹消息
            rospy.loginfo("创建轨迹消息...")
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names
            
            for i in range(len(timestamps)):
                point = JointTrajectoryPoint()
                point.positions = positions[i]
                point.velocities = velocities[i]
                point.time_from_start = rospy.Duration(timestamps[i])
                traj_msg.points.append(point)
            
            # 发送轨迹
            rospy.loginfo("发送轨迹到action server...")
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = traj_msg
            self.arm_client.send_goal(goal)
            
            # 等待动作完成和状态更新
            rospy.loginfo("等待动作执行完成...")
            if not self.arm_client.wait_for_result(rospy.Duration(2.0)):
                rospy.logwarn("动作执行超时！")
                return None, 0, True, {}
            
            # 等待状态更新
            rospy.loginfo("等待状态更新...")
            if not self.wait_for_state_update(timeout=0.5):
                rospy.logwarn("等待状态更新超时")
                return None, 0, True, {}
            
            # 获取新状态并验证变化
            with self.position_lock:
                new_state = self.current_state.copy()
                rospy.loginfo(f"新的关节位置: {new_state[:7]}")
            
            state_change = np.linalg.norm(new_state[:7] - old_state[:7])
            rospy.loginfo(f"状态变化量: {state_change}")
            
            if state_change < 1e-6:
                rospy.logwarn("状态似乎没有更新！")
                rospy.logwarn(f"旧状态: {old_state[:7]}")
                rospy.logwarn(f"新状态: {new_state[:7]}")
                return None, 0, True, {}
            
            # 计算奖励
            rospy.loginfo("计算奖励...")
            reward = self.calculate_reward(action)
            rospy.loginfo(f"获得奖励: {reward}")
            
            rospy.loginfo("step函数执行完成")
            return new_state, reward, False, {}
            
        except Exception as e:
            rospy.logerr(f"执行动作时出错: {str(e)}")
            # 打印更详细的错误信息
            import traceback
            rospy.logerr(f"错误堆栈: {traceback.format_exc()}")
            return None, 0, True, {}
    
    def reset(self):
        """重置环境到初始状态"""
        try:
            # 创建初始轨迹消息
            cmd_msg = JointTrajectory()
            cmd_msg.joint_names = self.joint_names
            
            # 设置初始关节角度（确保使用float类型）
            initial_positions = [float(x) for x in [0.0, 0.0, 0.0, -1.736, 0.0, 0, 0]]
            
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
    
    def _verify_robot_state(self):
        """验证机器人状态"""
        try:
            # 发布一个小的测试动作
            test_action = np.array([0.1, 0.0, 0.0, -1.736, 0.0, 0.0, 0.0])
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

def train():
    """主训练循环"""
    # 初始化环境和智能体
    env = RobotEnv()
    agent = PPOAgent(env.state_dim, env.action_dim)
    
    # 验证机器人状态
    if not env._verify_robot_state():
        rospy.logerr("机器人状态验证失败，请检查控制器和传感器！")
        return
    
    # 训练参数
    max_episodes = 1000
    episode_rewards = []
    
    for ep in range(max_episodes):
        if ep % 10 == 0 and agent.episode_length > agent.min_ep_length:
            agent.episode_length = int(agent.episode_length * agent.decay_rate)
        
        # 重置环境并确保获得有效状态
        state = env.reset()
        if state is None:
            rospy.logerr("无法获取初始状态，跳过此回合")
            continue
            
        episode_reward = 0
        memory = []
        
        for step in range(agent.episode_length):
            try:
                # 确保状态是正确的格式
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
                # 获取动作
                action, _ = agent.policy(state_tensor)
                action = action.detach().squeeze(0).numpy()
                
                # 执行动作并获取下一个状态
                next_state, reward, done, _ = env.step(action)
                
                # 验证next_state是否有效
                if next_state is None:
                    rospy.logwarn("获取到无效的next_state，结束当前回合")
                    break
                
                # 存储经验
                memory.append((state, action, reward, next_state, done))
                
                # 更新状态和奖励
                state = next_state
                episode_reward += reward
                
                if done:
                    break
                
                if len(memory) >= agent.batch_size:
                    agent.update_policy(*zip(*memory))
                    memory = []
                
                print(f"Step {step}, Action: {action}, Reward: {reward:.2f}")
                    
            except Exception as e:
                rospy.logerr(f"训练步骤出错: {str(e)}")
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {ep}, Reward: {episode_reward:.2f}, Length: {agent.episode_length}")
    
    # 保存模型
    try:
        pth_path = os.path.join(pkg_path, 'scripts/RL/pth', 'arm_ppo.pth')
        torch.save(agent.policy.state_dict(), pth_path)
        print(f"模型已保存到: {pth_path}")
    except Exception as e:
        rospy.logerr(f"保存模型失败: {str(e)}")



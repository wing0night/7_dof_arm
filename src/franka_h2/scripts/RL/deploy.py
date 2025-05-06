#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import actionlib 
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import roboticstoolbox as rtb
import rospkg
import os

from PPONetwork import PPONetwork

class ArmController:
    def __init__(self):
        # 获取包路径
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('franka_h2')
        
        # 加载训练好的模型
        pth_path = os.path.join(pkg_path, 'scripts/RL/pth', 'arm_ppo.pth')
        self.policy = PPONetwork(state_dim=24, action_dim=7)
        try:
            self.policy.load_state_dict(torch.load(pth_path, map_location='cpu'))
            rospy.loginfo(f"成功加载模型从: {pth_path}")
        except Exception as e:
            rospy.logerr(f"加载模型失败: {str(e)}")
            raise
            
        self.policy.eval()  # 设置为评估模式
        
        # ROS配置
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 
                          'joint5', 'joint6', 'joint7']
        self.rate = rospy.Rate(50)  # 控制频率50Hz，与训练时一致
        
        # 初始化action client
        self.arm_client = actionlib.SimpleActionClient(
            'panda_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction
        )
        rospy.loginfo("等待action server...")
        self.arm_client.wait_for_server()
        rospy.loginfo("Action server连接成功")
        
        # 添加机器人模型
        self.panda = rtb.models.URDF.Panda()
        
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
            self.state_cb
        )
        
        # 等待第一次状态更新
        rospy.loginfo("等待接收第一帧关节状态...")
        while self.current_state is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("成功接收到关节状态！")
        
    def state_cb(self, msg):
        """状态回调函数"""
        try:
            # 提取关节状态
            joint_pos = np.array(msg.position[:7])
            joint_vel = np.array(msg.velocity[:7] if msg.velocity else [0.0]*7)
            joint_eff = np.array(msg.effort[:7] if msg.effort else [0.0]*7)
            
            # 计算末端执行器位置
            T = self.panda.fkine(joint_pos)
            end_effector_pos = np.array([T.t[0], T.t[1], T.t[2]])
            
            # 更新各个状态组件
            self.current_joint_positions = joint_pos
            self.current_joint_velocities = joint_vel
            self.current_joint_efforts = joint_eff
            self.current_ee_pos = end_effector_pos
            
            # 构建完整状态向量
            self.current_state = np.concatenate([
                joint_pos, joint_vel, joint_eff, end_effector_pos
            ])
            
        except Exception as e:
            rospy.logerr(f"状态回调处理错误: {str(e)}")
            
    def generate_trajectory(self, current_pos, next_pos, duration=3.0):
        """生成轨迹"""
        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        
        # 创建轨迹点
        point = JointTrajectoryPoint()
        point.positions = next_pos
        point.velocities = [0.0] * len(self.joint_names)
        point.time_from_start = rospy.Duration(duration)
        traj_msg.points.append(point)
        
        return traj_msg
        
    def execute_action(self, action):
        """执行动作"""
        try:
            if self.current_joint_positions is None:
                rospy.logwarn("当前关节位置未知")
                return False
                
            # 计算目标位置
            next_pos = (self.current_joint_positions + action * 0.1).tolist()
            
            # 生成轨迹
            traj_msg = self.generate_trajectory(
                self.current_joint_positions, 
                next_pos
            )
            
            # 发送动作
            goal = FollowJointTrajectoryGoal()
            goal.trajectory = traj_msg
            self.arm_client.send_goal(goal)
            
            # 等待动作完成
            if not self.arm_client.wait_for_result(rospy.Duration(5.0)):
                rospy.logwarn("动作执行超时！")
                return False
                
            return True
            
        except Exception as e:
            rospy.logerr(f"执行动作时出错: {str(e)}")
            return False
            
    def control_loop(self):
        """主控制循环"""
        rospy.loginfo("开始控制循环...")
        
        while not rospy.is_shutdown():
            try:
                if self.current_state is not None:
                    # 生成动作
                    state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)
                    with torch.no_grad():
                        action, _ = self.policy(state_tensor)
                        action = action.squeeze(0).numpy()
                        
                    # 执行动作
                    rospy.loginfo(f"执行动作: {action}")
                    success = self.execute_action(action)
                    
                    if not success:
                        rospy.logwarn("动作执行失败")
                        
                self.rate.sleep()
                
            except Exception as e:
                rospy.logerr(f"控制循环出错: {str(e)}")
                rospy.sleep(1.0)  # 错误发生时等待一段时间

if __name__ == "__main__":
    try:
        rospy.init_node('rl_arm_controller')
        controller = ArmController()
        controller.control_loop()
    except rospy.ROSInterruptException:
        pass
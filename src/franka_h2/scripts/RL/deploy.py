#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


from PPONetwork import PPONetwork

class ArmController:
    def __init__(self):
        # 加载训练好的模型
        self.policy = PPONetwork(state_dim=24, action_dim=7)
        self.policy.load_state_dict(torch.load("arm_ppo.pth", map_location='cpu'))
        self.policy.eval() # 设置为模型评估模式
        
        # ROS配置
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 
                          'joint5', 'joint6', 'joint7']
        self.rate = rospy.Rate(30)  # 控制频率30Hz
        
        # 订阅者和发布者
        self.state_sub = rospy.Subscriber('/joint_states', JointState, self.state_cb)
        self.cmd_pub = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
        
        # 状态缓存
        self.current_state = None
        
    def state_cb(self, msg):
        """状态预处理（与训练时保持一致）"""
        # 解析传感器数据
        joint_pos = np.array(msg.position)
        joint_vel = np.array(msg.velocity)
        joint_eff = np.array(msg.effort)
        end_pos = np.zeros(3)  # 需替换实际正运动学计算
        
        # 组合状态向量
        self.current_state = np.concatenate([
            joint_pos, joint_vel, joint_eff, end_pos
        ])
        
    def generate_action(self):
        """生成控制指令"""
        if self.current_state is None:
            return None
            
        # 转换为Tensor并推理
        state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy(state_tensor)
            
        # 后处理（示例为速度控制）
        return action.numpy().flatten()
        
    def control_loop(self):
        """主控制循环"""
        while not rospy.is_shutdown():
            if self.current_state is not None:
                # 生成控制动作
                action = self.generate_action()
                
                # 创建控制指令
                cmd_msg = JointTrajectory()
                cmd_msg.joint_names = self.joint_names
                point = JointTrajectoryPoint()
                point.velocities = action.tolist()
                point.time_from_start = rospy.Duration(0.033)  # 30Hz周期
                cmd_msg.points.append(point)
                
                # 发布指令
                self.cmd_pub.publish(cmd_msg)
                
            self.rate.sleep()

if __name__ == "__main__":
    try:
        rospy.init_node('rl_arm_controller')
        controller = ArmController()
        controller.control_loop()
    except rospy.ROSInterruptException:
        pass

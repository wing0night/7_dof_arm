#!/usr/bin/env python
import rospy
import matplotlib.pyplot as plt
import numpy as np
from sensor_msgs.msg import JointState

from franka_h2.msg import TrajectoryData

import rospkg
import os

class JointStateLogger:
    def __init__(self):
        rospy.init_node('joint_state_logger')
        
        # 初始化数据存储（保持原有结构）
        self.time_data = []
        self.velocity_data = {f'joint{i+1}': [] for i in range(7)}
        self.torque_data = {f'joint{i+1}': [] for i in range(7)}
        self.acc_data = {f'joint{i+1}': [] for i in range(7)}
        self.position_data = {f'joint{i+1}': [] for i in range(7)}
        self.joint_names = []
        
        # 订阅话题（保持原有）
        self.sub = rospy.Subscriber('/trajectory_data', TrajectoryData, self.traj_callback)
        rospy.on_shutdown(self.save_plots)

    def traj_callback(self, msg):
        # 保持原有数据收集逻辑不变
        if not self.joint_names:
            self.joint_names = sorted([name for name in msg.joint_names if 'joint' in name])
            
        current_time = msg.stamp.to_sec()
        self.time_data.append(current_time)
        
        for i, name in enumerate(self.joint_names):
            try:
                idx = msg.joint_names.index(name)
                self.velocity_data[name].append(msg.velocities[idx])
                self.torque_data[name].append(msg.torques[idx])
                self.acc_data[name].append(msg.acc[idx])
                self.position_data[name].append(msg.positions[idx])
            except (ValueError, IndexError) as e:
                self.velocity_data[name].append(0.0)
                self.torque_data[name].append(0.0)
                self.acc_data[name].append(0.0)
                self.position_data[name].append(0.0)

    def save_plots(self):
        time_array = np.array(self.time_data) - self.time_data[0]
        
        # 创建颜色映射（使用tab10色系，支持10种颜色）
        colors = plt.cm.tab10.colors[:7]  # 取前7种颜色
        
        # 创建3个子图（速度、加速度、力矩）
        plt.figure(figsize=(20, 15))

        # 位置子图
        plt.subplot(4, 1, 1)
        for i, (name, data) in enumerate(self.position_data.items()):
            plt.plot(time_array, data, 
                    color=colors[i],
                    label=f'Joint {i+1}')
        plt.ylabel('Position (rad)')
        plt.title('All Joints Position')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 速度子图
        plt.subplot(4, 1, 2)
        for i, (name, data) in enumerate(self.velocity_data.items()):
            plt.plot(time_array, data, 
                    color=colors[i],
                    label=f'Joint {i+1}')
        plt.ylabel('Velocity (rad/s)')
        plt.title('All Joints Velocity')
        plt.grid(True)
        
        # 加速度子图
        plt.subplot(4, 1, 3)
        for i, (name, data) in enumerate(self.acc_data.items()):
            plt.plot(time_array, data,
                    color=colors[i],
                    label=f'Joint {i+1}')
        plt.ylabel('Acceleration (rad/s²)')
        plt.title('All Joints Acceleration')
        plt.grid(True)
        
        # 力矩子图
        plt.subplot(4, 1, 4)
        for i, (name, data) in enumerate(self.torque_data.items()):
            plt.plot(time_array, data,
                    color=colors[i],
                    label=f'Joint {i+1}')
        plt.ylabel('Torque (Nm)')
        plt.xlabel('Time (s)')
        plt.title('All Joints Torque')
        plt.grid(True)
        
        # 统一调整布局
        plt.tight_layout()
        
        # 修改保存路径
        rospack = rospkg.RosPack()
        save_path = os.path.join(
            rospack.get_path("franka_h2"),
            "plots",
            "all_joints_plot.png"
        )
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved combined plot to {save_path}")

if __name__ == "__main__":
    logger = JointStateLogger()
    rospy.spin()
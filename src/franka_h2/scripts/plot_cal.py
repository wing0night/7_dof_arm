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
        
        # 初始化数据存储
        self.time_data = []
        self.velocity_data = {f'joint{i+1}': [] for i in range(7)}
        self.torque_data = {f'joint{i+1}': [] for i in range(7)}  # 新增扭矩数据存储
        self.acc_data = {f'joint{i+1}': [] for i in range(7)}  # 新增加速度数据存储
        self.joint_names = []
        
        # 订阅话题
        self.sub = rospy.Subscriber('/trajectory_data', TrajectoryData, self.traj_callback)
        
        # 注册关闭钩子
        rospy.on_shutdown(self.save_plots)

    def traj_callback(self, msg):
        if not self.joint_names:
            self.joint_names = sorted([name for name in msg.joint_names if 'joint' in name])
            
        current_time = msg.stamp.to_sec()
        self.time_data.append(current_time)
        
        for i, name in enumerate(self.joint_names):
            try:
                idx = msg.joint_names.index(name)
                # 同时记录速度和扭矩
                self.velocity_data[name].append(msg.velocities[idx])
                self.torque_data[name].append(msg.torques[idx])  # 假设消息已包含torques字段
                self.acc_data[name].append(msg.acc[idx])
            except (ValueError, IndexError) as e:
                self.velocity_data[name].append(0.0)
                self.torque_data[name].append(0.0)
                self.acc_data[name].append(0.0)

    def save_plots(self):
        # 转换为相对时间
        time_array = np.array(self.time_data) - self.time_data[0]
        
        # 创建3列画布
        plt.figure(figsize=(30, 35))
        
        for i, (name, velocities) in enumerate(self.velocity_data.items(), 1):  # 从1开始计数
            # 速度子图（左列）
            plt.subplot(7, 3, 3*i-2)
            plt.plot(time_array, velocities, 'b')
            plt.ylabel('Velocity (rad/s)')
            plt.title(f'Joint {name[-1]} Velocity')
            plt.grid(True)
            if i != 7: plt.xticks([])
            
            # jiasudu子图（右列）
            plt.subplot(7, 3, 3*i-1)
            plt.plot(time_array, self.acc_data[name], 'r')
            plt.ylabel('Acceleration (Nm)')
            plt.title(f'Joint {name[-1]} Acceleration')            
            plt.grid(True)
            if i != 7: plt.xticks([])

            # 力矩子图（右列）
            plt.subplot(7, 3, 3*i)
            plt.plot(time_array, self.torque_data[name], 'g')
            plt.ylabel('Torque (Nm)')
            plt.title(f'Joint {name[-1]} Torque')
            plt.grid(True)
            if i != 7: plt.xticks([])
        
        # 设置公共标签
        plt.subplot(7,3,19)  # 最下方左侧
        plt.xlabel('Time (s)')
        plt.subplot(7,3,20)  # 最下方右侧 
        plt.xlabel('Time (s)')
        plt.subplot(7,3,21)  # 最下方右侧 
        plt.xlabel('Time (s)')
        
        plt.tight_layout()
        
        # 保存路径设置
        rospack = rospkg.RosPack()
        save_path = os.path.join(
            rospack.get_path("franka_h2"),
            "results",
            "joint_states_plot.png"
        )
        plt.savefig(save_path)
        print(f"Saved combined plot to {save_path}")

if __name__ == "__main__":
    logger = JointStateLogger()
    rospy.spin()
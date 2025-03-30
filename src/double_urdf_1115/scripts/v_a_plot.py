#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import JointState
from collections import defaultdict
import os
import time
import math as m

class JointMonitor:
    def __init__(self):
        # 数据存储结构
        self.data = defaultdict(lambda: {
            'time': [],
            'position': [],
            'velocity': [],
            'acceleration': []
        })
        self.last_velocity = {}
        self.last_time = None
        self.loss = []

        # para for calculate loss function
        self.lambda_v = 1
        self.lambda_acc = 1

    def joint_states_callback(self, msg):
        current_time = rospy.get_time()
        dt = current_time - self.last_time if self.last_time else 0

        print(msg)

        v_sq_sum = 0
        a_sq_sum = 0

        for i, name in enumerate(msg.name):
            # 获取当前状态
            position = msg.position[i]
            velocity = msg.velocity[i] if i < len(msg.velocity) else 0.0

            # 计算加速度
            acceleration = 0.0
            if name in self.last_velocity and dt > 0:
                acceleration = (velocity - self.last_velocity[name]) / dt

            # 记录数据
            self.data[name]['time'].append(current_time)
            self.data[name]['position'].append(position)
            self.data[name]['velocity'].append(velocity)
            self.data[name]['acceleration'].append(acceleration)

            v_sq_sum = v_sq_sum + m.pow(velocity, 2)
            a_sq_sum = a_sq_sum + m.pow(acceleration, 2)
            f_s = self.lambda_v*v_sq_sum+self.lambda_acc*a_sq_sum

            if name=="joint14":
                self.loss.append(f_s)
                v_sq_sum = 0
                a_sq_sum = 0

            # 更新上一次记录
            self.last_velocity[name] = velocity

        self.last_time = current_time



    def plot_data(self):
        plt.figure(figsize=(12, 8))
        # for joint in self.data:
        # for joint in self.data:
        #     # 时间归一化（从0开始）
        #     t = np.array(self.data[joint]['time']) - self.data[joint]['time'][0]
            
        #     # 绘制三条曲线
        #     plt.subplot(3, 1, 1)
        #     plt.plot(t, self.data[joint]['position'], label=joint)
        #     plt.ylabel('Position (rad)')
            
        #     plt.subplot(3, 1, 2)
        #     plt.plot(t, self.data[joint]['velocity'])
        #     plt.ylabel('Velocity (rad/s)')
            
        #     plt.subplot(3, 1, 3)
        #     plt.plot(t, self.data[joint]['acceleration'])
        #     plt.ylabel('Acceleration (rad/s²)')
        #     plt.xlabel('Time (s)')
        
        joint = "joint7"
        # 获取当前 ROS 时间戳（支持仿真时间和系统时间）
        ros_time = rospy.Time.now().to_sec()  # 统一时间源
        local_time = time.localtime(ros_time)  # 转换为本地时间结构体
        # 格式化为 yy_mm_dd_hh_mm_ss
        time_str = time.strftime("%y_%m_%d_%H_%M_%S", local_time)

        t = np.array(self.data[joint]['time']) - self.data[joint]['time'][0]
            
        # 绘制三条曲线
        plt.subplot(4, 1, 1)
        plt.plot(t, self.data[joint]['position'], label=joint)
        plt.ylabel('Position (rad)')
        
        plt.subplot(4, 1, 2)
        plt.plot(t, self.data[joint]['velocity'])
        plt.ylabel('Velocity (rad/s)')
        
        plt.subplot(4, 1, 3)
        plt.plot(t, self.data[joint]['acceleration'])
        plt.ylabel('Acceleration (rad/s²)')
        plt.xlabel('Time (s)')

        plt.subplot(4, 1, 4)
        plt.plot(t, self.loss)
        plt.ylabel('loss')
        plt.xlabel('Time (s)')

        plt.suptitle('Joint States Analysis')
        plt.legend()
        # 获取并打印完整保存路径
        save_path = os.path.join('/home/night/Desktop/catkin_ws/src/double_urdf_1115/plot', 'plot_'+joint+'_'+time_str+'.png')
        print("Saving plot...")
        print(f"\n\033[1;34m[INFO] 即将保存图表至：{save_path}\033[0m")
        plt.savefig(save_path)
        print("[INFO] Plot saved. Closing figure...")
        print(self.loss)
        print(t)
        plt.close('all')

if __name__ == '__main__':
    rospy.init_node('joint_monitor')
    # rospy.init_node('velocity_monitor')
    monitor = JointMonitor()
    rospy.Subscriber("/joint_states", JointState, monitor.joint_states_callback)
    
    # 注册关闭时保存绘图
    rospy.on_shutdown(monitor.plot_data)
    rospy.loginfo("Joint Monitor Started. Press Ctrl+C to save plot.")
    rospy.spin()




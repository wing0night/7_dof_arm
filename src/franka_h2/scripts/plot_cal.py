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
        
        # 初始化原有数据存储
        self.time_data = []
        self.velocity_data = {f'joint{i+1}': [] for i in range(7)}
        self.torque_data = {f'joint{i+1}': [] for i in range(7)}
        self.acc_data = {f'joint{i+1}': [] for i in range(7)}
        self.position_data = {f'joint{i+1}': [] for i in range(7)}
        
        # 添加目标函数数据存储
        self.smoothness_objective = []  # 平滑性目标
        self.energy_objective = []      # 能量消耗目标

        self.prev_positions = None
        self.prev_time = None
        # 获取包路径以读取URDF
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('franka_h2')
        self.urdf_path = os.path.join(self.pkg_path, 'urdf', 'panda_robot_gazebo.urdf')
        
        # 加载URDF获取惯性参数
        from urdf_parser_py.urdf import URDF
        self.robot = URDF.from_xml_file(self.urdf_path)
        self.inertia_params = self._load_inertia_params()
        
        self.joint_names = []
        self.sub = rospy.Subscriber('/trajectory_data', TrajectoryData, self.traj_callback)
        rospy.on_shutdown(self.save_plots)
    
    def _load_inertia_params(self):
        """从URDF中提取惯性参数"""
        inertia_params = {}
        joint_names = [f'joint{i+1}' for i in range(7)]
        
        for name in joint_names:
            for joint in self.robot.joints:
                if joint.name == name:
                    child_link = self.robot.link_map[joint.child]
                    if child_link.inertial:
                        inertia = child_link.inertial.inertia
                        inertia_matrix = [
                            [inertia.ixx, inertia.ixy, inertia.ixz],
                            [inertia.ixy, inertia.iyy, inertia.iyz],
                            [inertia.ixz, inertia.iyz, inertia.izz]
                        ]
                        inertia_params[name] = {
                            'mass': child_link.inertial.mass,
                            'inertia_matrix': inertia_matrix,
                            'com_position': child_link.inertial.origin.xyz if child_link.inertial.origin else [0,0,0]
                        }
                    break
        return inertia_params

    def calculate_objectives(self, positions, velocities, accelerations, torques, current_time):
        """按照NSGA-II中的实际计算逻辑计算目标函数"""
        if self.prev_positions is None:
            self.prev_positions = {k: [v[0]] for k, v in positions.items()}
            self.prev_time = current_time
            return 0.0, 0.0

        total_smoothness = 0.0
        total_energy = 0.0
        dt = current_time - self.prev_time

        for j in range(7):
            joint_name = f'joint{j+1}'
            
            # 获取关节参数
            params = self.inertia_params[joint_name]
            I_j = params['inertia_matrix'][2][2]  # 绕Z轴的转动惯量
            
            # 计算轨迹参数
            q_start = self.prev_positions[joint_name][-1]
            q_end = positions[joint_name][-1]
            v_start = velocities[joint_name][-1]
            
            # 计算多项式系数
            delta_q = q_end - q_start
            T = dt  # 使用实际的时间间隔
            
            a2 = (3*delta_q - 2*v_start*T) / (T**2)
            a3 = (-2*(delta_q - v_start*T/2)) / (T**3)
            
            # 计算平滑性指标
            integral_acc_sq = 4*a2**2*T + 12*a2*a3*T**2 + 12*a3**2*T**3
            total_smoothness += integral_acc_sq
            
            # 计算能量指标
            tau = torques[joint_name][-1]
            integral_tau_sq = (I_j**2 * integral_acc_sq) + (2*I_j*tau*(-v_start)) + (tau**2 * T)
            total_energy += integral_tau_sq
            
        # 更新前一时刻的位置
        self.prev_positions = {k: [v[-1]] for k, v in positions.items()}
        self.prev_time = current_time
        
        return total_smoothness, total_energy

    def traj_callback(self, msg):
        # 原有数据收集逻辑
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
        
        # 计算目标函数值
        if len(self.time_data) > 1:
            dt = self.time_data[-1] - self.time_data[-2]
            current_time = msg.stamp.to_sec()
            smoothness, energy = self.calculate_objectives(
                self.position_data,
                self.velocity_data,
                self.acc_data,
                self.torque_data,
                current_time
            )
            self.smoothness_objective.append(smoothness)
            self.energy_objective.append(energy)
        else:
            self.smoothness_objective.append(0.0)
            self.energy_objective.append(0.0)

    def save_plots(self):
        time_array = np.array(self.time_data) - self.time_data[0]
        colors = plt.cm.tab10.colors[:7]  # 取前7种颜色
        
        # 创建5个子图（位置、速度、加速度、力矩、目标函数）
        plt.figure(figsize=(20, 20))

        # 位置子图
        plt.subplot(5, 1, 1)
        for i, (name, data) in enumerate(self.position_data.items()):
            plt.plot(time_array, data, color=colors[i], label=f'Joint {i+1}')
        plt.ylabel('Position (rad)')
        plt.title('All Joints Position')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 速度子图
        plt.subplot(5, 1, 2)
        for i, (name, data) in enumerate(self.velocity_data.items()):
            plt.plot(time_array, data, color=colors[i], label=f'Joint {i+1}')
        plt.ylabel('Velocity (rad/s)')
        plt.title('All Joints Velocity')
        plt.grid(True)
        
        # 加速度子图
        plt.subplot(5, 1, 3)
        for i, (name, data) in enumerate(self.acc_data.items()):
            plt.plot(time_array, data, color=colors[i], label=f'Joint {i+1}')
        plt.ylabel('Acceleration (rad/s²)')
        plt.title('All Joints Acceleration')
        plt.grid(True)
        
        # 力矩子图
        plt.subplot(5, 1, 4)
        for i, (name, data) in enumerate(self.torque_data.items()):
            plt.plot(time_array, data, color=colors[i], label=f'Joint {i+1}')
        plt.ylabel('Torque (Nm)')
        plt.title('All Joints Torque')
        plt.grid(True)
        
        # 目标函数子图
        plt.subplot(5, 1, 5)
        
        plt.plot(time_array, self.smoothness_objective, 
                label='Smoothness Objective', color='blue')
        plt.plot(time_array, self.energy_objective, 
                label='Energy Objective', color='red')
        plt.ylabel('Objective Value')
        plt.xlabel('Time (s)')
        plt.title('NSGA-II Objective Functions')
        plt.grid(True)
        plt.legend()
        
        # 统一调整布局
        plt.tight_layout()
        
        # 保存图像
        rospack = rospkg.RosPack()
        save_path = os.path.join(
            rospack.get_path("franka_h2"),
            "plots",
            "all_joints_with_objectives_plot.png"
        )
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved combined plot to {save_path}")

if __name__ == "__main__":
    logger = JointStateLogger()
    rospy.spin()
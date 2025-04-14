#!/usr/bin/env python3

from __future__ import print_function 
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

from RRT_Planner import RRTPlanner

from threading import Lock

class LM_ik_solver():
    def __init__(self):
        self.position_lock = Lock()  # 关节状态访问锁

        # 初始化标准轨迹动作客户端
        self.arm_client = actionlib.SimpleActionClient(
            'panda_controller/follow_joint_trajectory', 
            FollowJointTrajectoryAction
        )
        self.arm_client.wait_for_server()

        self.arm_client.done_cb = self._handle_action_done  # 注册回调

        # 获取包路径
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('franka_h2')

        # 构建URDF文件路径
        self.urdf_path = os.path.join(pkg_path, 'urdf', 'panda_robot_gazebo.urdf')

        # 加载URDF并解析惯性参数
        self.robot = URDF.from_xml_file(self.urdf_path)
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.inertia_params = self._load_inertia_params()
        self.prev_velocities = None
        self.prev_time = None

        # 自定义轨迹的发布器
        self.traj_pub = rospy.Publisher('/trajectory_data', TrajectoryData, queue_size=10)
        # 标准轨迹的发布器（可选，根据需求选择动作或话题）
        self.std_traj_pub = rospy.Publisher('/joint_trajectory', JointTrajectory, queue_size=10)

        # 获取当前关节状态
        self.current_joint_positions = None
        rospy.Subscriber('/joint_states', JointState, self.joint_state_callback)
        rospy.wait_for_message('/joint_states', JointState)

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
        self.rrt_planner = RRTPlanner(self.joint_limits, step_size=0.3)

    def __del__(self):
        self.arm_client.cancel_all_goals()  # 清理未完成的目标
        rospy.sleep(0.5)
    
    def _handle_action_done(self, status, result):
        # try:
        #     if status == actionlib.GoalStatus.SUCCEEDED:
        #         if not hasattr(result, 'actual') or not hasattr(result.actual, 'positions'):
        #             raise ValueError("结果缺少必要字段")
        #         with self.position_lock:
        #             self.current_joint_positions = result.actual.positions
        #         rospy.loginfo("动作成功完成")
        #     else:
        #         rospy.logwarn(f"动作未成功，状态码: {status}")
        # except Exception as e:
        #     rospy.logerr(f"处理回调时发生错误: {str(e)}")
                with self.position_lock:
                    self.current_joint_positions = result.actual.positions

    def joint_state_callback(self, msg):
        with self.position_lock:
            self.current_joint_positions = msg.position

    def _validate_trajectory(self, positions):
        """轨迹安全性检查"""
        for j in range(7):
            if not (self.joint_limits[j][0] <= positions[j] <= self.joint_limits[j][1]):
                rospy.logerr(f"关节{j+1}超出限位: {positions[j]:.2f}")
                return False
        return True
    
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
    
    def trapezoidal_interpolation(self, start, goal, duration, freq=50):
        """梯形速度曲线插值（三段式）"""
        num_points = int(duration * freq)
        t = np.linspace(0, duration, num_points)
        positions = []
        velocities = []
        
        delta = goal - start  # 总位移
        t_acc = t_dec = duration / 3.0  # 加速/减速段时间（各占1/3）
        t_const = duration - t_acc - t_dec  # 匀速段时间
        
        # 计算运动参数
        if abs(duration) > 1e-6:  # 防止除以0
            a = (9.0 * delta) / (2.0 * duration**2)  # 加速度计算
            v_max = a * t_acc  # 最大速度
        else:
            a = 0.0
            v_max = 0.0
        
        for ti in t:
            if ti < t_acc:  # 加速段
                phase_pos = 0.5 * a * ti**2
                phase_vel = a * ti
            elif ti < t_acc + t_const:  # 匀速段
                phase_pos = 0.5 * a * t_acc**2 + v_max * (ti - t_acc)
                phase_vel = v_max
            else:  # 减速段
                t_dec_phase = ti - (t_acc + t_const)
                phase_pos = (0.5 * a * t_acc**2 + 
                            v_max * t_const + 
                            v_max * t_dec_phase - 
                            0.5 * a * t_dec_phase**2)
                phase_vel = v_max - a * t_dec_phase
            
            # 计算绝对位置和速度
            positions.append(start + phase_pos)
            velocities.append(phase_vel)
        
        return positions, velocities, t
    
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

    def move_to_goal(self, goal_positions, duration=5.0):
        # 实时获取最新关节状态（阻塞直到获得有效数据）
        while self.current_joint_positions is None and not rospy.is_shutdown():
            rospy.loginfo("等待关节状态...")
            rospy.sleep(0.1)
        
        # 使用深拷贝避免原始数据被修改
        start_positions = list(self.current_joint_positions)  # 从当前实际位置开始
        goal_positions = list(goal_positions)

        # 参数维度验证
        if len(start_positions) != 7 or len(goal_positions) !=7:
            rospy.logerr("关节位置维度必须为7")
            return

        num_points = int(duration * 50)
        timestamps = np.linspace(0, duration, num_points)

        # 生成插值轨迹（实时计算）
        all_positions = []
        all_velocities = []
        for joint_idx in range(7):  # 固定处理7个关节
            # 使用梯形插值生成轨迹
            positions, velocities, _ = self.quintic_interpolation(
                start=start_positions[joint_idx],
                goal=goal_positions[joint_idx],
                duration=duration,
                freq=50
            )
            all_positions.append(positions)
            all_velocities.append(velocities)

        # ========== 标准轨迹消息构建（带时间累加） ==========
        std_traj_msg = JointTrajectory()
        std_traj_msg.joint_names = self.joint_names
        
        # 获取轨迹开始时间（带时间连续性）
        start_time = rospy.Time.now() + rospy.Duration(0.1)  # 留出100ms准备时间
        
        for i in range(num_points):
            point = JointTrajectoryPoint()
            point.positions = [all_positions[j][i] for j in range(7)]
            point.velocities = [all_velocities[j][i] for j in range(7)]
            point.time_from_start = rospy.Duration(timestamps[i])
            std_traj_msg.points.append(point)
        
        # 发送标准轨迹（带结果回调）
        goal = FollowJointTrajectoryGoal(trajectory=std_traj_msg)
        self.arm_client.send_goal(
            goal,
            done_cb=lambda status, result: self._update_current_position(status, result, goal_positions)
        )
        
        # ========== 自定义轨迹消息构建（实时更新） ==========
        rate = rospy.Rate(50)  # 50Hz
        for i in range(num_points):
            # 实时获取最新关节状态（非阻塞）
            current_pos = self.current_joint_positions
            
            # 构造自定义消息
            target_msg = TrajectoryData()
            target_msg.joint_names = self.joint_names
            target_msg.positions = [all_positions[j][i] for j in range(7)]
            target_msg.velocities = [all_velocities[j][i] for j in range(7)]
            
            # 加速度计算（带状态更新）
            if i == 0:
                accelerations = [0.0]*7
            else:
                dt = timestamps[i] - timestamps[i-1]
                accelerations = [
                    (target_msg.velocities[j] - all_velocities[j][i-1])/dt 
                    for j in range(7)
                ]
            target_msg.acc = accelerations
            
            # 力矩计算（带实时参数）
            gravity_torques = self.calculate_gravity_torques()
            target_msg.torques = [
                np.dot(
                    self.inertia_params[self.joint_names[j]]['inertia_matrix'],
                    [0, 0, accelerations[j]]
                )[2] + gravity_torques[self.joint_names[j]]
                for j in range(7)
            ]
            
            # 时间戳对齐标准轨迹
            target_msg.stamp = start_time + rospy.Duration(timestamps[i])
            
            # 发布消息
            self.traj_pub.publish(target_msg)
            rate.sleep()

        # 等待动作执行完成
        if not self.arm_client.wait_for_result(rospy.Duration(duration+1)):
            rospy.logwarn("轨迹执行超时")
        
        self.current_joint_positions = goal_positions  # 解除注释此行

    def _update_current_position(self, status, result, target_positions):
        """处理动作执行完成回调"""
        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("轨迹执行成功，更新关节状态")
            with self.position_lock:
                self.current_joint_positions = target_positions  # 解除注释此行
        else:
            rospy.logwarn(f"轨迹执行失败，状态码: {status}")
            if hasattr(result, 'error_code'):
                rospy.logerr(f"错误代码: {result.error_code}")
    
    def plan_and_execute(self, goal_positions):
        # 验证输入
        if len(goal_positions) != 7:
            rospy.logerr("目标位置维度错误")
            return

        # 获取当前状态
        if self.current_joint_positions is None:
            rospy.logerr("未获取当前关节状态")
            return

        # RRT路径规划
        path = self.rrt_planner.plan(
            start_config=np.array(self.current_joint_positions),
            goal_config=np.array(goal_positions)
        )

        if path is None:
            return

        # 路径后处理（可根据需要插入B样条曲线平滑）
        self._execute_path(path)

    def _execute_path(self, path):
        """执行分段轨迹（同步模式）"""
        for i in range(len(path)-1):
            goal = path[i+1]
            
            # 发送当前段轨迹
            self.move_to_goal(goal, duration=2.0)
            
            # 阻塞等待执行完成
            if not self.arm_client.wait_for_result(rospy.Duration(2.0 + 1)):  # 2秒轨迹+1秒缓冲
                rospy.logwarn(f"第{i+1}段轨迹执行超时")
                return

panda = rtb.models.URDF.Panda()
# print(panda)

T = panda.fkine(panda.qz, end='panda_hand')
#print(T)

# x = float(input("Enter X Co-ordinate: "))
# y = float(input("Enter Y Co-ordinate: "))
# z = float(input("Enter Z Co-ordinate: "))

# point = SE3(x,y,z)
# point_sol = panda.ikine_LM(point)
# print("IK Solution: ",point_sol.q)

# 定义目标位置和旋转
x = 0.5
y = 0.2
z = 0.5
R_custom = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
])
# R_custom = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ])

# 创建目标位姿
T_goal = SE3.Rt(R_custom, [x, y, z])

# 逆运动学解算  
point_sol = panda.ikine_LM(T_goal)
print("IK Solution: ", point_sol.q)


if __name__ == '__main__':

  try:

    rospy.init_node('rrt_planner_demo')

    solve_ik = LM_ik_solver()
    #move_robot_arm([point_sol[0] , point_sol[1] , point_sol[2] , point_sol[3] , point_sol[4]])
    # solve_ik.move_robot_arm(point_sol.q)

    solve_ik.plan_and_execute(point_sol.q)

    # solve_ik.move_to_goal(point_sol.q, duration=5.0)
    
    print("Robotic arm has successfully reached the goal!")
     
  except rospy.ROSInterruptException:
    print("Program interrupted before completion.", file=sym.stderr)





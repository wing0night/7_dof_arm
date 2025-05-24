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

from gazebo_msgs.srv import GetModelState, GetLinkState, GetJointProperties

from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from franka_h2.msg import TrajectoryData

from threading import Lock
from urdf_parser_py.urdf import URDF
from fk import cal_fk
from extract_pose import extract_pose

class ArmController:
    def __init__(self):
        self.position_lock = Lock()  # 关节状态访问锁

        # 获取包路径
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('franka_h2')
        
        # 加载训练好的模型
        pth_path = os.path.join(pkg_path, 'scripts/RL/pth', 'arm_ppo.pth')
        self.policy = PPONetwork(state_dim=28, action_dim=7)
        try:
            self.policy.load_state_dict(torch.load(pth_path, map_location='cpu'))
            rospy.loginfo(f"成功加载模型从: {pth_path}")
        except Exception as e:
            rospy.logerr(f"加载模型失败: {str(e)}")
            raise
            
        self.policy.eval()  # 设置为评估模式

        self.dt = 3.0

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

        # 这里可以是自定义消息的发布器用来绘图
        self.traj_pub = rospy.Publisher('/trajectory_data', TrajectoryData, queue_size=10)
        
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

        # 定义目标位置和旋转
        self.x = 0.5
        self.y = 0.2
        self.z = 0.5
        self.R_custom = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])

        self.max_joint_velocity = 3.14

        # # 创建4x4单位矩阵
        # self.T_tar = np.eye(4)
        # # 填入旋转部分 (左上3x3)
        # self.T_tar[:3, :3] = self.R_custom
        # # 填入平移部分 (前三行第四列)
        # self.T_tar[:3, 3] = [self.x, self.y, self.z]
        # self.T_tar = cal_fk([2.22601256 , 1.2024973 , -2.72513796 ,-2.21730977, -2.19911507 , 0.1795953,  0])
        self.T_tar = cal_fk(np.array([1, 0, 0, -2, 0, 0, 0]))

        # 构建URDF文件路径
        self.urdf_path = os.path.join(pkg_path, 'urdf', 'panda_robot_gazebo.urdf')
        # 加载URDF并解析惯性参数
        self.robot = URDF.from_xml_file(self.urdf_path)
        self.inertia_params = self._load_inertia_params()

        self.velocity_pub = rospy.Publisher(
            '/arm_controller/command',  # 示例topic
            Float64MultiArray, 
            queue_size=1
        )
        
        # 订阅者
        self.state_sub = rospy.Subscriber(
            '/joint_states', 
            JointState, 
        )

        self.prev_velocities = None
        self.prev_time = None

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
        
        # 等待第一次状态更新
        # rospy.loginfo("等待接收第一帧关节状态...")
        # while self.current_state is None and not rospy.is_shutdown():
        #     rospy.sleep(0.1)
        # rospy.loginfo("成功接收到关节状态！")
        
    # def state_cb(self, msg):
    #     """状态回调函数"""
    #     try:
    #         # 提取关节状态
    #         joint_pos = np.array(msg.position[:7])
    #         joint_vel = np.array(msg.velocity[:7] if msg.velocity else [0.0]*7)
    #         joint_eff = np.array(msg.effort[:7] if msg.effort else [0.0]*7)
            
    #         # 计算末端执行器位置
    #         T = self.panda.fkine(joint_pos)
    #         end_effector_pos = np.array([T.t[0], T.t[1], T.t[2]])
            
    #         # 更新各个状态组件
    #         self.current_joint_positions = joint_pos
    #         self.current_joint_velocities = joint_vel
    #         self.current_joint_efforts = joint_eff
    #         self.current_ee_pos = end_effector_pos
            
    #         # 构建完整状态向量
    #         self.current_state = np.concatenate([
    #             joint_pos, joint_vel, joint_eff, end_effector_pos
    #         ])
            
    #     except Exception as e:
    #         rospy.logerr(f"状态回调处理错误: {str(e)}")

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

    def get_gazebo_joint_state(self, joint_name):
        """获取指定关节的状态"""
        try:
            # 获取关节属性
            resp = self.get_joint_properties(f"{joint_name}")
            # print(resp)
            if resp.success:
                print(f"获取关节 {joint_name} 状态成功：", resp)
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
                    self.current_joint_positions,
                    self.current_joint_velocities,
                    # self.current_joint_efforts,
                    # self.current_ee_pos
                    self.position_ee,
                    self.axis_angle,
                    self.position_ee_tar,
                    self.axis_angle_tar
                ])
                
                self.state_updated = True
                self.last_update_time = rospy.Time.now()
                
            return True
            
        except Exception as e:
            rospy.logerr(f"从Gazebo更新状态时出错: {str(e)}")
            return False
            
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
        
    def execute_action(self, action, step):
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

            timestamps = [t + step * self.dt for t in timestamps]
            
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
            
            
            # 11. 终止判断
            done = True
            
            return done
            
        except Exception as e:
            rospy.logerr(f"执行步骤时出错: {str(e)}")
            return None, 0, True, {}
            
    def control_loop(self):
        """主控制循环"""
        rospy.loginfo("开始控制循环...")
        step = 0
        
        while not rospy.is_shutdown():
            try:

                self.update_state_from_gazebo()
                if self.current_state is not None:
                    # 生成动作
                    state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)
                    with torch.no_grad():
                        action, _ = self.policy(state_tensor)
                        action = action.squeeze(0).numpy()
                        
                    # 执行动作
                    rospy.loginfo(f"执行动作: {action}")
                    success = self.execute_action(action, step)
                    step += 1
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
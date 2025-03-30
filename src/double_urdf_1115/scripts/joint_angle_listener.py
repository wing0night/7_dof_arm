#!/usr/bin/env python
import rospy
from sensor_msgs.msg import JointState

def joint_states_callback(msg):
    # 获取当前时间戳
    timestamp = rospy.get_time()
    
    # 打印关节信息
    rospy.loginfo("\n\nReceived JointStates at [%f]", timestamp)
    
    # 遍历所有关节
    for name, position in zip(msg.name, msg.position):
        # 转换为角度值（如果需要的话）
        # 注意：这里假设位置值已经是弧度单位，如需转换为角度需乘以 180/π
        angle_deg = position * (180.0 / 3.141592653589793)
        rospy.loginfo("Joint %s: %.2f rad (%.2f°)", name, position, angle_deg)
        
        # 这里可以添加自定义处理逻辑，比如：
        # - 存储到文件
        # - 进行安全检测
        # - 发送到其他节点

def listener():
    rospy.init_node('joint_angle_listener', anonymous=True)
    # 订阅/joint_states话题，队列长度设为10
    rospy.Subscriber("/joint_states", JointState, joint_states_callback)
    rospy.loginfo("Joint Angle Listener started. Waiting for joint states...")
    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass






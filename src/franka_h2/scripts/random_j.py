import numpy as np
from fk import cal_fk

def g_rand_pose():
    """
    随机生成一个有效的末端执行器位置
    """
    # 随机生成一个有效的末端执行器位置
    # 定义关节限制
    Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    # 生成随机关节角度（每个关节独立采样）
    random_joint_angles = np.random.uniform(low=Q_MIN, high=Q_MAX)

    # 打印结果
    # print("随机生成的关节角度（弧度）：")
    # print(np.round(random_joint_angles, 4)) # 保留四位小数
    # print(random_joint_angles)
    
    T_total = cal_fk(random_joint_angles)
    # print(T_total)
    return T_total
    


if __name__ == "__main__":
    g_rand_pose()



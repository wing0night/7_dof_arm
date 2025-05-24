import numpy as np

def dh_matrix(alpha, a, d, theta):
    """生成改进DH齐次变换矩阵"""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, a],
        [np.sin(theta)*np.cos(alpha),  np.cos(theta)*np.cos(alpha), -np.sin(alpha), -d*np.sin(alpha)],
        [np.sin(theta)*np.sin(alpha),             np.cos(theta)* np.sin(alpha),      np.cos(alpha), d*np.cos(alpha)],
        [0,              0,                            0,                            1]
    ])
    

# DH参数表（单位：米，弧度）
dh_params = [
    [0,         0,      0.333,   0],  # 关节1
    [-np.pi/2,  0,      0,       0],  # 关节2
    [np.pi/2,   0,      0.316,   0],  # 关节3
    [np.pi/2,   0.0825, 0,       0],  # 关节4
    [-np.pi/2, -0.0825, 0.384,   0],  # 关节5
    [np.pi/2,   0,      0,       0],  # 关节6
    [np.pi/2,   0.088,  0,   0],   # 关节7
    [0, 0, 0.107, 0], # flange
    [0, 0, 0.1034, np.pi/4], # end effector
]
T_trans = [
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

theta_offset = [2.22601256 , 1.2024973 , -2.72513796 ,-2.21730977, -2.19911507 , 0.1795953,  0]  # 移除第8个偏移

def cal_fk(theta_offset):
    T_total = np.eye(4)  # 初始化总变换矩阵为单位矩阵
    

    for i in range(7):
        alpha, a, d, theta = dh_params[i]
        T = dh_matrix(alpha, a, d, theta + theta_offset[i])
        T_total = T_total @ T
    alpha, a, d, theta = dh_params[7]
    T1 = dh_matrix(alpha, a, d, theta)
    T_total = T_total @ T1
    alpha, a, d, theta = dh_params[8]
    T2 = dh_matrix(alpha, a, d, theta)
    T_total = T_total @ T2
    T_total = T_total @ T_trans

    # print("Actual position:")
    # print(T_total)
    return T_total


# main函数
if __name__ == "__main__":
    print(cal_fk(theta_offset))
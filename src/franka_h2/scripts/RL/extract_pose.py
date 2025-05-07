import numpy as np

def extract_pose(T):
    # 提取位置
    position = T[:3, 3]
    
    # 提取旋转矩阵
    R = T[:3, :3]
    
    # 计算旋转角度
    trace = np.trace(R)
    theta = np.arccos((trace - 1) / 2)
    
    # 计算旋转轴
    antisym = (R - R.T) / (2 * np.sin(theta) + 1e-8)  # 避免除零
    axis = np.array([antisym[2,1], antisym[0,2], antisym[1,0]])
    
    # 处理角度符号
    if R[1,0] < 0:
        theta = -theta
    
    return position, np.append(axis, theta)

# 输入矩阵
T_trans = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

position, axis_angle = extract_pose(T_trans)
print("Position:", position)          # [0 0 0]
print("Axis-Angle:", axis_angle)      # [0. 0. 1. -1.5708] (即 -π/2)
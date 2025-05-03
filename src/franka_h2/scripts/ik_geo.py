import numpy as np
from typing import List, Tuple, Optional
from fk import cal_fk
import time

# Constants
D1 = 0.3330
D3 = 0.3160
D5 = 0.3840
D7E = 0.2104
A4 = 0.0825
A7 = 0.0880

# Precomputed constants
LL24 = 0.10666225  # a4^2 + d3^2
LL46 = 0.15426225  # a4^2 + d5^2
L24 = 0.326591870689  # sqrt(LL24)
L46 = 0.392762332715  # sqrt(LL46)

THETA_H46 = 1.35916951803  # atan(d5/a4)
THETA_342 = 1.31542071191  # atan(d3/a4)
THETA_46H = 0.211626808766  # acot(d5/a4)

# Joint limits
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

def franka_IK_EE(O_T_EE: np.ndarray, q7: float, q_actual: np.ndarray) -> List[np.ndarray]:
    """
    Analytical inverse kinematics for Franka Emika robot
    
    Args:
        O_T_EE: 4x4 transformation matrix of end effector
        q7: Joint angle for joint 7 (redundant joint)
        q_actual: Current joint angles
        
    Returns:
        List of possible joint configurations (up to 4 solutions)
    """
    # Check q7 limits
    if q7 <= Q_MIN[6] or q7 >= Q_MAX[6]:
        return [np.full(7, np.nan) for _ in range(4)]
    
    # Extract rotation and position from transform
    R_EE = O_T_EE[:3, :3]
    z_EE = O_T_EE[:3, 2]
    p_EE = O_T_EE[:3, 3]
    
    # Compute p7 and p6
    p_7 = p_EE - D7E * z_EE
    x_EE_6 = np.array([np.cos(q7 - np.pi/4), -np.sin(q7 - np.pi/4), 0.0])
    x_6 = R_EE @ x_EE_6
    x_6 = x_6 / np.linalg.norm(x_6)
    p_6 = p_7 - A7 * x_6
    
    # Compute q4
    p_2 = np.array([0.0, 0.0, D1])
    V26 = p_6 - p_2
    LL26 = np.sum(V26**2)
    L26 = np.sqrt(LL26)
    
    # Check triangle inequality
    if L24 + L46 < L26 or L24 + L26 < L46 or L26 + L46 < L24:
        return [np.full(7, np.nan) for _ in range(4)]
    
    theta246 = np.arccos((LL24 + LL46 - LL26)/(2.0 * L24 * L46))
    q4 = theta246 + THETA_H46 + THETA_342 - 2.0 * np.pi
    
    if q4 <= Q_MIN[3] or q4 >= Q_MAX[3]:
        return [np.full(7, np.nan) for _ in range(4)]
    
    # Compute q6
    theta462 = np.arccos((LL26 + LL46 - LL24)/(2.0 * L26 * L46))
    theta26H = THETA_46H + theta462
    D26 = -L26 * np.cos(theta26H)
    
    Z_6 = np.cross(z_EE, x_6)
    Y_6 = np.cross(Z_6, x_6)
    R_6 = np.column_stack((x_6, Y_6/np.linalg.norm(Y_6), Z_6/np.linalg.norm(Z_6)))
    V_6_62 = R_6.T @ (-V26)
    
    Phi6 = np.arctan2(V_6_62[1], V_6_62[0])
    Theta6 = np.arcsin(D26/np.sqrt(V_6_62[0]**2 + V_6_62[1]**2))
    
    q6_options = [np.pi - Theta6 - Phi6, Theta6 - Phi6]
    solutions = []
    
    for q6 in q6_options:
        # Normalize q6 to joint limits
        if q6 <= Q_MIN[5]:
            q6 += 2.0 * np.pi
        elif q6 >= Q_MAX[5]:
            q6 -= 2.0 * np.pi
            
        if q6 <= Q_MIN[5] or q6 >= Q_MAX[5]:
            continue
            
        # Compute q1, q2
        thetaP26 = 3.0 * np.pi/2 - theta462 - theta246 - THETA_342
        thetaP = np.pi - thetaP26 - theta26H
        LP6 = L26 * np.sin(thetaP26)/np.sin(thetaP)
        
        z_6_5 = np.array([np.sin(q6), np.cos(q6), 0.0])
        z_5 = R_6 @ z_6_5
        V2P = p_6 - LP6 * z_5 - p_2
        L2P = np.linalg.norm(V2P)
        
        if abs(V2P[2]/L2P) > 0.999:
            q1 = q_actual[0]
            q2 = 0.0
            q1_options = [q1]
            q2_options = [q2]
        else:
            q1 = np.arctan2(V2P[1], V2P[0])
            q2 = np.arccos(V2P[2]/L2P)
            q1_options = [q1, q1 + np.pi if q1 < 0 else q1 - np.pi]
            q2_options = [q2, -q2]
        
        for q1 in q1_options:
            for q2 in q2_options:
                if (q1 <= Q_MIN[0] or q1 >= Q_MAX[0] or 
                    q2 <= Q_MIN[1] or q2 >= Q_MAX[1]):
                    continue
                
                # Compute q3
                z_3 = V2P/L2P
                Y_3 = -np.cross(V26, V2P)
                y_3 = Y_3/np.linalg.norm(Y_3)
                x_3 = np.cross(y_3, z_3)
                
                R_1 = np.array([[np.cos(q1), -np.sin(q1), 0],
                               [np.sin(q1), np.cos(q1), 0],
                               [0, 0, 1]])
                R_1_2 = np.array([[np.cos(q2), -np.sin(q2), 0],
                                 [0, 0, 1],
                                 [-np.sin(q2), -np.cos(q2), 0]])
                R_2 = R_1 @ R_1_2
                x_2_3 = R_2.T @ x_3
                q3 = np.arctan2(x_2_3[2], x_2_3[0])
                
                if q3 <= Q_MIN[2] or q3 >= Q_MAX[2]:
                    continue
                
                # Compute q5
                VH4 = p_2 + D3*z_3 + A4*x_3 - p_6 + D5*z_5
                R_5_6 = np.array([[np.cos(q6), -np.sin(q6), 0],
                                 [0, 0, -1],
                                 [np.sin(q6), np.cos(q6), 0]])
                R_5 = R_6 @ R_5_6.T
                V_5_H4 = R_5.T @ VH4
                
                q5 = -np.arctan2(V_5_H4[1], V_5_H4[0])
                
                if q5 <= Q_MIN[4] or q5 >= Q_MAX[4]:
                    continue
                
                solution = np.array([q1, q2, q3, q4, q5, q6, q7])
                solutions.append(solution)
    
    # Pad with NaN solutions if needed
    while len(solutions) < 4:
        solutions.append(np.full(7, np.nan))
    
    return solutions[:4]  # Return at most 4 solutions

def test_ik():
    # Test transformation matrix
    # T_test = np.array([
    #     [1, 0, 0, 0.3],
    #     [0, 1, 0, 0.2],
    #     [0, -0, 1, 0.5],
    #     [0, 0, 0, 1]
    # ])
#     T_test = np.array([
#         [-0.50410872, -0.13489031, -0.85304103 ,-0.0760815 ],
#  [-0.81089773,  0.41381235 , 0.41376831,  0.16729992],
#  [ 0.29718558  ,0.90031325, -0.31798866 , 0.76910605],
#  [ 0.   ,       0.     ,     0.    ,      1.        ]
#     ])
    T_test = np.array([
        [-0.85458596, -0.4960149, -0.15379227 ,-0.20078374 ],
 [-0.43994801,  0.84885586 , -0.29306907,  0.1906267],
 [ 0.27591409  ,-0.18279211, -0.94364106 , 0.51011827],
 [ 0.   ,       0.     ,     0.    ,      1.        ]
    ])
    
    # Test current joint angles
    q_current = np.zeros(7)
    
    # Test q7 value
    q7_test = 0.0
    
    # Compute IK solutions
    solutions = franka_IK_EE(T_test, q7_test, q_current)
    
    
    # Print results
    # for i, solution in enumerate(solutions):
    #     # if not np.any(np.isnan(solution)):
    #     print(f"Solution {i+1}:", solution)
    print("Target position:")
    print(T_test)
    print(f"ik Solution :", solutions[0])
    T_total = cal_fk(solutions[0])
    print("Actual position:")
    print(T_total)

if __name__ == "__main__":
    start_time = time.perf_counter()
    test_ik() 

    # 记录结束时间
    end_time = time.perf_counter()
    
    # 计算并存储耗时（转换为毫秒）
    elapsed_ms = (end_time - start_time) * 1000

    print(f"耗时: {elapsed_ms:.4f} ms")
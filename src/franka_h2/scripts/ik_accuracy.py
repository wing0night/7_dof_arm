from fk import cal_fk
import time
import roboticstoolbox as rtb
import numpy as np
from ik_geo import franka_IK_EE
from random_j import g_rand_pose
import matplotlib.pyplot as plt


panda = rtb.models.URDF.Panda()

T_goal = cal_fk([2.22601256 , 1.2024973 , -2.72513796 ,-2.21730977, -2.19911507 , 0.1795953,  0])


def test_ik_LM(T_test):
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

    point_sol = panda.ikine_LM(T_test)
    
    # Compute IK solutions
    solution = point_sol.q
    
    
    # Print results
    # for i, solution in enumerate(solutions):
    #     # if not np.any(np.isnan(solution)):
    #     print(f"Solution {i+1}:", solution)
    print("Target position:")
    print(T_test)
    print(f"ik Solution :", solution)
    T_total = cal_fk(solution)
    print("Actual position:")
    print(T_total)

    # 取平移部分
    pos_target = T_test[:3, 3]
    pos_actual = T_total[:3, 3]
    error = np.linalg.norm(pos_target - pos_actual)
    return error

def test_ik_geo(T_test):
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

    point_sol = franka_IK_EE(T_test, q7_test, q_current)
    
    # Compute IK solutions
    solution = point_sol[0]
    
    
    # Print results
    # for i, solution in enumerate(solutions):
    #     # if not np.any(np.isnan(solution)):
    #     print(f"Solution {i+1}:", solution)
    print("Target position:")
    print(T_test)
    print(f"ik Solution :", solution)
    T_total = cal_fk(solution)
    print("Actual position:")
    print(T_total)

    pos_target = T_test[:3, 3]
    pos_actual = T_total[:3, 3]
    error = np.linalg.norm(pos_target - pos_actual)
    return error

if __name__ == "__main__":
    # 初始化数据存储
    errors_LM = []
    errors_geo = []
    generation_counts = []

    for i in range(100):
        T_test = g_rand_pose()

        # LM法误差
        error_LM = test_ik_LM(T_test)
        errors_LM.append(error_LM)

        # Geo法误差
        error_geo = test_ik_geo(T_test)
        errors_geo.append(error_geo)

        generation_counts.append(i+1)
        print(f"第{i+1}次: LM误差: {error_LM:.6f} m, Geo误差: {error_geo:.6f} m")

    plt.rcParams["figure.facecolor"] = 'white'
    plt.rcParams["axes.facecolor"] = 'white'
    plt.rcParams["savefig.facecolor"] = 'white'
    plt.figure(figsize=(12, 6))

    plt.plot(generation_counts, errors_LM, marker='o', linestyle='--', linewidth=1,
             markersize=4, color='steelblue', label='LM IK Error')
    plt.plot(generation_counts, errors_geo, marker='s', linestyle='-', linewidth=1,
             markersize=4, color='orange', label='Geo IK Error')

    plt.title('IK Position Error Profile (LM vs Geo)', fontsize=14)
    plt.xlabel('Experiment Count', fontsize=12)
    plt.ylabel('Position Error (m)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)

    avg_error_LM = np.mean(errors_LM)
    avg_error_geo = np.mean(errors_geo)
    plt.axhline(y=avg_error_LM, color='steelblue', linestyle='--',
                label=f'LM Avg Error: {avg_error_LM:.6f} m')
    plt.axhline(y=avg_error_geo, color='orange', linestyle='--',
                label=f'Geo Avg Error: {avg_error_geo:.6f} m')
    plt.legend()
    plt.tight_layout()
    plt.show()





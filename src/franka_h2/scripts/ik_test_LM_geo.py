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

if __name__ == "__main__":
    # 初始化数据存储
    execution_times_LM = []
    execution_times_geo = []
    generation_counts = []

    # 执行100次时间测量循环
    for i in range(100):
        # 随机生成目标位姿
        T_test = g_rand_pose()

        # 记录LM法开始时间
        start_time_LM = time.perf_counter()
        test_ik_LM(T_test)
        end_time_LM = time.perf_counter()
        elapsed_ms_LM = (end_time_LM - start_time_LM) * 1000
        execution_times_LM.append(elapsed_ms_LM)

        # 记录Geo法开始时间
        start_time_geo = time.perf_counter()
        test_ik_geo(T_test)
        end_time_geo = time.perf_counter()
        elapsed_ms_geo = (end_time_geo - start_time_geo) * 1000
        execution_times_geo.append(elapsed_ms_geo)

        generation_counts.append(i+1)
        print(f"第{i+1}次: LM耗时: {elapsed_ms_LM:.4f} ms, Geo耗时: {elapsed_ms_geo:.4f} ms")

    plt.rcParams["figure.facecolor"] = 'white'   # 画布背景色
    plt.rcParams["axes.facecolor"] = 'white'     # 坐标区背景色
    plt.rcParams["savefig.facecolor"] = 'white'  # 保存图片背景色
    # 创建可视化图表
    plt.figure(figsize=(12, 6))

    # 绘制时间序列折线图
    plt.plot(generation_counts, 
            execution_times_LM, 
            marker='o', 
            linestyle='--', 
            linewidth=1,
            markersize=4,
            color='steelblue',
            label='LM IK')

    plt.plot(generation_counts, 
            execution_times_geo, 
            marker='s', 
            linestyle='-', 
            linewidth=1,
            markersize=4,
            color='orange',
            label='Geo IK')

    # 设置图表样式
    plt.title('IK Execution Time Profile (LM vs Geo)', fontsize=14)
    plt.xlabel('Generation Count', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)

    # 添加统计信息
    avg_time_LM = np.mean(execution_times_LM)
    avg_time_geo = np.mean(execution_times_geo)
    plt.axhline(y=avg_time_LM, color='steelblue', linestyle='--', 
            label=f'LM Avg: {avg_time_LM:.4f} ms')
    plt.axhline(y=avg_time_geo, color='orange', linestyle='--', 
            label=f'Geo Avg: {avg_time_geo:.4f} ms')
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()






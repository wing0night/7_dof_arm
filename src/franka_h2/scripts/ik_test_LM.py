from fk import cal_fk
import time
import roboticstoolbox as rtb
import numpy as np
from ik_geo import franka_IK_EE
from random_j import g_rand_pose
import matplotlib.pyplot as plt


panda = rtb.models.URDF.Panda()

T_goal = cal_fk([2.22601256 , 1.2024973 , -2.72513796 ,-2.21730977, -2.19911507 , 0.1795953,  0])
# print(T_goal)

# 逆运动学解算  
# point_sol = panda.ikine_LM(T_goal)
# print("IK Solution: ", point_sol.q)
# T_actual = panda.fkine(point_sol.q)
# print(T_actual)

# print("Target position:")
# print(T_goal)
# start_time = time.perf_counter()
# point_sol = panda.ikine_LM(T_goal)
# print("ik Solution :", point_sol.q)
# T_actual = cal_fk(point_sol.q)
# print("Actual position:")
# print(T_actual)

# # 记录结束时间
# end_time = time.perf_counter()
# # 计算并存储耗时（转换为毫秒）
# elapsed_ms = (end_time - start_time) * 1000
# print(f"耗时: {elapsed_ms:.4f} ms")



def test_ik(T_test):
    # Test transformation matrix
    # T_test = np.array([
    #     [1, 0, 0, 0.3],
    #     [0, 1, 0, 0.2],
    #     [0, -0, 1, 0.5],
    #     [0, 0, 0, 1]
    # ])

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

if __name__ == "__main__":
    # 初始化数据存储
    execution_times = []
    generation_counts = []

    # 执行100次时间测量循环
    for i in range(100):
        # 记录开始时间（使用高精度计时器）
        start_time = time.perf_counter()
        
        T_test = g_rand_pose()
        # 调用逆运动学函数
        test_ik(T_test)
        
        # 记录结束时间
        end_time = time.perf_counter()
        
        # 计算并存储耗时（转换为毫秒）
        elapsed_ms = (end_time - start_time) * 1000
        print(f"耗时: {elapsed_ms:.4f} ms")
        execution_times.append(elapsed_ms)
        generation_counts.append(i+1)

    # 创建可视化图表
    plt.figure(figsize=(12, 6))

    # 绘制时间序列折线图
    plt.plot(generation_counts, 
            execution_times, 
            marker='o', 
            linestyle='--', 
            linewidth=1,
            markersize=4,
            color='steelblue')

    # 设置图表样式
    plt.title('Joint Angle Generation Time Profile', fontsize=14)
    plt.xlabel('Generation Count', fontsize=12)
    plt.ylabel('Execution Time (ms)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, 101, 10))
    plt.xlim(0, 100)

    # 添加统计信息
    avg_time = np.mean(execution_times)
    plt.axhline(y=avg_time, color='firebrick', linestyle='--', 
            label=f'Average: {avg_time:.4f} ms')
    plt.legend()

    # 显示图表
    plt.tight_layout()
    plt.show()






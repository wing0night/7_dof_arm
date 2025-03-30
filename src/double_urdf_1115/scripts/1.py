import matplotlib
matplotlib.use('Agg')  # 设置非 GUI 后端
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
time = np.linspace(0, 10, 100)
position = np.sin(time)

# 绘图并保存
plt.figure(figsize=(10, 6))
plt.plot(time, position, label='Joint Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (rad)')
plt.suptitle('Joint States Analysis')
plt.legend()

# 保存后直接关闭，不显示窗口
plt.savefig('joint_states_analysis.png')
plt.close('all')  # 确保所有图形资源释放
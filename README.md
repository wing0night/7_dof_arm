## intro



## structure

自定义msg：TrajectoryData(msg/TrajectoryData.msg)传递给plot_cal.py用于绘制速度、加速度、力矩随时间变化的图像

LM_ik_solver.py中利用LM方法进行逆运动学解算，并将解算结果进行梯形速度曲线or五次多项插值得到轨迹，进而发布消息给到Gazebo机械臂运动控制&绘图函数作图

RRT.py：RRT方法进行轨迹规划，并测试过程中的速度&加速度&力矩数据

plot_cal.py：集成作图、一般轨迹规划方法损失函数计算

franka_h2/rl：强化学习训练、部署代码


## recording




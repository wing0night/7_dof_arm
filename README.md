## intro



## structure

自定义msg：TrajectoryData(msg/TrajectoryData.msg)传递给plot_cal.py用于绘制速度、加速度、力矩随时间变化的图像

LM_ik_solver.py中利用LM方法进行逆运动学解算，并将解算结果进行梯形速度曲线or五次多项插值得到轨迹，进而发布消息给到Gazebo机械臂运动控制&绘图函数作图

RRT.py：RRT方法进行轨迹规划，并测试过程中的速度&加速度&力矩数据

plot_cal.py：集成作图、一般轨迹规划方法损失函数计算

franka_h2/rl：强化学习训练、部署代码


## recording




- RRT中还存在bug，中move_to_goal函数中，在发送运动消息后如何等到机械臂运动到目标位置后再进行当前位置更新（`_update_current_position`函数），而不是直接在函数最后加一个`self.current_joint_positions = goal_positions`（不太规范）


## scripts

### 逆运动学解算

ik_geo.py：逆解论文的解法的python版（ik_geo_cpp是官方写的cpp版）

fk.py：panda的正向运动学计算

random_j.py：随机生成末端位姿

test_ik_plot.py：测试几何解法需要时间并绘图

ik_test_LM.py：测试LM法逆解需要时间并绘图

Cubic_spline_interpolation_simple.py：测试三次样条差值(只有起始点)。输出ROS运行视频，运动过程中速度-时间曲线、加速度-时间曲线、力矩-时间曲线。

Cubic_spline_interpolation_inter.py：测试三次样条差值（有中间点）

CSI.py：含中间点的三次样条差值规划中用到的插值求解器类

NSGA-2.py：测试NSGA-2群优化算法

plot_cal.py：根据ROS速度、力矩消息进行position, velocity, acceleration, torque, loss function（计算和NSGA中一致）-t图像绘制








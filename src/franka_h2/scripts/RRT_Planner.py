import numpy as np
import rospy

class RRTPlanner:
    def __init__(self, joint_limits, step_size=0.1):
        """
        :param joint_limits: 关节角度限制 [[min1,max1],...]
        :param step_size: 单步扩展步长（弧度）
        """
        self.joint_limits = np.array(joint_limits)
        self.step_size = step_size
        self.max_iter = 1000

    def plan(self, start_config, goal_config):
        tree = [{'config': np.array(start_config), 'parent': None}]
        
        for _ in range(self.max_iter):
            rand_config = self._random_sample(goal_bias=0.9, goal=goal_config)
            nearest_node = self._nearest_neighbor(tree, rand_config)
            new_config = self._step_towards(nearest_node['config'], rand_config)
            
            if not self._check_collision(new_config):
                # 创建新节点并验证数据
                new_node = {
                    'config': new_config,
                    'parent': nearest_node
                }
                assert isinstance(new_config, np.ndarray), "配置必须为 numpy 数组"
                assert new_config.shape == (len(self.joint_limits),), "配置维度错误"
                tree.append(new_node)
                
                if self._is_goal_reached(new_config, goal_config):
                    return self._generate_path(tree, tree[-1])  # 传递节点对象
        
        rospy.logwarn("RRT未找到可行路径")
        return None

    def _generate_path(self, tree, end_node):
        """容差匹配版路径生成"""
        path = []
        node = end_node
        
        # 查找匹配节点（带容差）
        found = False
        for n in tree:
            if (np.allclose(n['config'], node['config'], atol=0.001) and 
                n['parent'] == node['parent']):
                found = True
                break
        if not found:
            rospy.logerr("错误：终点节点未在树中找到")
            return None

        # 构建路径
        while node is not None:
            path.append(node['config'])
            node = node['parent']
        
        return np.array(path[::-1]) if path else None

    def _random_sample(self, goal_bias, goal):
        """随机采样（含目标偏向）""" 
        if np.random.rand() < goal_bias:
            return goal
        else:
            return np.array([np.random.uniform(low, high) 
                           for (low, high) in self.joint_limits])

    def _nearest_neighbor(self, tree, target):
        """寻找最近的树节点"""
        distances = [np.linalg.norm(node['config'] - target) for node in tree]
        return tree[np.argmin(distances)]

    def _step_towards(self, from_config, to_config):
        """单步扩展"""
        direction = to_config - from_config
        distance = np.linalg.norm(direction)
        if distance < self.step_size:
            return to_config
        else:
            step = direction * (self.step_size / distance)
            return from_config + step

    def _check_collision(self, config):
        """碰撞检测（需根据实际场景实现）"""
        # 可集成MoveIt的碰撞检查接口
        return False  # 暂时假设无障碍

    def _is_goal_reached(self, config, goal, tolerance=0.1):
        """到达目标判断"""
        return np.linalg.norm(config - goal) < tolerance





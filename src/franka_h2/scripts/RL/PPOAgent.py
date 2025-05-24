from PPONetwork import PPONetwork
import torch
import torch.nn as nn

import random

class PPOAgent:
    """PPO算法实现类"""
    def __init__(self, state_dim, action_dim):

        # 调整超参数
        self.gamma = 0.99  # 保持不变
        self.epsilon = 0.1  # 降低截断参数
        self.batch_size = 256  # 增加批次大小
        self.epochs = 10  # 减少训练轮数
        self.lr = 3e-4  # 调整学习率
        
        # 探索参数
        self.exploration_rate = 0.1  # 增加初始探索率
        self.min_exploration_rate = 0.01  # 提高最小探索率
        self.exploration_decay = 0.995  # 降低衰减速度
        self.action_noise_std = 0.3  # 增加初始噪声
        self.noise_decay = 0.995  # 降低噪声衰减速度
        
        
        # 使用CPU进行训练
        self.device = torch.device("cpu")
        # 网络和优化器
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 衰减episode参数
        self.episode_length = 100  # 初始episode长度
        self.decay_rate = 0.95    # 衰减率
        self.min_ep_length = 10   # 最小长度
    
    def get_action(self, state, training=True):
        """获取动作，包含探索机制"""
        with torch.no_grad():
            action, _ = self.policy(state)
            
            if training:
                # 添加探索噪声
                if random.random() < self.exploration_rate:
                    noise = torch.randn_like(action) * self.action_noise_std
                    action = action + noise
                    # 确保动作在[-1,1]范围内
                    action = torch.clamp(action, -1, 1)
                
                # 更新探索率
                self.exploration_rate = max(
                    self.min_exploration_rate,
                    self.exploration_rate * self.exploration_decay
                )
                # 更新噪声标准差
                self.action_noise_std *= self.noise_decay
            
            return action

    def update_policy(self, states, actions, rewards, next_states, dones):
        """策略更新函数"""
        # 转换数据为Tensor
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 计算优势函数和回报
        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)
            advantages = rewards + (1 - dones) * self.gamma * next_values - values
        
        # 优化循环
        for _ in range(self.epochs):
            # 计算新旧策略概率比
            new_actions, new_values = self.policy(states)
            ratio = torch.exp(new_actions - actions)
            
            # 计算截断目标函数
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值函数损失
            value_loss = nn.MSELoss()(new_values, rewards + self.gamma * next_values)
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        # 添加熵正则化项，鼓励探索
        entropy_coef = 0.01
        entropy = -torch.mean(torch.sum(new_actions * torch.log(new_actions + 1e-10), dim=1))
        
        # 修改总损失
        loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy



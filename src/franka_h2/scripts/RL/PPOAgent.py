from PPONetwork import PPONetwork
import torch
import torch.nn as nn

class PPOAgent:
    """PPO算法实现类"""
    def __init__(self, state_dim, action_dim):
        # 超参数设置
        self.gamma = 0.99       # 折扣因子
        self.epsilon = 0.2      # PPO截断参数
        self.batch_size = 64    # 批大小。使用较小的批次大小，提高CPU训练速度
        self.epochs = 10        # 优化轮次
        self.lr = 3e-4         # 学习率
        
        # 使用CPU进行训练
        self.device = torch.device("cpu")
        # 网络和优化器
        self.policy = PPONetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 衰减episode参数
        self.episode_length = 10  # 初始episode长度
        self.decay_rate = 0.995    # 衰减率
        self.min_ep_length = 10   # 最小长度

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



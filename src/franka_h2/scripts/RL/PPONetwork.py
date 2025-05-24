import torch.nn as nn
import numpy as np


class PPONetwork(nn.Module):
    """PPO策略网络和价值网络定义"""
    def __init__(self, state_dim, action_dim):
        super(PPONetwork, self).__init__()
        
        # 增加网络容量和稳定性
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        
        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """使用正交初始化"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    def forward(self, state):
        # 添加维度检查和处理
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # 添加batch维度
            
        features = self.shared_layer(state)
        return self.actor(features), self.critic(features)




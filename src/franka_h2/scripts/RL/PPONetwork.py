import torch.nn as nn


import torch.nn as nn

class PPONetwork(nn.Module):
    """PPO策略网络和价值网络定义"""
    def __init__(self, state_dim, action_dim):
        super(PPONetwork, self).__init__()
        
        # 修正共享特征提取层的维度
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),  # 24 -> 128
            nn.ReLU(),
            nn.Linear(128, 64),         # 128 -> 64
            nn.ReLU()
        )
        
        # 修正Actor网络（策略网络）维度
        self.actor = nn.Sequential(
            nn.Linear(64, 32),          # 64 -> 32
            nn.ReLU(),
            nn.Linear(32, action_dim),  # 32 -> 7
            nn.Tanh()                   # 输出[-1,1]范围动作
        )
        
        # 修正Critic网络（价值网络）维度
        self.critic = nn.Sequential(
            nn.Linear(64, 32),          # 64 -> 32
            nn.ReLU(),
            nn.Linear(32, 1)            # 32 -> 1
        )

    def forward(self, state):
        # 添加维度检查和处理
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # 添加batch维度
            
        features = self.shared_layer(state)
        return self.actor(features), self.critic(features)




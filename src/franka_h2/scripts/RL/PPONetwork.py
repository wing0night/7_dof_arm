import torch.nn as nn


class PPONetwork(nn.Module):
    """PPO策略网络和价值网络定义"""
    def __init__(self, state_dim, action_dim):
        super(PPONetwork, self).__init__()
        # 共享特征提取层
        self.shared_layer = nn.Sequential( # 使用CPU训练所以使用较小的dim
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Actor网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出[-1,1]范围动作
        )
        
        # Critic网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        features = self.shared_layer(state)
        return self.actor(features), self.critic(features)




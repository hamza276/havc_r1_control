import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomPolicyNetwork(nn.Module):
    """
    Custom policy network optimized for HVAC control
    """
    def __init__(self, input_dim, hidden_dim=64):
        super(CustomPolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.network(x)

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for HVAC control
    """
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        
        # Custom network initialization
        self.shared_net = nn.Sequential(
            nn.Linear(self.features_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        
        self.action_net = nn.Sequential(
            nn.Linear(64, self.action_space.shape[0]),
            nn.Tanh()
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(64, 1)
        )
    
    def forward(self, obs, deterministic=False):
        """
        Forward pass in the neural network
        """
        shared_features = self.shared_net(obs)
        
        # Actor (policy) head
        mean_actions = self.action_net(shared_features)
        
        # Critic (value) head
        values = self.value_net(shared_features)
        
        return mean_actions, values
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy
        """
        shared_features = self.shared_net(obs)
        
        values = self.value_net(shared_features)
        mean_actions = self.action_net(shared_features)
        
        log_prob = -0.5 * ((actions - mean_actions) ** 2)
        
        return values, log_prob, mean_actions
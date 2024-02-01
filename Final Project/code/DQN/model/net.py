import torch
import numpy as np
import torch.nn as nn
from random import uniform, randint

class DeepQNetwork(nn.Module):
    def __init__(self, obs_shape, action_count):
        super(DeepQNetwork, self).__init__()
        self.obs_shape = obs_shape
        self.action_count = action_count

        self.features = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self._feature_dimensions, 512),
            nn.ReLU(),
            nn.Linear(512, action_count)
        )

    def forward(self, input_tensor):
        conv_out = self.features(input_tensor).reshape(input_tensor.size()[0], -1).contiguous()
        return self.fc(conv_out)
    
    def select_action(self, observation, exploration_rate, computation_device):
        if uniform(0, 1) > exploration_rate:
            observation_tensor = torch.tensor(observation, dtype=torch.float32) \
                .unsqueeze(0).to(computation_device)
            predicted_q_values = self(observation_tensor)
            selected_action = predicted_q_values.max(1)[1].view(1, 1).item()
        else:
            selected_action = randint(0, self.action_count - 1)
        return selected_action
    
    @property
    def _feature_dimensions(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.obs_shape)
            conv_out = self.features(dummy_input)
            return conv_out.view(1, -1).size(1)
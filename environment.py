import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HelpdeskEnvironment(gym.Env):
    def __init__(self, training_data, embedding_model):
        super(HelpdeskEnvironment, self).__init__()
        self.training_data = training_data
        self.embedding_model = embedding_model
        self.current_step = 0
        self.action_space = spaces.Discrete(3) # 0=HR, 1=IT, 2=Payroll
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(384,), dtype=np.float32)

    def _get_reward(self, action, correct_action):
        return 1.0 if action == correct_action else -1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs, _ = self._next_observation()
        return obs, {}

    def _next_observation(self):
        query, correct_action_idx = self.training_data[self.current_step]
        embedding = self.embedding_model.encode(query, convert_to_tensor=False)
        return embedding, correct_action_idx

    def step(self, action):
        _, correct_action_idx = self._next_observation()
        reward = self._get_reward(action, correct_action_idx)
        self.current_step += 1
        terminated = self.current_step >= len(self.training_data)
        next_obs, _ = self._next_observation() if not terminated else (np.zeros(384, dtype=np.float32), -1)
        return next_obs, reward, terminated, False, {}
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class OptimizedHVACEnv(gym.Env):
    """
    Optimized HVAC Environment designed for convergence
    """
    def __init__(self, df, features, valve_target='Valve', delta_target='valve_delta', 
                 max_steps=50, difficulty_level=None, max_delta=20.0):
        super(OptimizedHVACEnv, self).__init__()
        
        # Store data
        self.df = df.copy().reset_index(drop=True)
        self.features = features
        self.valve_target = valve_target
        self.delta_target = delta_target
        self.max_delta = max_delta
        
        # Episode parameters
        self.max_steps = min(max_steps, len(df) - 5)
        self.current_step = 0
        self.current_idx = 0
        
        # Curriculum learning parameter
        self.difficulty_level = difficulty_level
        
        # Action space: Continuous normalized delta (-1 to 1)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # Observation space: Normalized features
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(len(features),),
            dtype=np.float32
        )
        
        # For tracking performance
        self.episode_errors = []
        self.current_valve = 0.0
        
        # For early stopping logic
        self.consecutive_large_errors = 0
        self.error_threshold = 30.0
        self.max_consecutive_large_errors = 5
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.episode_errors = []
        self.consecutive_large_errors = 0
        
        valid_indices = np.arange(len(self.df))
        
        if self.difficulty_level is not None and 'difficulty' in self.df.columns:
            difficulty_mask = self.df['difficulty'] <= self.difficulty_level
            valid_indices = valid_indices[difficulty_mask.values]
        
        valid_indices = [idx for idx in valid_indices if idx < len(self.df) - self.max_steps - 2]
        
        if len(valid_indices) > 0:
            self.current_idx = np.random.choice(valid_indices)
        else:
            self.current_idx = np.random.randint(0, max(0, len(self.df) - self.max_steps - 2))
        
        self.current_valve = float(self.df.iloc[self.current_idx][self.valve_target])
        observation = self._get_obs()
        
        return observation, {}
    
    def step(self, action):
        observation = self._get_obs()
        
        actual_valve = float(self.df.iloc[self.current_idx][self.valve_target])
        actual_delta = float(self.df.iloc[self.current_idx][self.delta_target])
        
        predicted_delta = float(action[0]) * self.max_delta
        predicted_valve = np.clip(self.current_valve + predicted_delta, 0, 100)
        
        self.current_valve = predicted_valve
        absolute_error = abs(predicted_valve - actual_valve)
        self.episode_errors.append(absolute_error)
        
        # Exponential reward function
        reward = 2.0 * np.exp(-0.05 * absolute_error) - 1.0
        
        if absolute_error > self.error_threshold:
            self.consecutive_large_errors += 1
        else:
            self.consecutive_large_errors = 0
        
        self.current_step += 1
        self.current_idx += 1
        
        done = (
            self.current_step >= self.max_steps or 
            self.current_idx >= len(self.df) - 1 or
            self.consecutive_large_errors >= self.max_consecutive_large_errors
        )
        
        info = {
            'absolute_error': absolute_error,
            'predicted_delta': predicted_delta,
            'actual_delta': actual_delta,
            'predicted_valve': predicted_valve,
            'actual_valve': actual_valve,
            'early_stop': self.consecutive_large_errors >= self.max_consecutive_large_errors
        }
        
        if done:
            info['episode'] = {
                'r': float(reward),
                'mean_absolute_error': float(np.mean(self.episode_errors)),
                'length': self.current_step
            }
        
        next_observation = self._get_obs() if not done else observation
        return next_observation, reward, done, False, info
    
    def _get_obs(self):
        try:
            observation = self.df.iloc[self.current_idx][self.features].values.astype(np.float32)
            return observation
        except Exception as e:
            print(f"Error getting observation: {e}")
            return np.zeros(len(self.features), dtype=np.float32)
    
    def set_difficulty(self, level):
        self.difficulty_level = level
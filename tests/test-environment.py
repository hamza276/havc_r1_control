import unittest
import numpy as np
import pandas as pd
from src.environment.hvac_env import OptimizedHVACEnv

class TestHVACEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up test data
        """
        # Create sample data
        cls.sample_data = pd.DataFrame({
            'ts': pd.date_range('2023-01-01', periods=100, freq='H'),
            'Valve': np.random.uniform(0, 100, 100),
            'RaTemp': np.random.uniform(18, 25, 100),
            'SaTemp': np.random.uniform(15, 22, 100),
            'main.temp': np.random.uniform(10, 30, 100),
            'main.humidity': np.random.uniform(30, 70, 100)
        })
        
        # Add calculated columns
        cls.sample_data['Valve_lag1'] = cls.sample_data['Valve'].shift(1)
        cls.sample_data['valve_delta'] = cls.sample_data['Valve'] - cls.sample_data['Valve_lag1']
        cls.sample_data = cls.sample_data.dropna().reset_index(drop=True)
        
        # Define features
        cls.features = ['RaTemp', 'SaTemp', 'main.temp', 'main.humidity', 'Valve_lag1']
    
    def setUp(self):
        """
        Create environment instance for each test
        """
        self.env = OptimizedHVACEnv(
            df=self.sample_data,
            features=self.features,
            valve_target='Valve',
            delta_target='valve_delta',
            max_steps=50
        )
    
    def test_env_initialization(self):
        """
        Test environment initialization
        """
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.action_space.shape, (1,))
        self.assertEqual(self.env.observation_space.shape, (len(self.features),))
    
    def test_env_reset(self):
        """
        Test environment reset
        """
        observation, _ = self.env.reset()
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (len(self.features),))
    
    def test_env_step(self):
        """
        Test environment step
        """
        self.env.reset()
        action = np.array([0.5])  # Example action
        
        observation, reward, done, truncated, info = self.env.step(action)
        
        self.assertIsInstance(observation, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
        
        # Check info dictionary contains required keys
        required_keys = ['absolute_error', 'predicted_valve', 'actual_valve']
        for key in required_keys:
            self.assertIn(key, info)
    
    def test_reward_range(self):
        """
        Test reward function range
        """
        self.env.reset()
        action = np.array([0.0])  # Neutral action
        
        _, reward, _, _, _ = self.env.step(action)
        
        # Reward should be in range [-1, 1] based on exponential reward function
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)
    
    def test_early_stopping(self):
        """
        Test early stopping mechanism
        """
        self.env.reset()
        
        # Force consecutive large errors
        action = np.array([1.0])  # Maximum action
        done = False
        steps = 0
        
        while not done and steps < 10:
            _, _, done, _, info = self.env.step(action)
            steps += 1
        
        # Should stop early due to consecutive large errors
        self.assertTrue(info['early_stop'])
        self.assertLess(steps, 10)

if __name__ == '__main__':
    unittest.main()
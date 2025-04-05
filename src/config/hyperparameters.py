"""
Configuration parameters for the HVAC control system
"""

# Environment parameters
ENV_CONFIG = {
    'max_steps': 50,
    'max_delta': 20.0,
    'error_threshold': 30.0,
    'max_consecutive_large_errors': 5
}

# PPO hyperparameters
PPO_PARAMS = {
    'learning_rate': 0.0003,
    'n_steps': 512,
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'use_sde': False,
    'sde_sample_freq': -1,
}

# Neural network architecture
POLICY_NETWORK = {
    'hidden_dim': 64,
    'n_layers': 2,
    'activation': 'tanh'
}

# Training parameters
TRAINING_PARAMS = {
    'total_timesteps': 150000,
    'eval_freq': 5000,
    'n_eval_episodes': 5,
    'save_freq': 20000
}

# Curriculum learning parameters
CURRICULUM_PARAMS = {
    'difficulty_stages': [0, 1, 2],
    'stage_duration': 25000,
}

# Feature engineering parameters
FEATURE_PARAMS = {
    'use_time_features': True,
    'use_weather_features': True,
    'use_lag_features': True,
    'n_lags': 2
}

# Validation parameters
VALIDATION_PARAMS = {
    'test_size': 0.1,
    'val_size': 0.1,
    'metrics': ['mae', 'rmse', 'r2', 'within_5', 'within_10', 'within_20']
}
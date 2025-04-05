# HVAC Reinforcement Learning Control

This repository contains an implementation of a Reinforcement Learning-based HVAC control system using Proximal Policy Optimization (PPO).

## Overview

The system uses PPO to learn optimal valve control strategies for HVAC systems. It includes:
- Custom OpenAI Gym environment for HVAC control
- Optimized PPO implementation
- Curriculum learning for improved convergence
- Comprehensive validation and visualization tools

## Requirements

- Python 3.8+
- PyTorch
- Stable Baselines3
- Gymnasium
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hvac-rl-control.git
cd hvac-rl-control

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a Model

```python
from src.training.train import train_optimized_ppo
from src.data.data_preprocessing import load_and_preprocess_data

# Load and preprocess data
df, features, valve_target, delta_target, scaler = load_and_preprocess_data()

# Train model
model, train_env, eval_env = train_optimized_ppo(
    train_df=train_df,
    val_df=val_df,
    features=features,
    valve_target=valve_target,
    delta_target=delta_target,
    log_dir="logs",
    model_dir="models"
)
```

### Validating a Model

```python
from src.evaluation.validate import validate_ppo_model

metrics = validate_ppo_model(
    model=model,
    test_df=test_df,
    features=features,
    valve_target=valve_target,
    delta_target=delta_target,
    output_dir="validation"
)
```

## Project Structure

- `src/`: Source code
  - `config/`: Configuration files
  - `data/`: Data preprocessing
  - `environment/`: HVAC environment
  - `models/`: Neural network models
  - `training/`: Training scripts
  - `evaluation/`: Validation tools
  - `utils/`: Utility functions
- `tests/`: Unit tests
- `examples/`: Example scripts

## Results

The model achieves:
- MAE: 13.70
- RMSE: 27.09
- R² Score: 0.5571
- Within 10 units accuracy: 70.35%

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests.
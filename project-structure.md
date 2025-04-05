hvac_rl_control/
├── README.md
├── requirements.txt
├── setup.py
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── hyperparameters.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_preprocessing.py
│   ├── environment/
│   │   ├── __init__.py
│   │   └── hvac_env.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── custom_policy.py
│   │   └── callbacks.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── validate.py
│   └── utils/
│       ├── __init__.py
│       └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── test_environment.py
│   └── test_training.py
└── examples/
    └── train_model.py
import os
import json
from datetime import datetime
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import logging

from ..environment.hvac_env import OptimizedHVACEnv
from ..models.callbacks import SimpleCurriculumCallback
from stable_baselines3.common.callbacks import CheckpointCallback

logger = logging.getLogger(__name__)

def train_optimized_ppo(train_df, val_df, features, valve_target, delta_target,
                       log_dir, model_dir, total_timesteps=150000):
    """
    Train with optimized PPO implementation focused on convergence
    """
    logger.info("\nStarting optimized PPO training...")
    
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define difficulty stages
    difficulty_stages = [0, 1, 2]
    
    def make_env(df, features, valve_target, delta_target, difficulty_level=0):
        def _init():
            env = OptimizedHVACEnv(
                df=df,
                features=features,
                valve_target=valve_target,
                delta_target=delta_target,
                max_steps=50,
                difficulty_level=difficulty_level
            )
            return env
        return _init
    
    # Create environments
    train_env = make_env(
        df=train_df,
        features=features,
        valve_target=valve_target,
        delta_target=delta_target,
        difficulty_level=difficulty_stages[0]
    )()
    
    train_env = Monitor(train_env, log_dir)
    train_env = DummyVecEnv([lambda: train_env])
    train_env = VecNormalize(
        train_env, 
        norm_obs=True, 
        norm_reward=True, 
        clip_obs=10.0,
        clip_reward=10.0
    )
    
    eval_env = make_env(
        df=val_df,
        features=features,
        valve_target=valve_target,
        delta_target=delta_target,
        difficulty_level=difficulty_stages[0]
    )()
    
    eval_env = Monitor(eval_env, f"{log_dir}/eval")
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True, 
        norm_reward=False,
        training=False
    )
    
    # Policy hyperparameters
    policy_kwargs = dict(
        net_arch=dict(
            pi=[64, 64],
            vf=[64, 64]
        ),
        activation_fn=nn.Tanh
    )
    
    # Create model
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=0.0003,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="auto"
    )
    
    # Set up callbacks
    curriculum_callback = SimpleCurriculumCallback(
        eval_env=eval_env,
        log_dir=model_dir,
        eval_freq=5000,
        difficulty_stages=difficulty_stages,
        stage_duration=25000,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=model_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    try:
        logger.info(f"\nTraining for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[curriculum_callback, checkpoint_callback],
            tb_log_name="optimized_ppo",
            reset_num_timesteps=True
        )
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        # Save partial model
        try:
            model.save(f"{model_dir}/error_recovery_model")
            train_env.save(f"{model_dir}/error_recovery_vec_normalize.pkl")
            logger.info("Saved recovery model")
        except:
            pass
        raise
    
    # Save final model
    final_model_path = f"{model_dir}/final_model"
    model.save(final_model_path)
    train_env.save(f"{model_dir}/vec_normalize.pkl")
    
    # Save metadata
    with open(f"{model_dir}/training_info.json", "w") as f:
        json.dump({
            'features': features,
            'difficulty_stages': difficulty_stages,
            'total_timesteps': total_timesteps,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=4)
    
    logger.info(f"Model saved to {final_model_path}")
    
    return model, train_env, eval_env
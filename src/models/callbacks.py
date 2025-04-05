import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import logging

logger = logging.getLogger(__name__)

class SimpleCurriculumCallback(BaseCallback):
    """
    Curriculum learning callback with evaluation logic
    """
    def __init__(self, eval_env, log_dir, eval_freq=5000, difficulty_stages=None, 
                 stage_duration=10000, verbose=1):
        super(SimpleCurriculumCallback, self).__init__(verbose)
        
        self.eval_env = eval_env
        self.difficulty_stages = difficulty_stages or [0, 2, 4]
        self.current_stage_idx = 0
        self.stage_duration = stage_duration
        self.log_dir = log_dir
        self.eval_freq = eval_freq
        
        self.stage_start_step = 0
        self.best_mean_error = float('inf')
        self.evaluation_results = []
        
        self.is_evaluating = False
        self.max_eval_episodes = 5
        
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_step(self):
        """
        Called on each training step
        """
        if self.is_evaluating:
            return True
        
        current_difficulty = self.difficulty_stages[self.current_stage_idx]
        steps_in_stage = self.n_calls - self.stage_start_step
        
        # Advance difficulty if needed
        if (steps_in_stage >= self.stage_duration and 
            self.current_stage_idx < len(self.difficulty_stages) - 1):
            
            self.current_stage_idx += 1
            new_difficulty = self.difficulty_stages[self.current_stage_idx]
            self.stage_start_step = self.n_calls
            
            if self.verbose > 0:
                logger.info(f"\nAdvancing to difficulty level {new_difficulty} at step {self.n_calls}")
            
            training_env = self.model.get_env()
            training_env.env_method('set_difficulty', new_difficulty)
        
        # Evaluate periodically
        if self.n_calls % self.eval_freq == 0:
            try:
                self.is_evaluating = True
                self._evaluate()
                self.is_evaluating = False
            except Exception as e:
                logger.error(f"Evaluation error: {e}")
                self.is_evaluating = False
        
        return True
    
    def _evaluate(self):
        """
        Evaluate model performance
        """
        current_difficulty = self.difficulty_stages[self.current_stage_idx]
        self.eval_env.env_method('set_difficulty', current_difficulty)
        
        episode_errors = []
        episode_rewards = []
        
        for _ in range(self.max_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            episode_error = []
            
            step_count = 0
            max_eval_steps = 100
            
            while not done and step_count < max_eval_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                
                episode_reward += reward
                if 'absolute_error' in info:
                    episode_error.append(info['absolute_error'])
                
                step_count += 1
            
            episode_rewards.append(episode_reward)
            if episode_error:
                episode_errors.append(np.mean(episode_error))
        
        mean_error = np.mean(episode_errors) if episode_errors else float('inf')
        mean_reward = np.mean(episode_rewards)
        
        if self.verbose > 0:
            logger.info(f"\nEvaluation at step {self.n_calls} (difficulty {current_difficulty}):")
            logger.info(f"  Mean error: {mean_error:.2f}")
            logger.info(f"  Mean reward: {mean_reward:.2f}")
        
        # Store results
        self.evaluation_results.append({
            'step': self.n_calls,
            'difficulty': current_difficulty,
            'mean_error': float(mean_error),
            'mean_reward': float(mean_reward)
        })
        
        # Save best model
        if mean_error < self.best_mean_error:
            self.best_mean_error = mean_error
            best_model_path = f"{self.log_dir}/best_model"
            self.model.save(best_model_path)
            
            if self.verbose > 0:
                logger.info(f"  New best model saved with mean error: {mean_error:.2f}")
            
            try:
                self.eval_env.save(f"{self.log_dir}/vec_normalize.pkl")
            except:
                pass
        
        # Save evaluation results
        results_df = pd.DataFrame(self.evaluation_results)
        results_df.to_csv(f"{self.log_dir}/eval_results.csv", index=False)
        
        if len(self.evaluation_results) > 1:
            self._plot_learning_curve()
    
    def _plot_learning_curve(self):
        """
        Plot learning curves
        """
        try:
            steps = [r['step'] for r in self.evaluation_results]
            errors = [r['mean_error'] for r in self.evaluation_results]
            rewards = [r['mean_reward'] for r in self.evaluation_results]
            
            plt.figure(figsize=(12, 5))
            
            # Error plot
            plt.subplot(1, 2, 1)
            plt.plot(steps, errors, 'r-o')
            plt.xlabel('Training Steps')
            plt.ylabel('Mean Absolute Error')
            plt.title('Evaluation Error')
            plt.grid(True, alpha=0.3)
            
            # Reward plot
            plt.subplot(1, 2, 2)
            plt.plot(steps, rewards, 'b-o')
            plt.xlabel('Training Steps')
            plt.ylabel('Mean Reward')
            plt.title('Evaluation Reward')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{self.log_dir}/learning_curve.png")
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting learning curve: {e}")
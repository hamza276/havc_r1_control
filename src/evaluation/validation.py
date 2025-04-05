import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..environment.hvac_env import OptimizedHVACEnv
import logging

logger = logging.getLogger(__name__)

def validate_ppo_model(model, test_df, features, valve_target, delta_target, output_dir='validation'):
    """
    Validate the model with enhanced error handling and visualization
    """
    logger.info("\nValidating model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create validation environment
        env = OptimizedHVACEnv(
            df=test_df,
            features=features,
            valve_target=valve_target,
            delta_target=delta_target,
            max_steps=len(test_df) - 1
        )
        
        # Reset environment
        obs, _ = env.reset()
        
        # Storage for results
        actual_valves = []
        predicted_valves = []
        absolute_errors = []
        rewards = []
        
        # Run prediction
        done = False
        step = 0
        max_steps = len(test_df) - 1
        
        while not done and step < max_steps:
            try:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take step
                obs, reward, done, _, info = env.step(action)
                
                # Store results
                actual_valves.append(info['actual_valve'])
                predicted_valves.append(info['predicted_valve'])
                absolute_errors.append(info['absolute_error'])
                rewards.append(reward)
                
                step += 1
                
                if step % 100 == 0:
                    logger.info(f"Processed {step}/{max_steps} steps")
                    
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                break
        
        # Calculate metrics
        mae = np.mean(absolute_errors)
        rmse = np.sqrt(np.mean(np.square(absolute_errors)))
        
        # Calculate R² score
        ss_tot = np.sum((np.array(actual_valves) - np.mean(actual_valves))**2)
        ss_res = np.sum(np.square(np.array(actual_valves) - np.array(predicted_valves)))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate accuracy thresholds
        within_5 = 100 * np.mean(np.array(absolute_errors) <= 5)
        within_10 = 100 * np.mean(np.array(absolute_errors) <= 10)
        within_20 = 100 * np.mean(np.array(absolute_errors) <= 20)
        
        # Print metrics
        logger.info("\nValidation Metrics:")
        logger.info(f"Mean Absolute Error (MAE): {mae:.2f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logger.info(f"R² Score: {r2:.4f}")
        logger.info(f"Within 5 units: {within_5:.2f}%")
        logger.info(f"Within 10 units: {within_10:.2f}%")
        logger.info(f"Within 20 units: {within_20:.2f}%")
        
        # Save metrics
        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'within_5': float(within_5),
            'within_10': float(within_10),
            'within_20': float(within_20),
            'num_samples': len(actual_valves)
        }
        
        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        # Save predictions
        results_df = pd.DataFrame({
            'actual_valve': actual_valves,
            'predicted_valve': predicted_valves,
            'absolute_error': absolute_errors,
            'reward': rewards
        })
        
        results_df.to_csv(f"{output_dir}/predictions.csv", index=False)
        
        # Create visualizations
        create_validation_plots(
            actual_valves, 
            predicted_valves, 
            absolute_errors, 
            r2, 
            mae, 
            output_dir
        )
        
        logger.info(f"Validation complete. Results saved to {output_dir}/")
        return metrics
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise

def create_validation_plots(actual_valves, predicted_valves, absolute_errors, r2, mae, output_dir):
    """
    Create validation visualization plots
    """
    try:
        # Prediction vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_valves, predicted_valves, alpha=0.3)
        
        min_val = min(min(actual_valves), min(predicted_valves))
        max_val = max(max(actual_valves), max(predicted_valves))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Valve Position')
        plt.ylabel('Predicted Valve Position')
        plt.title(f'Actual vs Predicted Valve Position (R² = {r2:.4f})')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/actual_vs_predicted.png")
        plt.close()
        
        # Time series
        plt.figure(figsize=(12, 6))
        display_len = min(300, len(actual_valves))
        
        plt.plot(range(display_len), actual_valves[:display_len], 'k-', 
                label='Actual', linewidth=2)
        plt.plot(range(display_len), predicted_valves[:display_len], 'r-', 
                label='Predicted', alpha=0.7)
        
        plt.xlabel('Time Steps')
        plt.ylabel('Valve Position')
        plt.title('Valve Position Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/prediction_timeseries.png")
        plt.close()
        
        # Error histogram
        plt.figure(figsize=(10, 6))
        plt.hist(absolute_errors, bins=20, alpha=0.7)
        plt.axvline(mae, color='r', linestyle='--', label=f'MAE: {mae:.2f}')
        
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/error_histogram.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating validation plots: {e}")
        raise
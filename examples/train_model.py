import os
import logging
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_preprocessing import load_and_preprocess_data
from src.training.train import train_optimized_ppo
from src.evaluation.validate import validate_ppo_model
from src.config.hyperparameters import TRAINING_PARAMS, VALIDATION_PARAMS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Main training script
    """
    try:
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results_optimized_ppo_{timestamp}"
        log_dir = f"{results_dir}/logs"
        model_dir = f"{results_dir}/models"
        validation_dir = f"{results_dir}/validation"
        
        for directory in [results_dir, log_dir, model_dir, validation_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Load and preprocess data
        logger.info("\n=== Data Preprocessing ===")
        df, features, valve_target, delta_target, scaler = load_and_preprocess_data(
            equip_file='data/Equip.csv',
            weather_file='data/weather.csv'
        )
        
        # Split data
        logger.info("\n=== Data Splitting ===")
        test_size = int(VALIDATION_PARAMS['test_size'] * len(df))
        val_size = int(VALIDATION_PARAMS['val_size'] * len(df))
        
        test_df = df.iloc[:test_size].copy().reset_index(drop=True)
        val_df = df.iloc[test_size:test_size+val_size].copy().reset_index(drop=True)
        train_df = df.iloc[test_size+val_size:].copy().reset_index(drop=True)
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Validation set: {len(val_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Train model
        logger.info("\n=== Training Model ===")
        model, train_env, eval_env = train_optimized_ppo(
            train_df=train_df,
            val_df=val_df,
            features=features,
            valve_target=valve_target,
            delta_target=delta_target,
            log_dir=log_dir,
            model_dir=model_dir,
            total_timesteps=TRAINING_PARAMS['total_timesteps']
        )
        
        # Validate model
        logger.info("\n=== Validating Model ===")
        metrics = validate_ppo_model(
            model=model,
            test_df=test_df,
            features=features,
            valve_target=valve_target,
            delta_target=delta_target,
            output_dir=validation_dir
        )
        
        # Generate final report
        logger.info("\n=== Generating Final Report ===")
        with open(f"{results_dir}/final_report.txt", "w") as f:
            f.write("=== HVAC Control PPO Training Report ===\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=== Data Information ===\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Training samples: {len(train_df)}\n")
            f.write(f"Validation samples: {len(val_df)}\n")
            f.write(f"Test samples: {len(test_df)}\n")
            f.write(f"Features used: {len(features)}\n\n")
            
            f.write("=== Model Performance ===\n")
            f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.2f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f}\n")
            f.write(f"R² Score: {metrics['r2']:.4f}\n")
            f.write(f"Within 5 units: {metrics['within_5']:.2f}%\n")
            f.write(f"Within 10 units: {metrics['within_10']:.2f}%\n")
            f.write(f"Within 20 units: {metrics['within_20']:.2f}%\n")
        
        logger.info(f"Training complete! Results saved to {results_dir}/")
        
    except Exception as e:
        logger.error(f"Error in training process: {e}")
        raise

if __name__ == "__main__":
    main()
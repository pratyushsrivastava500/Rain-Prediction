"""
Model management module for Rain Prediction.
Handles model loading, prediction, and model-related operations.
"""
import pickle
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import logging
from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages machine learning model operations."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the model manager.
        
        Args:
            model_path: Path to the trained model file. If None, uses default from config.
        """
        self.model_path = model_path or Config.MODEL_PATH
        self.model = None
        self.is_loaded = False
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        try:
            model_file = Path(self.model_path)
            
            if not model_file.exists():
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. "
                    f"Please ensure the model is trained and saved."
                )
            
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.is_loaded = True
            logger.info(f"✓ Model loaded successfully from {self.model_path}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"✗ Model file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"✗ Error loading model: {e}")
            raise Exception(f"Failed to load model: {e}")
    
    def predict(self, features: List[float]) -> int:
        """
        Make a prediction using the loaded model.
        
        Args:
            features: List of feature values in correct order
            
        Returns:
            Prediction (0 for sunny, 1 for rainy)
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If features are invalid
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                "Model not loaded. Please load the model before making predictions."
            )
        
        try:
            # Convert to numpy array and reshape for single prediction
            features_array = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features_array)
            
            # Return the prediction (assuming it returns array)
            if isinstance(prediction, np.ndarray):
                return int(prediction[0])
            return int(prediction)
            
        except Exception as e:
            logger.error(f"✗ Prediction error: {e}")
            raise ValueError(f"Error making prediction: {e}")
    
    def predict_proba(self, features: List[float]) -> Dict[str, float]:
        """
        Get prediction probabilities if model supports it.
        
        Args:
            features: List of feature values
            
        Returns:
            Dictionary with probabilities for each class
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            features_array = np.array(features).reshape(1, -1)
            
            # Check if model has predict_proba method
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_array)[0]
                return {
                    'sunny_probability': float(probabilities[0]),
                    'rainy_probability': float(probabilities[1])
                }
            else:
                # If predict_proba not available, return binary prediction
                prediction = self.predict(features)
                return {
                    'sunny_probability': 1.0 if prediction == 0 else 0.0,
                    'rainy_probability': 0.0 if prediction == 0 else 1.0
                }
                
        except Exception as e:
            logger.error(f"✗ Error getting probabilities: {e}")
            raise
    
    def get_prediction_label(self, prediction: int) -> str:
        """
        Convert prediction to human-readable label.
        
        Args:
            prediction: Model prediction (0 or 1)
            
        Returns:
            Label string ('sunny' or 'rainy')
        """
        return Config.PREDICTION_LABELS.get(prediction, "unknown")
    
    def predict_with_details(self, features: List[float]) -> Dict[str, Any]:
        """
        Make prediction and return detailed results.
        
        Args:
            features: List of feature values
            
        Returns:
            Dictionary containing prediction, label, and probabilities
        """
        try:
            # Make prediction
            prediction = self.predict(features)
            label = self.get_prediction_label(prediction)
            
            # Get probabilities if available
            try:
                probabilities = self.predict_proba(features)
            except:
                probabilities = None
            
            result = {
                'prediction': prediction,
                'label': label,
                'is_rainy': prediction == 1,
                'is_sunny': prediction == 0,
                'probabilities': probabilities
            }
            
            logger.info(f"Prediction: {label} (confidence: {probabilities})")
            return result
            
        except Exception as e:
            logger.error(f"✗ Error in detailed prediction: {e}")
            raise
    
    def reload_model(self) -> bool:
        """
        Reload the model from disk.
        
        Returns:
            True if successful
        """
        logger.info("Reloading model...")
        self.is_loaded = False
        self.model = None
        return self.load_model()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {'loaded': False, 'message': 'Model not loaded'}
        
        info = {
            'loaded': True,
            'model_path': self.model_path,
            'model_type': type(self.model).__name__,
            'model_name': Config.MODEL_NAME
        }
        
        # Add additional info if available
        if hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        return info

"""
Configuration module for Rain Prediction application.
Contains all paths, constants, and settings used across the application.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Application configuration class."""
    
    # Flask Configuration
    DEBUG = True
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Paths
    TEMPLATE_FOLDER = str(BASE_DIR / "template")
    STATIC_FOLDER = str(BASE_DIR / "static")
    MODEL_PATH = str(BASE_DIR / "models" / "cat.pkl")
    DATA_PATH = str(BASE_DIR / "data" / "weatherAUS.csv")
    
    # Model Configuration
    MODEL_NAME = "CatBoost Rain Predictor"
    CONFIDENCE_THRESHOLD = 0.5
    
    # Feature Configuration
    FEATURE_NAMES = [
        'location', 'mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine',
        'windgustdir', 'windgustspeed', 'winddir9am', 'winddir3pm', 
        'windspeed9am', 'windspeed3pm', 'humidity9am', 'humidity3pm',
        'pressure9am', 'pressure3pm', 'cloud9am', 'cloud3pm',
        'temp9am', 'temp3pm', 'raintoday', 'month', 'day'
    ]
    
    # Prediction Labels
    PREDICTION_LABELS = {
        0: "sunny",
        1: "rainy"
    }
    
    # Date Format
    DATE_FORMAT = "%Y-%m-%dT"
    
    # API Configuration
    CORS_ORIGINS = ["*"]
    
    @staticmethod
    def init_app(app):
        """Initialize application with this config."""
        pass


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    @classmethod
    def init_app(cls, app):
        """Initialize production settings."""
        Config.init_app(app)


class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    DEBUG = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

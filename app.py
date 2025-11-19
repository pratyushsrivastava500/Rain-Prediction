"""
Rain Prediction Flask Application
A modular web application for predicting weather conditions using machine learning.
"""
from flask import Flask
from flask_cors import CORS
import logging
import os

from config.config import config, Config
from src.model import ModelManager
from src.routes import register_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app(config_name: str = None) -> Flask:
    """
    Application factory function.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
        
    Returns:
        Configured Flask application instance
    """
    # Determine configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    # Create Flask application
    app = Flask(
        __name__,
        template_folder=Config.TEMPLATE_FOLDER,
        static_folder=Config.STATIC_FOLDER
    )
    
    # Load configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Enable CORS
    CORS(app, origins=Config.CORS_ORIGINS)
    
    logger.info(f"Initializing Rain Prediction Application ({config_name} mode)")
    
    # Initialize model manager
    try:
        model_manager = ModelManager()
        logger.info("✓ Model manager initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize model manager: {e}")
        raise
    
    # Register routes
    register_routes(app, model_manager)
    
    logger.info("✓ Application initialization complete")
    
    return app


# Create application instance
app = create_app()


if __name__ == '__main__':
    # Get configuration from environment
    debug_mode = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    
    logger.info(f"Starting server on {host}:{port} (debug={debug_mode})")
    
    # Run application
    app.run(
        host=host,
        port=port,
        debug=debug_mode
    )
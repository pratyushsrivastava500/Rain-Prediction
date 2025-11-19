"""
Routes module for Rain Prediction Flask application.
Contains all route handlers and view functions.
"""
from flask import render_template, request, jsonify
from flask_cors import cross_origin
import logging
from typing import Dict, Any

from src.data_preprocessing import DataPreprocessor
from src.model import ModelManager
from src.utils import sanitize_form_data, log_prediction, format_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RouteHandlers:
    """Class to handle all Flask routes."""
    
    def __init__(self, model_manager: ModelManager):
        """
        Initialize route handlers with model manager.
        
        Args:
            model_manager: Instance of ModelManager for predictions
        """
        self.model_manager = model_manager
        self.data_preprocessor = DataPreprocessor()
    
    @cross_origin()
    def home(self):
        """
        Home page route handler.
        
        Returns:
            Rendered index.html template
        """
        logger.info("Home page accessed")
        return render_template("index.html")
    
    @cross_origin()
    def predict(self):
        """
        Prediction route handler.
        Handles both GET (show form) and POST (process prediction) requests.
        
        Returns:
            - GET: Rendered predictor.html template
            - POST: Rendered result template (after_sunny.html or after_rainy.html)
        """
        if request.method == "POST":
            try:
                # Get and sanitize form data
                form_data = sanitize_form_data(request.form.to_dict())
                logger.info("Received prediction request")
                
                # Prepare features from form data
                features = self.data_preprocessor.prepare_features(form_data)
                
                # Validate feature ranges
                self.data_preprocessor.validate_feature_range(features)
                
                # Make prediction
                prediction_result = self.model_manager.predict_with_details(features)
                
                # Log prediction
                log_prediction(
                    features,
                    prediction_result['prediction'],
                    prediction_result['label']
                )
                
                # Return appropriate template based on prediction
                if prediction_result['prediction'] == 0:
                    logger.info("Prediction: Sunny weather")
                    return render_template("after_sunny.html")
                else:
                    logger.info("Prediction: Rainy weather")
                    return render_template("after_rainy.html")
                    
            except ValueError as e:
                logger.error(f"Validation error: {e}")
                return render_template(
                    "predictor.html",
                    error=f"Invalid input: {str(e)}"
                )
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return render_template(
                    "predictor.html",
                    error="An error occurred during prediction. Please try again."
                )
        
        # GET request - show prediction form
        logger.info("Prediction form accessed")
        return render_template("predictor.html")
    
    @cross_origin()
    def predict_api(self):
        """
        API endpoint for prediction (JSON response).
        
        Returns:
            JSON response with prediction results
        """
        try:
            # Get JSON data from request
            data = request.get_json()
            
            if not data:
                return jsonify(format_response(
                    False,
                    "No data provided in request"
                )), 400
            
            logger.info("API prediction request received")
            
            # Prepare features
            features = self.data_preprocessor.prepare_features(data)
            
            # Validate features
            self.data_preprocessor.validate_feature_range(features)
            
            # Make prediction
            prediction_result = self.model_manager.predict_with_details(features)
            
            # Log prediction
            log_prediction(
                features,
                prediction_result['prediction'],
                prediction_result['label']
            )
            
            # Return JSON response
            return jsonify(format_response(
                True,
                "Prediction successful",
                prediction_result
            )), 200
            
        except ValueError as e:
            logger.error(f"Validation error in API: {e}")
            return jsonify(format_response(False, f"Invalid input: {str(e)}")), 400
            
        except Exception as e:
            logger.error(f"API prediction error: {e}")
            return jsonify(format_response(
                False,
                "An error occurred during prediction"
            )), 500
    
    @cross_origin()
    def health_check(self):
        """
        Health check endpoint.
        
        Returns:
            JSON response with service status
        """
        model_info = self.model_manager.get_model_info()
        
        response = {
            'status': 'healthy' if model_info['loaded'] else 'unhealthy',
            'service': 'Rain Prediction API',
            'model': model_info
        }
        
        status_code = 200 if model_info['loaded'] else 503
        return jsonify(response), status_code
    
    @cross_origin()
    def model_info(self):
        """
        Get model information endpoint.
        
        Returns:
            JSON response with detailed model information
        """
        try:
            info = self.model_manager.get_model_info()
            return jsonify(format_response(True, "Model info retrieved", info)), 200
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return jsonify(format_response(False, "Error retrieving model info")), 500


def register_routes(app, model_manager: ModelManager):
    """
    Register all routes with the Flask application.
    
    Args:
        app: Flask application instance
        model_manager: ModelManager instance
    """
    handlers = RouteHandlers(model_manager)
    
    # Web routes
    app.add_url_rule('/', 'home', handlers.home, methods=['GET'])
    app.add_url_rule('/predict', 'predict', handlers.predict, methods=['GET', 'POST'])
    
    # API routes
    app.add_url_rule('/api/predict', 'predict_api', handlers.predict_api, methods=['POST'])
    app.add_url_rule('/api/health', 'health_check', handlers.health_check, methods=['GET'])
    app.add_url_rule('/api/model-info', 'model_info', handlers.model_info, methods=['GET'])
    
    logger.info("✓ All routes registered successfully")

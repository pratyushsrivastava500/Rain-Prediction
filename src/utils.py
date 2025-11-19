"""
Utility functions for Rain Prediction application.
Contains helper functions used across the application.
"""
import pandas as pd
from datetime import datetime
from typing import Any, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def date_parser(date_string: str, date_format: str = "%Y-%m-%dT") -> datetime:
    """
    Parse date string to datetime object.
    
    Args:
        date_string: Date string to parse
        date_format: Expected date format
        
    Returns:
        Datetime object
        
    Raises:
        ValueError: If date string is invalid
    """
    try:
        return pd.to_datetime(date_string, format=date_format)
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")


def validate_input(data: Dict[str, Any], required_fields: list) -> bool:
    """
    Validate that all required fields are present in input data.
    
    Args:
        data: Input data dictionary
        required_fields: List of required field names
        
    Returns:
        True if all required fields present
        
    Raises:
        ValueError: If required fields are missing
    """
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    return True


def format_response(success: bool, message: str, data: Any = None) -> Dict[str, Any]:
    """
    Format API response in consistent structure.
    
    Args:
        success: Whether operation was successful
        message: Response message
        data: Optional response data
        
    Returns:
        Formatted response dictionary
    """
    response = {
        'success': success,
        'message': message
    }
    
    if data is not None:
        response['data'] = data
    
    return response


def log_prediction(features: list, prediction: int, label: str) -> None:
    """
    Log prediction details for monitoring and debugging.
    
    Args:
        features: Input features used for prediction
        prediction: Model prediction value
        label: Prediction label
    """
    logger.info(f"Prediction made: {label} (value: {prediction})")
    logger.debug(f"Input features: {features}")


def sanitize_form_data(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and clean form data.
    
    Args:
        form_data: Raw form data
        
    Returns:
        Sanitized form data
    """
    sanitized = {}
    
    for key, value in form_data.items():
        # Remove leading/trailing whitespace from strings
        if isinstance(value, str):
            sanitized[key] = value.strip()
        else:
            sanitized[key] = value
    
    return sanitized


def get_feature_importance_mapping() -> Dict[str, str]:
    """
    Get human-readable names for features.
    
    Returns:
        Dictionary mapping feature codes to readable names
    """
    return {
        'location': 'Location',
        'mintemp': 'Minimum Temperature',
        'maxtemp': 'Maximum Temperature',
        'rainfall': 'Rainfall',
        'evaporation': 'Evaporation',
        'sunshine': 'Sunshine Hours',
        'windgustdir': 'Wind Gust Direction',
        'windgustspeed': 'Wind Gust Speed',
        'winddir9am': 'Wind Direction 9am',
        'winddir3pm': 'Wind Direction 3pm',
        'windspeed9am': 'Wind Speed 9am',
        'windspeed3pm': 'Wind Speed 3pm',
        'humidity9am': 'Humidity 9am',
        'humidity3pm': 'Humidity 3pm',
        'pressure9am': 'Pressure 9am',
        'pressure3pm': 'Pressure 3pm',
        'cloud9am': 'Cloud Cover 9am',
        'cloud3pm': 'Cloud Cover 3pm',
        'temp9am': 'Temperature 9am',
        'temp3pm': 'Temperature 3pm',
        'raintoday': 'Rain Today',
        'month': 'Month',
        'day': 'Day'
    }


def calculate_confidence_score(probabilities: Dict[str, float]) -> float:
    """
    Calculate confidence score from probabilities.
    
    Args:
        probabilities: Dictionary with class probabilities
        
    Returns:
        Confidence score (0-100)
    """
    if not probabilities:
        return 0.0
    
    # Get the maximum probability
    max_prob = max(probabilities.values())
    return round(max_prob * 100, 2)


def format_temperature(temp: float, unit: str = 'C') -> str:
    """
    Format temperature with unit.
    
    Args:
        temp: Temperature value
        unit: Temperature unit ('C' or 'F')
        
    Returns:
        Formatted temperature string
    """
    return f"{temp:.1f}°{unit}"


def is_extreme_weather(features: list) -> Dict[str, bool]:
    """
    Check for extreme weather conditions in features.
    
    Args:
        features: List of weather features
        
    Returns:
        Dictionary with extreme condition flags
    """
    return {
        'extreme_temperature': features[1] < -10 or features[2] > 45,  # min/max temp
        'high_rainfall': features[3] > 50,  # rainfall
        'strong_winds': features[7] > 70,  # wind gust speed
        'high_humidity': features[12] > 90 or features[13] > 90  # humidity 9am/3pm
    }


def validate_temperature_consistency(min_temp: float, max_temp: float) -> bool:
    """
    Validate that maximum temperature is greater than minimum.
    
    Args:
        min_temp: Minimum temperature
        max_temp: Maximum temperature
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If max temp is less than min temp
    """
    if max_temp < min_temp:
        raise ValueError(
            f"Maximum temperature ({max_temp}°C) cannot be less than "
            f"minimum temperature ({min_temp}°C)"
        )
    return True

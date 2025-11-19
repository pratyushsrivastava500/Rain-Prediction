"""
Data preprocessing module for Rain Prediction.
Handles input validation, feature extraction, and data transformation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from config.config import Config


class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.feature_names = Config.FEATURE_NAMES
    
    def extract_date_features(self, date_string: str) -> Tuple[float, float]:
        """
        Extract day and month from date string.
        
        Args:
            date_string: Date in format YYYY-MM-DD
            
        Returns:
            Tuple of (day, month) as floats
        """
        try:
            date_obj = pd.to_datetime(date_string, format=Config.DATE_FORMAT)
            day = float(date_obj.day)
            month = float(date_obj.month)
            return day, month
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")
    
    def validate_numeric_input(self, value: Any, field_name: str) -> float:
        """
        Validate and convert input to float.
        
        Args:
            value: Input value to validate
            field_name: Name of the field for error messages
            
        Returns:
            Validated float value
            
        Raises:
            ValueError: If value cannot be converted to float
        """
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value for {field_name}: {value}")
    
    def prepare_features(self, form_data: Dict[str, Any]) -> List[float]:
        """
        Prepare features from form data for model prediction.
        
        Args:
            form_data: Dictionary containing form inputs
            
        Returns:
            List of features in the correct order for model
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            # Extract date features
            date = form_data.get('date')
            if not date:
                raise ValueError("Date is required")
            day, month = self.extract_date_features(date)
            
            # Extract and validate all features
            location = self.validate_numeric_input(form_data.get('location'), 'location')
            min_temp = self.validate_numeric_input(form_data.get('mintemp'), 'minimum temperature')
            max_temp = self.validate_numeric_input(form_data.get('maxtemp'), 'maximum temperature')
            rainfall = self.validate_numeric_input(form_data.get('rainfall'), 'rainfall')
            evaporation = self.validate_numeric_input(form_data.get('evaporation'), 'evaporation')
            sunshine = self.validate_numeric_input(form_data.get('sunshine'), 'sunshine')
            
            # Wind features
            wind_gust_dir = self.validate_numeric_input(form_data.get('windgustdir'), 'wind gust direction')
            wind_gust_speed = self.validate_numeric_input(form_data.get('windgustspeed'), 'wind gust speed')
            wind_dir_9am = self.validate_numeric_input(form_data.get('winddir9am'), 'wind direction 9am')
            wind_dir_3pm = self.validate_numeric_input(form_data.get('winddir3pm'), 'wind direction 3pm')
            wind_speed_9am = self.validate_numeric_input(form_data.get('windspeed9am'), 'wind speed 9am')
            wind_speed_3pm = self.validate_numeric_input(form_data.get('windspeed3pm'), 'wind speed 3pm')
            
            # Humidity and pressure features
            humidity_9am = self.validate_numeric_input(form_data.get('humidity9am'), 'humidity 9am')
            humidity_3pm = self.validate_numeric_input(form_data.get('humidity3pm'), 'humidity 3pm')
            pressure_9am = self.validate_numeric_input(form_data.get('pressure9am'), 'pressure 9am')
            pressure_3pm = self.validate_numeric_input(form_data.get('pressure3pm'), 'pressure 3pm')
            
            # Cloud and temperature features
            cloud_9am = self.validate_numeric_input(form_data.get('cloud9am'), 'cloud 9am')
            cloud_3pm = self.validate_numeric_input(form_data.get('cloud3pm'), 'cloud 3pm')
            temp_9am = self.validate_numeric_input(form_data.get('temp9am'), 'temperature 9am')
            temp_3pm = self.validate_numeric_input(form_data.get('temp3pm'), 'temperature 3pm')
            
            # Rain today
            rain_today = self.validate_numeric_input(form_data.get('raintoday'), 'rain today')
            
            # Prepare feature list in correct order
            features = [
                location, min_temp, max_temp, rainfall, evaporation, sunshine,
                wind_gust_dir, wind_gust_speed, wind_dir_9am, wind_dir_3pm,
                wind_speed_9am, wind_speed_3pm, humidity_9am, humidity_3pm,
                pressure_9am, pressure_3pm, cloud_9am, cloud_3pm,
                temp_9am, temp_3pm, rain_today, month, day
            ]
            
            return features
            
        except KeyError as e:
            raise ValueError(f"Missing required field: {e}")
        except Exception as e:
            raise ValueError(f"Error preparing features: {e}")
    
    def validate_feature_range(self, features: List[float]) -> bool:
        """
        Validate that features are within reasonable ranges.
        
        Args:
            features: List of feature values
            
        Returns:
            True if all features are valid
            
        Raises:
            ValueError: If features are out of reasonable range
        """
        # Basic validation - can be extended with specific ranges
        if any(pd.isna(features)):
            raise ValueError("Features contain NaN values")
        
        # Temperature validation (-50 to 60 Celsius)
        temp_indices = [1, 2, 18, 19]  # minTemp, maxTemp, temp9am, temp3pm
        for idx in temp_indices:
            if not -50 <= features[idx] <= 60:
                raise ValueError(f"Temperature value out of range: {features[idx]}")
        
        # Humidity validation (0-100%)
        humidity_indices = [12, 13]  # humidity9am, humidity3pm
        for idx in humidity_indices:
            if not 0 <= features[idx] <= 100:
                raise ValueError(f"Humidity value out of range: {features[idx]}")
        
        return True
    
    def get_feature_dataframe(self, features: List[float]) -> pd.DataFrame:
        """
        Convert feature list to DataFrame with proper column names.
        
        Args:
            features: List of feature values
            
        Returns:
            DataFrame with named columns
        """
        return pd.DataFrame([features], columns=self.feature_names)

# 🌧️ Rain Prediction Machine Learning System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Flask](https://img.shields.io/badge/Flask-2.0%2B-green) ![CatBoost](https://img.shields.io/badge/CatBoost-0.25-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

> An intelligent weather prediction system using machine learning to forecast rain with high accuracy. Built with clean architecture, modular design, and production-ready code.

## 📋 Overview

The Rain Prediction System enables users to:

• **Accurate Weather Forecasting** - Predict rain with ~85-90% accuracy using advanced machine learning  
• **Real-time Predictions** - Get instant weather forecasts based on comprehensive meteorological data  
• **User-Friendly Interface** - Clean web interface for easy data input and result visualization  
• **API Support** - RESTful API endpoints for integration with other applications  
• **Modular Architecture** - Clean, maintainable code structure for easy extension

The system uses the **CatBoost** algorithm trained on Australian weather data to predict whether it will rain tomorrow based on today's weather conditions.

## ✨ Features

### 🎯 Machine Learning Technology

• **High Accuracy** - CatBoost classifier with ~85-90% accuracy on test data  
• **Comprehensive Features** - Uses 23 weather parameters for prediction  
• **Robust Model** - Handles missing values and categorical features efficiently  
• **Multiple Algorithms** - Supports CatBoost, XGBoost, Random Forest, and more  
• **SMOTE Balancing** - Handles imbalanced datasets with oversampling techniques

### 🏗️ Clean Architecture

• **Modular Design** - Separation of concerns with config, src, and templates  
• **Type Hints** - Comprehensive type annotations for better code quality  
• **Error Handling** - Production-ready exception handling and logging  
• **Configuration Management** - Centralized config for easy customization  
• **Cross-platform** - Compatible path handling for Windows, Linux, and macOS

### 💻 User Experience

• **Interactive Web UI** - Built with Flask and modern HTML/CSS  
• **Responsive Design** - Works seamlessly on desktop and mobile devices  
• **Visual Feedback** - Clear prediction results with sunny/rainy templates  
• **Input Validation** - Comprehensive validation for all user inputs  
• **Error Messages** - Helpful error messages for troubleshooting

### 🔒 API Features

• **RESTful API** - JSON-based API for programmatic access  
• **Health Checks** - Monitor service health and model status  
• **CORS Support** - Enable cross-origin requests for web apps  
• **Detailed Responses** - Includes predictions, probabilities, and confidence scores

## 🚀 Quick Start

### Prerequisites

• Python 3.8 or higher  
• pip package manager  
• 4GB RAM minimum  
• Internet connection for installation

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Rain-Prediction.git
cd Rain-Prediction
```

2. **Create virtual environment** (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
python app.py
```

5. **Access the application**

• Web Interface: `http://localhost:5000`  
• API Health Check: `http://localhost:5000/api/health`

### Training a New Model

To train a new model with your own data:

```python
# Open and run RainPrediction2.ipynb
jupyter notebook RainPrediction2.ipynb

# The notebook will:
# 1. Load and preprocess data
# 2. Handle missing values and outliers
# 3. Train multiple models (CatBoost, XGBoost, etc.)
# 4. Save the best model to models/cat.pkl
```

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      Flask Web Interface            │
│  • HTML Templates                   │
│  • Static assets (CSS/images)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│        Routes Layer                 │
│  • Route handlers                   │
│  • Request/response management      │
│  • API endpoints                    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     Business Logic Layer            │
│  • Data preprocessing               │
│  • Feature validation               │
│  • Model prediction                 │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    Model Management Layer           │
│  • Model loading                    │
│  • Prediction engine                │
│  • Probability calculation          │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│    Configuration Layer              │
│  • Settings & paths                 │
│  • Feature definitions              │
│  • Environment configs              │
└─────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.8+ |
| **Web Framework** | Flask | 2.0+ |
| **ML Algorithm** | CatBoost | 0.25 |
| **Data Processing** | Pandas, NumPy | Latest |
| **Visualization** | Matplotlib, Seaborn | Latest |
| **CORS Support** | Flask-CORS | 3.0+ |
| **Imbalanced Learning** | imbalanced-learn | 0.8+ |
| **Model Persistence** | Pickle/Joblib | Built-in |

## 📁 Project Structure

```
Rain-Prediction/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── Procfile                    # Heroku deployment config
├── README.md                   # This file
├── weatherAUS.csv              # Australian weather dataset
├── RainPrediction2.ipynb       # Main training notebook
│
├── config/
│   ├── __init__.py
│   └── config.py              # Configuration settings
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data handling & validation
│   ├── model.py               # Model management
│   ├── routes.py              # Flask route handlers
│   └── utils.py               # Utility functions
│
├── models/
│   ├── cat.pkl                # CatBoost trained model
│   ├── xgb.pkl                # XGBoost model
│   ├── gnb.pkl                # Gaussian Naive Bayes
│   └── logreg.pkl             # Logistic Regression
│
├── static/
│   ├── style.css              # Main stylesheet
│   ├── style1.css             # Additional styles
│   ├── predictor.css          # Predictor page styles
│   ├── after_rainy.css        # Rainy result styles
│   └── *.png, *.jpg           # Images and assets
│
├── template/
│   ├── index.html             # Home page
│   ├── predictor.html         # Prediction form
│   ├── after_sunny.html       # Sunny result page
│   └── after_rainy.html       # Rainy result page
│
├── testing_notebooks/
│   ├── Prediction.ipynb       # Prediction testing
│   ├── Prepocessing.ipynb     # Data preprocessing
│   └── RainPrediction3.ipynb  # Model experimentation
│
└── catboost_info/             # CatBoost training logs
    ├── learn_error.tsv
    └── time_left.tsv
```

## 📊 Model Information

**Algorithm**: CatBoost (Categorical Boosting)

**Performance Metrics**:

| Metric | Value |
|--------|-------|
| Accuracy | ~85-90% |
| Precision | ~86% |
| Recall | ~84% |
| F1-Score | ~85% |
| Training Time | ~3-5 minutes |

**Input Features** (23 total):

| Category | Features |
|----------|----------|
| **Location** | Location code |
| **Temperature** | Min/Max Temperature, Temperature at 9am/3pm |
| **Precipitation** | Rainfall, Evaporation, Rain Today |
| **Wind** | Wind Direction (9am/3pm), Wind Speed (9am/3pm), Gust Direction, Gust Speed |
| **Atmospheric** | Pressure (9am/3pm), Humidity (9am/3pm), Cloud Cover (9am/3pm) |
| **Environmental** | Sunshine hours |
| **Temporal** | Month, Day |

**Output**: Binary classification (0 = Sunny, 1 = Rainy)

## 📖 Usage Guide

### Web Interface

1. **Navigate to Home Page**
   - Open browser to `http://localhost:5000`
   - Click "Make Prediction"

2. **Enter Weather Data**
   - Fill in all required fields:
     - Date, Location code
     - Temperature readings (min, max, 9am, 3pm)
     - Wind data (speed, direction, gust)
     - Humidity and pressure readings
     - Cloud cover and sunshine hours
     - Current rainfall status

3. **Submit Prediction**
   - Click "Predict" button
   - View results on result page

4. **Interpret Results**
   - Sunny: Clear weather expected tomorrow
   - Rainy: Rain expected tomorrow

### API Usage

**Predict Weather (POST)**

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-11-19T",
    "location": 1,
    "mintemp": 15.5,
    "maxtemp": 25.3,
    "rainfall": 0.2,
    "evaporation": 4.8,
    "sunshine": 8.5,
    "windgustdir": 3,
    "windgustspeed": 35,
    "winddir9am": 2,
    "winddir3pm": 4,
    "windspeed9am": 15,
    "windspeed3pm": 20,
    "humidity9am": 65,
    "humidity3pm": 45,
    "pressure9am": 1015.5,
    "pressure3pm": 1013.2,
    "cloud9am": 4,
    "cloud3pm": 3,
    "temp9am": 18.5,
    "temp3pm": 24.0,
    "raintoday": 0
  }'
```

**Response**:

```json
{
  "success": true,
  "message": "Prediction successful",
  "data": {
    "prediction": 0,
    "label": "sunny",
    "is_rainy": false,
    "is_sunny": true,
    "probabilities": {
      "sunny_probability": 0.87,
      "rainy_probability": 0.13
    }
  }
}
```

**Health Check (GET)**

```bash
curl http://localhost:5000/api/health
```

**Model Info (GET)**

```bash
curl http://localhost:5000/api/model-info
```

### Example Workflow

**Scenario**: Predicting weather for tomorrow

```
Input Data:
- Date: 2024-11-20
- Location: Sydney (code 1)
- Min Temp: 16°C, Max Temp: 24°C
- Rainfall: 0mm
- Humidity: 65% (9am), 45% (3pm)
- Wind Speed: 15 km/h (9am), 20 km/h (3pm)
- Cloud Cover: Moderate (4/8 at 9am)
- Sunshine: 8.5 hours

Processing:
1. Input validation ✓
2. Feature extraction ✓
3. Model prediction ✓

Output:
- Prediction: Sunny (0)
- Confidence: 87%
- Result: No rain expected tomorrow
```

## 🤖 Model Training Process

The model training pipeline includes:

1. **Data Loading** - Load weatherAUS.csv dataset (145k+ records)
2. **Exploratory Analysis** - Visualize distributions and correlations
3. **Data Cleaning** - Handle missing values using random sampling
4. **Feature Engineering** - Encode categorical variables, extract date features
5. **Imbalance Handling** - Apply SMOTE for balanced training
6. **Model Training** - Train multiple algorithms:
   - CatBoost Classifier
   - XGBoost Classifier
   - Random Forest
   - Logistic Regression
   - Gaussian Naive Bayes
   - K-Nearest Neighbors
7. **Model Evaluation** - Compare accuracy, precision, recall, F1-score
8. **Model Selection** - Choose best performing model (CatBoost)
9. **Model Persistence** - Save model to `models/cat.pkl`

## 🔮 Future Enhancements

- [ ] Deep learning models (LSTM for time-series)
- [ ] Multi-day forecast (3-day, 7-day predictions)
- [ ] Location-based auto-fill using weather APIs
- [ ] Mobile application (React Native)
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Database backend (PostgreSQL)
- [ ] User authentication and history
- [ ] Email/SMS notifications
- [ ] Interactive dashboard with charts
- [ ] Model retraining pipeline
- [ ] A/B testing framework

## 🔧 Troubleshooting

**Issue**: `Model file not found`

```bash
# Solution: Train model using notebook
jupyter notebook RainPrediction2.ipynb
# Run all cells to generate models/cat.pkl
```

**Issue**: `ModuleNotFoundError`

```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**Issue**: `Port already in use`

```bash
# Solution: Change port in app.py or kill existing process
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/macOS
lsof -ti:5000 | xargs kill -9
```

**Issue**: `Invalid input format`

- Ensure all numeric fields contain valid numbers
- Date must be in format: YYYY-MM-DD
- Check that no required fields are empty
- Temperature values should be in Celsius
- Humidity should be 0-100%

**Issue**: `Low prediction accuracy`

- Ensure input data quality
- Check for extreme outliers
- Verify feature encoding matches training data
- Consider retraining model with more data

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Coding Standards**:
- Follow PEP 8 style guide
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 🙏 Acknowledgments

• **Scikit-learn** community for machine learning tools  
• **CatBoost** team for the excellent gradient boosting library  
• **Flask** developers for the lightweight web framework  
• **Australian Government Bureau of Meteorology** for weather data  
• Open source contributors who make projects like this possible

## 📧 Contact

For questions, suggestions, or support:

**Project Maintainer**: Pratyush Srivastava  
**Email**: pratyushsrivastava500@gmail.com  
**GitHub**: [@pratyushsrivastava500](https://github.com/pratyushsrivastava500)

**Issue Tracker**: [GitHub Issues](https://github.com/yourusername/Rain-Prediction/issues)

---

⚠️ **Disclaimer**: This system is designed for educational and research purposes. Weather predictions should not be solely relied upon for critical decisions. Always consult official meteorological services for accurate weather forecasts.

**Made with ❤️ and Python | © 2024 Rain Prediction Team**

---

## 📈 Project Statistics

![GitHub Stars](https://img.shields.io/github/stars/yourusername/Rain-Prediction?style=social)
![GitHub Forks](https://img.shields.io/github/forks/yourusername/Rain-Prediction?style=social)
![GitHub Issues](https://img.shields.io/github/issues/yourusername/Rain-Prediction)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/yourusername/Rain-Prediction)

**Dataset**: 145,460 observations  
**Features**: 23 input features  
**Models**: 4 trained models available  
**Accuracy**: 85-90% on test data  
**Response Time**: < 100ms per prediction

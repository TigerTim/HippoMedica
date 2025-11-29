# 🏥 Multi-Modal Early Disease Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-Ensemble-green.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)

---

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Model Performance](#model-performance)
- [Technology Stack](#technology-stack)

---

## Overview

**HippoMedica** is an AI-powered early disease detection platform that predicts risk for diabetes, heart disease, and stroke using ensemble machine learning. The system combines multiple accessible health indicators to provide accurate risk assessments with explainable AI features.

### Why HippoMedica?

Early detection of chronic diseases is crucial for:
- 💰 Reducing healthcare costs
- 🏥 Enabling timely medical intervention
- 🌍 Bridging healthcare disparities in underserved areas
- 💪 Empowering individuals to take charge of their health

### System Capabilities

**Input:**
- Patient demographics
- Basic health measurements
- Lifestyle factors
- Historical health data

**Output:**
- Disease risk scores with confidence intervals
- Feature importance analysis
- Real-time predictions via web interface

---

## Features

- 🤖 **Ensemble ML Models**: Random Forest, XGBoost, Neural Networks
- 📊 **Explainable AI**: Feature importance visualization and analysis
- 🌐 **Interactive Web App**: Real-time risk assessment via Streamlit
- 🎯 **High Accuracy**: 85%+ accuracy across all disease models
- 🔄 **Automated Pipeline**: End-to-end data processing and model training
- 📈 **Performance Monitoring**: Comprehensive metrics and visualizations

---

## Quick Start


### Prerequisites
- Python 3.8 or higher
- pip package manager

### Run the Web App (Streamlit)

```bash
# 1. Clone the repository
git clone https://github.com/TigerTim/HippoMedica.git
cd HippoMedica

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete ML pipeline (first time only)
cd pipeline
python run_pipeline.py

# 4. Launch the Streamlit web app
cd ../web_app
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8502`

---

## Installation

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `scikit-learn` - Machine learning models
- `xgboost` - Gradient boosting
- `tensorflow/keras` - Neural networks
- `pandas`, `numpy` - Data processing
- `streamlit` - Web application
- `matplotlib`, `seaborn` - Visualization

---

## Usage

### Option 1: Web Application (Recommended)

```bash
# Navigate to web app directory
cd web_app

# Launch Streamlit app
streamlit run app.py
```

**Features:**
- Input patient data through interactive forms
- Get instant risk predictions for multiple diseases
- View feature importance and model explanations
- Compare predictions across different models

### Option 2: Run Complete Pipeline

```bash
# Navigate to pipeline directory
cd pipeline

# Run all steps: Download → EDA → Preprocessing → Training → Storage
python run_pipeline.py

# Or run with options:
python run_pipeline.py --skip-download    # Skip data download
python run_pipeline.py --skip-eda         # Skip exploratory analysis
```

**Pipeline Steps:**
1. **Data Download**: Fetch datasets from UCI/Kaggle
2. **Exploratory Analysis**: Generate visualizations and statistics
3. **Preprocessing**: Clean, impute, and standardize data
4. **Model Training**: Train Random Forest, XGBoost, Neural Networks
5. **Model Storage**: Save models and performance metrics

### Option 3: Individual Scripts

```bash
# Train specific models
python pipeline/04_model_training.py

# Run preprocessing only
python pipeline/03_preprocessing.py

# Generate EDA visualizations
python pipeline/02_exploratory_analysis.py
```

---

## Project Structure

```
HippoMedica/
├── datasets/
│   ├── raw/                    # Original downloaded datasets
│   └── processed/              # Cleaned and preprocessed data
|
├── models/
│   ├── trained_models/         # Saved model files (.pkl)
│   ├── scalers/                # StandardScaler objects
│   └── metadata/               # Performance metrics and feature importance
|
├── pipeline/
│   ├── 01_data_download.py     # Download datasets
│   ├── 02_exploratory_analysis.py
│   ├── 03_preprocessing.py     # Data cleaning and feature engineering
│   ├── 04_model_training.py    # Train all models
│   ├── 05_model_storage.py     # Save models and metadata
│   ├── run_pipeline.py         # Orchestrate entire pipeline
│   └── outputs/                # EDA visualizations
|
├── src/
|   ├── data_acquisition.py
|   ├── data_preprocessing.py
|   ├── knn_baseline.py
|   └── neural_network.py
|
├── web_app/
│   ├── app.py                  # Streamlit web application
|   └── requirements.txt         # Web Dependencies 
|
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Datasets

### Primary Datasets

| Dataset | Source | Samples | Target Disease | Key Features |
|---------|--------|---------|----------------|--------------|
| **Pima Indians Diabetes** | UCI | 768 | Diabetes | Glucose, BMI, Age, Pregnancies |
| **Heart Disease Cleveland** | UCI | 303 | Heart Disease | Chest pain, Cholesterol, ECG |
| **Stroke Prediction** | Kaggle | 5,110 | Stroke | Hypertension, Heart Disease, BMI |

### Secondary Datasets (Validation)

**Diabetes:**
- [UCI Diabetes Dataset](https://archive.ics.uci.edu/dataset/34/diabetes)
- [Figshare Diabetes CSV](https://figshare.com/articles/dataset/diabetes_csv/25421347)
- [Mendeley Dataset 1](https://data.mendeley.com/datasets/m8cgwxs9s6/2)
- [Mendeley Dataset 2](https://data.mendeley.com/datasets/wj9rwkp9c2/1)

**Heart Disease:**
- [IEEE Cardiovascular Dataset](https://ieee-dataport.org/documents/cardiovascular-disease-dataset)

**Stroke:**
- [Mendeley Stroke Dataset](https://data.mendeley.com/datasets/s2nh6fm925/1)

---

## Model Performance

### Diabetes Detection
- **Accuracy**: 85%+
- **Best Model**: Random Forest / XGBoost
- **Cross-Validation**: 5-fold stratified

### Key Metrics
- Precision: High positive prediction accuracy
- Recall: Effective case detection
- F1-Score: Balanced performance
- ROC-AUC: Strong discriminative ability

### Feature Importance (Diabetes)
1. **Glucose** (25.2%) - Most significant predictor
2. **Age** (16.9%) - Strong risk factor
3. **Insulin** (13.8%)
4. **BMI** (11.2%)
5. **DiabetesPedigreeFunction** (11.1%)

---

## Technology Stack

### Machine Learning
- **scikit-learn** - Random Forest, preprocessing, metrics
- **XGBoost** - Gradient boosting models
- **TensorFlow/Keras** - Neural networks
- **GridSearchCV** - Hyperparameter tuning

### Data Processing
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **StandardScaler** - Feature standardization

### Visualization & Deployment
- **Streamlit** - Interactive web application
- **matplotlib** - Static visualizations
- **seaborn** - Statistical graphics
- **plotly** - Interactive charts

### Model Management
- **joblib** - Model serialization
- **json** - Metadata storage

---

## AI Techniques

### Ensemble Learning
1. **Random Forest**
   - Feature aggregation and importance ranking
   - Handles non-linear relationships
   - Robust to overfitting

2. **XGBoost**
   - Gradient boosting for complex patterns
   - Handles class imbalance
   - Superior performance on tabular data

3. **Neural Networks**
   - Deep feature learning
   - Captures non-linear interactions
   - Multi-layer perceptron architecture

### Explainable AI
- Feature importance analysis
- SHAP-style explanations (planned)
- Performance visualization
- Confidence intervals

---

## Success Criteria

**Achieved:**
- Accuracy >85% on test set with proper cross-validation
- Clear identification of top 5 risk factors
- Fast prediction times (<1 second)
- Complete reproducible pipeline

---

## Team

Developed as part of a machine learning group project focused on healthcare AI applications.

---

## Acknowledgments

- UCI Machine Learning Repository for datasets
- Kaggle community for stroke prediction data
- Open source ML community for tools and libraries

---

**Disclaimer**: This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
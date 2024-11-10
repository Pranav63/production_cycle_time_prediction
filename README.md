# Process Manufacturing Cycle Time Prediction

An end-to-end machine learning solution for predicting cycle times in process manufacturing operations, featuring real-time predictions, interactive visualizations, and comprehensive model analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Dictionary](#data-dictionary)
- [Feature Analysis](#feature-analysis)
- [Model Architecture](#model-architecture)
- [Usage Guide](#usage-guide)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a machine learning solution to predict manufacturing cycle times. The system analyzes various production parameters to forecast completion times for manufacturing batches, helping optimize production planning and resource allocation.

https://pctmodel.streamlit.app/

### Primary Objectives
- Predict manufacturing cycle times with high accuracy
- Identify key factors affecting production duration
- Provide real-time insights through interactive visualizations
- Enable data-driven production planning

## ✨ Features

### Core Capabilities
- **Real-time Predictions**: Instant cycle time forecasting
- **Multi-model Comparison**: 
  - Linear Regression (baseline)
  - Random Forest
  - XGBoost
  - LightGBM
- **Interactive Dashboard**: Built with Streamlit
- **Comprehensive Analytics**:
  - Historical performance analysis
  - Feature importance visualization
  - Model performance metrics
  - SHAP value analysis

### Technical Features
- Automated data preprocessing
- Feature engineering
- Model performance comparison
- Interactive visualizations
- Production parameter optimization

## 🚀 Installation

### Prerequisites
```bash
- Python 3.8+
- pip package manager
```

### Setup Steps

1. **Clone Repository**
```bash
git clone https://github.com/yourusername/process-manufacturing-prediction.git
cd process-manufacturing-prediction
```

2. **Create Virtual Environment**
```bash
python -m venv venv

# For Unix/macOS
source venv/bin/activate

# For Windows
venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Application**
```bash
streamlit run PCT_modelling.py
```

## 📊 Data Dictionary

### Input Features

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| Batch_ID | Production batch identifier | String | BATCH_XXXX |
| Date | Production timestamp | Datetime | - |
| Product_Type | Product category | Categorical | Type_A, B, C |
| Batch_Size | Production quantity | Numerical | 100-1000 |
| Raw_Material_Purity | Input material quality | Numerical | 90-100% |
| Temperature_Setting | Process temperature | Numerical | 150-250°C |
| Pressure_Setting | Process pressure | Numerical | 2-10 bar |
| Reactor_Type | Manufacturing unit | Categorical | R1, R2, R3 |
| Equipment_Age | Machinery age | Numerical | 0-10 years |
| Maintenance_Status | Time since maintenance | Numerical | 0-30 days |
| Operator_Shift | Work period | Categorical | Morning/Afternoon/Night |
| Season | Production season | Categorical | Spring/Summer/Fall/Winter |
| Initial_Quality_Score | Starting material quality | Numerical | 70-100 |
| Current_Reactor_Capacity | Unit utilization | Numerical | 60-100% |

### Target Variable
- `Historical_Cycle_Time`: Production completion time (hours)

## 🔍 Feature Analysis

### Batch Size Significance

Batch Size emerges as the most influential feature for cycle time prediction due to several key factors:

#### 1. Physical Impact
- **Processing Time**: Linear correlation with basic operations
- **Equipment Utilization**: Direct impact on machinery usage
- **Material Handling**: Affects movement and setup times

#### 2. Resource Implications
- **Thermal Processing**: Heating/cooling duration scales with size
- **Mixing Requirements**: Larger batches need extended mixing
- **Quality Control**: More extensive testing needed

#### 3. Process Constraints
- **Equipment Capacity**: Utilization optimization
- **Parameter Stabilization**: Larger volumes need more stabilization
- **Setup Requirements**: Preparation time correlation

#### 4. Quality Factors
- **Testing Scope**: More comprehensive quality checks
- **Heat Transfer**: Non-linear scaling effects
- **Mixing Efficiency**: Volume-dependent variations

## 🏗️ Model Architecture

### Model Comparison

1. **Linear Regression (Baseline)**
   - Establishes performance baseline
   - Identifies linear relationships
   - Fast training and prediction

2. **Random Forest**
   - Handles non-linear relationships
   - Feature interaction capture
   - Robust to outliers

3. **XGBoost**
   - Advanced pattern recognition
   - Gradient boosting efficiency
   - High prediction accuracy

4. **LightGBM**
   - Fast training on large datasets
   - Memory efficient
   - Handling of categorical features

## 📱 Usage Guide

### Prediction Interface

1. **Parameter Input**
   - Set production parameters
   - Select product type
   - Input batch specifications

2. **Results Analysis**
   - View multi-model predictions
   - Compare confidence intervals
   - Check feature importance

### Analytics Dashboard

1. **Historical Analysis**
   - Filter by date range
   - View temporal patterns
   - Analyze seasonality

2. **Performance Metrics**
   - Model comparison
   - Error analysis
   - Feature importance plots

## 🤝 Contributing

### Development Process
1. Fork repository
2. Create feature branch
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit changes
   ```bash
   git commit -m "Add YourFeature"
   ```
4. Push to branch
   ```bash
   git push origin feature/YourFeature
   ```
5. Create Pull Request

### Code Standards
- Follow PEP 8 guidelines
- Include docstrings
- Add unit tests
- Update documentation

## 📄 License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md)

## 📚 Additional Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

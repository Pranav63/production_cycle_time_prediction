import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import json
from datetime import datetime, timedelta
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('artifacts', exist_ok=True)

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate dates for historical analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, periods=n_samples, freq='D')
    
    data = {
        'Batch_ID': [f'BATCH_{i:04d}' for i in range(n_samples)],
        'Date': dates,
        'Product_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples),
        'Batch_Size': np.random.uniform(100, 1000, n_samples),
        'Raw_Material_Purity': np.random.normal(95, 2, n_samples),
        'Temperature_Setting': np.random.uniform(150, 250, n_samples),
        'Pressure_Setting': np.random.uniform(2, 10, n_samples),
        'Reactor_Type': np.random.choice(['R1', 'R2', 'R3'], n_samples),
        'Equipment_Age': np.random.uniform(0, 10, n_samples),
        'Maintenance_Status': np.random.uniform(0, 30, n_samples),
        'Operator_Shift': np.random.choice(['Morning', 'Afternoon', 'Night'], n_samples),
        'Season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
        'Initial_Quality_Score': np.random.normal(85, 5, n_samples),
        'Current_Reactor_Capacity': np.random.uniform(60, 100, n_samples)
    }
    
    # Generate target variable with some realistic relationships
    cycle_time = (
        data['Batch_Size'] * 0.1 +
        (100 - data['Raw_Material_Purity']) * 2 +
        np.random.normal(0, 2, n_samples) +
        (data['Equipment_Age'] * 0.5) +
        (data['Maintenance_Status'] * 0.1)
    )
    
    # Add some non-linear relationships and seasonal effects
    cycle_time += np.where(data['Product_Type'] == 'Type_A', 2, 
                          np.where(data['Product_Type'] == 'Type_B', 4, 6))
    
    # Add seasonal variation
    cycle_time += np.where(data['Season'] == 'Winter', 2,
                          np.where(data['Season'] == 'Summer', -1, 0))
    
    data['Historical_Cycle_Time'] = np.maximum(cycle_time, 10)
    
    return pd.DataFrame(data)

def preprocess_data(df):
    # Create copy of dataframe
    df_processed = df.copy()
    
    # Convert date to numerical features
    df_processed['Year'] = df_processed['Date'].dt.year
    df_processed['Month'] = df_processed['Date'].dt.month
    df_processed['DayOfWeek'] = df_processed['Date'].dt.dayofweek
    
    # Label encoding for categorical variables
    categorical_columns = ['Product_Type', 'Reactor_Type', 'Operator_Shift', 'Season']
    label_encoders = {}
    
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df_processed[column] = label_encoders[column].fit_transform(df_processed[column])
    
    # Split features and target
    # Remove original Date column and non-feature columns
    X = df_processed.drop(['Batch_ID', 'Historical_Cycle_Time', 'Date'], axis=1)
    y = df_processed['Historical_Cycle_Time']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled, y, label_encoders, scaler

def train_and_save_models():
    """Train models and save all necessary artifacts"""
    print("Starting training pipeline...")
    
    # Generate and preprocess data
    print("Generating synthetic data...")
    df = generate_synthetic_data(1000)
    X_scaled, y, label_encoders, scaler = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize models
    print("Training models...")
    models = {
        'Linear_Regression': LinearRegression(),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42)
    }
    
    # Train models and collect results
    results = {}
    predictions = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        # Train model
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, f'models/{name.lower()}_model.pkl')
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        predictions[name] = y_pred_test
        
        # Calculate metrics
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'MSE': mean_squared_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'R2': r2_score(y_test, y_pred_test),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred_test) * 100,
            'Train_R2': r2_score(y_train, y_pred_train)
        }
    
    print("Saving artifacts...")
    
    # 1. Save preprocessors
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    # 2. Save feature names and importance
    feature_names = X_scaled.columns.tolist()
    
    # Calculate feature importance for Random Forest
    rf_model = models['Random_Forest']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Save feature importance
    feature_importance.to_pickle('artifacts/feature_importance.pkl')
    
    # 3. Save model results
    with open('artifacts/model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # 4. Save predictions for visualization
    pred_df = pd.DataFrame(predictions)
    pred_df.to_pickle('artifacts/predictions.pkl')
    
    # 5. Save test data for later use
    test_data = {
        'X_test': X_test,
        'y_test': y_test
    }
    joblib.dump(test_data, 'artifacts/test_data.pkl')
    
    # 6. Save sample data for visualizations
    df.to_pickle('artifacts/sample_data.pkl')
    
    # 7. Save feature names
    with open('artifacts/feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    print("""
    Training completed successfully!
    
    Saved artifacts:
    - Models: Linear Regression, Random Forest, XGBoost, LightGBM
    - Preprocessors: Scaler, Label Encoders
    - Feature Importance
    - Model Results
    - Sample Data
    - Test Data
    - Predictions
    - Feature Names
    """)


if __name__ == "__main__":
    train_and_save_models()

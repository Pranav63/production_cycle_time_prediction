import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import tensorflow as tf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import shap
from datetime import datetime, timedelta
# Data Generation
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

# Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        results[name] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }
    
    return results, trained_models

# Feature Importance Analysis
def analyze_feature_importance(model, X):
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = np.abs(model.coef_)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    })
    return feature_importance.sort_values('importance', ascending=False)

def evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    predictions = {}
    
    for name, model in models.items():
        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Train the model
        model.fit(X_train, y_train)
        
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
            'CV_R2_Mean': cv_scores.mean(),
            'CV_R2_Std': cv_scores.std(),
            'Train_R2': r2_score(y_train, y_pred_train)  # Check for overfitting
        }
    
    return results, predictions

# Historical Performance Visualizations
def create_historical_visualizations(df, predictions, y_test):
    visualizations = {}
    
    # 1. Cycle Time Trends
    fig_trends = px.line(df, x='Date', y='Historical_Cycle_Time',
                        color='Product_Type',
                        title='Historical Cycle Time Trends by Product Type')
    visualizations['trends'] = fig_trends
    
    # 2. Cycle Time Distribution
    fig_dist = ff.create_distplot([df['Historical_Cycle_Time']], ['Cycle Time'],
                                 bin_size=2)
    fig_dist.update_layout(title='Cycle Time Distribution')
    visualizations['distribution'] = fig_dist
    
    # 3. Box Plot by Product Type and Season
    fig_box = px.box(df, x='Product_Type', y='Historical_Cycle_Time',
                     color='Season',
                     title='Cycle Time Distribution by Product Type and Season')
    visualizations['box_plot'] = fig_box
    
    # 4. Prediction vs Actual Scatter
    scatter_data = []
    for model_name, pred in predictions.items():
        scatter_data.append(go.Scatter(x=y_test, y=pred,
                                     mode='markers',
                                     name=model_name))
    
    fig_scatter = go.Figure(data=scatter_data)
    fig_scatter.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                   y=[y_test.min(), y_test.max()],
                                   mode='lines',
                                   name='Perfect Prediction',
                                   line=dict(dash='dash')))
    fig_scatter.update_layout(title='Predicted vs Actual Cycle Time',
                            xaxis_title='Actual Cycle Time',
                            yaxis_title='Predicted Cycle Time')
    visualizations['scatter'] = fig_scatter
    
    return visualizations
def create_enhanced_streamlit_ui(df, models, label_encoders, scaler, feature_importance, 
                               model_results, visualizations, X_scaled):
    st.set_page_config(layout="wide")
    st.title('Process Manufacturing Cycle Time Prediction')
    
    # Create tabs for different sections
    tabs = st.tabs(['Prediction', 'Historical Analysis', 'Model Performance', 'Feature Analysis'])
    
    # Tab 1: Prediction
    with tabs[0]:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader('Input Parameters')
            
            # Create a form for batch input
            with st.form("prediction_form"):
                # Get current date for temporal features
                current_date = datetime.now()
                
                # Input fields with more information
                product_type = st.selectbox('Product Type', 
                                          label_encoders['Product_Type'].classes_,
                                          help='Select the type of product to manufacture')
                
                batch_size = st.slider('Batch Size', 
                                     min_value=100, max_value=1000, value=500,
                                     help='Size of the production batch')
                
                raw_material_purity = st.slider('Raw Material Purity (%)', 
                                              min_value=90, max_value=100, value=95,
                                              help='Purity level of input materials')
                
                temperature = st.slider('Temperature Setting', 
                                      min_value=150, max_value=250, value=200,
                                      help='Process temperature setting')
                
                pressure = st.slider('Pressure Setting', 
                                   min_value=2, max_value=10, value=6,
                                   help='Process pressure setting')
                
                reactor_type = st.selectbox('Reactor Type', 
                                          label_encoders['Reactor_Type'].classes_,
                                          help='Type of reactor to use')
                
                equipment_age = st.slider('Equipment Age', 
                                        min_value=0, max_value=10, value=5,
                                        help='Age of equipment in years')
                
                maintenance_status = st.slider('Days Since Last Maintenance', 
                                             min_value=0, max_value=30, value=15,
                                             help='Days since last maintenance')
                
                operator_shift = st.selectbox('Operator Shift', 
                                            label_encoders['Operator_Shift'].classes_,
                                            help='Working shift')
                
                season = st.selectbox('Season', 
                                    label_encoders['Season'].classes_,
                                    help='Current season')
                
                quality_score = st.slider('Initial Quality Score', 
                                        min_value=70, max_value=100, value=85,
                                        help='Initial quality score')
                
                reactor_capacity = st.slider('Current Reactor Capacity (%)', 
                                           min_value=60, max_value=100, value=80,
                                           help='Current reactor capacity')
                
                # Submit button
                submitted = st.form_submit_button("Make Predictions")
        
        with col2:
            if submitted:
                st.subheader('Predictions')
                
                # Create input data with all features including temporal ones
                input_data = pd.DataFrame({
                    'Product_Type': [label_encoders['Product_Type'].transform([product_type])[0]],
                    'Batch_Size': [batch_size],
                    'Raw_Material_Purity': [raw_material_purity],
                    'Temperature_Setting': [temperature],
                    'Pressure_Setting': [pressure],
                    'Reactor_Type': [label_encoders['Reactor_Type'].transform([reactor_type])[0]],
                    'Equipment_Age': [equipment_age],
                    'Maintenance_Status': [maintenance_status],
                    'Operator_Shift': [label_encoders['Operator_Shift'].transform([operator_shift])[0]],
                    'Season': [label_encoders['Season'].transform([season])[0]],
                    'Initial_Quality_Score': [quality_score],
                    'Current_Reactor_Capacity': [reactor_capacity],
                    'Year': [current_date.year],
                    'Month': [current_date.month],
                    'DayOfWeek': [current_date.weekday()]
                })
                
                # Ensure column order matches training data
                input_data = input_data[X_scaled.columns]
                
                # Scale the input data
                input_scaled = scaler.transform(input_data)
                
                # Create columns for model predictions
                pred_cols = st.columns(len(models))
                
                # Make predictions with all models
                for idx, (name, model) in enumerate(models.items()):
                    with pred_cols[idx]:
                        pred = model.predict(input_scaled)[0]
                        
                        # Add confidence intervals if available
                        if hasattr(model, 'predict_proba'):
                            lower, upper = calculate_prediction_interval(model, input_scaled)
                            st.metric(
                                f"{name}",
                                f"{pred:.2f} hrs",
                                f"CI: [{lower:.2f}, {upper:.2f}]",
                                help=f"Prediction from {name} model"
                            )
                        else:
                            st.metric(
                                f"{name}",
                                f"{pred:.2f} hrs",
                                help=f"Prediction from {name} model"
                            )
                
                # Add a section for prediction analysis
                st.subheader("Prediction Analysis")
                
                # Create expander for detailed analysis
                with st.expander("View Detailed Analysis"):
                    # Show feature contributions
                    st.write("Feature Contributions:")
                    
                    # Use SHAP for feature contribution analysis
                    best_model = models['Random Forest']  # or select based on performance
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(input_scaled)
                    
                    # Create a DataFrame with feature contributions
                    contrib_df = pd.DataFrame({
                        'Feature': X_scaled.columns,
                        'Contribution': shap_values[0]
                    }).sort_values('Contribution', key=abs, ascending=False)
                    
                    # Plot feature contributions
                    fig = px.bar(contrib_df, 
                               x='Contribution', 
                               y='Feature',
                               orientation='h',
                               title='Feature Impact on Current Prediction')
                    st.plotly_chart(fig)
                    
                    # Add recommendations based on predictions
                    st.subheader("Optimization Recommendations")
                    
                    # Example recommendations based on feature contributions
                    recommendations = []
                    for idx, row in contrib_df.iterrows():
                        if abs(row['Contribution']) > 0.1:  # threshold for significant impact
                            if row['Contribution'] > 0:
                                recommendations.append(f"Consider reducing {row['Feature']} to decrease cycle time")
                            else:
                                recommendations.append(f"Consider increasing {row['Feature']} to decrease cycle time")
                    
                    for rec in recommendations[:3]:  # Show top 3 recommendations
                        st.info(rec)
    # Tab 2: Historical Analysis
    with tabs[1]:
        st.subheader('Historical Performance Analysis')
        
        # Time range selector with datetime conversion
        date_range = st.date_input('Select Date Range',
                                 [df['Date'].min().date(), df['Date'].max().date()],
                                 min_value=df['Date'].min().date(),
                                 max_value=df['Date'].max().date())
        
        # Convert date_range to pandas datetime
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])
        
        # Filter data based on date range
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        filtered_df = df[mask]
        
        # Display interactive visualizations
        st.plotly_chart(visualizations['trends'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(visualizations['distribution'])
        with col2:
            st.plotly_chart(visualizations['box_plot'])
    
    # Tab 3: Model Performance
    with tabs[2]:
        st.subheader('Model Performance Comparison')
        
        # Create detailed model performance comparison
        metrics_df = pd.DataFrame(model_results).T
        
        # Display metrics table
        st.dataframe(metrics_df.style.highlight_max(axis=0))
        
        # Display prediction vs actual scatter plot
        st.plotly_chart(visualizations['scatter'])
        
        # Model performance analysis
        col1, col2 = st.columns(2)
        with col1:
            # RMSE Comparison
            fig_rmse = px.bar(metrics_df, y=metrics_df.index, x='RMSE',
                            title='RMSE by Model',
                            orientation='h')
            st.plotly_chart(fig_rmse)
        
        with col2:
            # R² Comparison
            fig_r2 = px.bar(metrics_df, y=metrics_df.index, x='R2',
                           title='R² Score by Model',
                           orientation='h')
            st.plotly_chart(fig_r2)
    
    # Tab 4: Feature Analysis
    with tabs[3]:
        st.subheader('Feature Importance Analysis')
        
        # Display feature importance plot
        fig_importance = px.bar(feature_importance,
                              x='importance',
                              y='feature',
                              orientation='h',
                              title='Feature Importance')
        st.plotly_chart(fig_importance)
        
        # Add SHAP values analysis for the best model
        best_model = models['Random Forest']  # Or select based on performance
        st.subheader('SHAP Value Analysis')
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_scaled)
        
        # Plot SHAP summary
        st.pyplot(shap.summary_plot(shap_values, X_scaled))

# Helper function for prediction intervals
def calculate_prediction_interval(model, X, confidence=0.95):
    # This is a simplified version - you might want to implement
    # a more sophisticated method based on your specific needs
    predictions = []
    if hasattr(model, 'estimators_'):
        for estimator in model.estimators_:
            predictions.append(estimator.predict(X))
    
    predictions = np.array(predictions)
    lower = np.percentile(predictions, (1 - confidence) / 2 * 100)
    upper = np.percentile(predictions, (1 + confidence) / 2 * 100)
    
    return lower, upper

# Main execution
if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_data(1000)
    
    # Preprocess data
    X_scaled, y, label_encoders, scaler = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'LightGBM': LGBMRegressor(random_state=42)
    }
    
    # Train and evaluate models
    model_results, predictions = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # Get feature importance
    feature_importance = analyze_feature_importance(models['Random Forest'], X_scaled)
    
    # Create visualizations
    visualizations = create_historical_visualizations(df, predictions, y_test)
    
    # Create enhanced Streamlit UI
    create_enhanced_streamlit_ui(df, models, label_encoders, scaler, 
                               feature_importance, model_results, visualizations,X_scaled )
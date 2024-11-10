import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from datetime import datetime
import joblib
import json
import shap
import os

# Configure the app
st.set_page_config(layout="wide", page_title="Process Manufacturing Predictor")

# Cache data loading
@st.cache_resource
def load_models_and_artifacts():
    """Load all saved models and preprocessors with proper error handling"""
    required_files = {
        'models/scaler.pkl': 'Scaler',
        'models/label_encoders.pkl': 'Label Encoders',
        'models/linear_regression_model.pkl': 'Linear Regression Model',
        'models/random_forest_model.pkl': 'Random Forest Model',
        'models/xgboost_model.pkl': 'XGBoost Model',
        'models/lightgbm_model.pkl': 'LightGBM Model',
        'artifacts/feature_importance.pkl': 'Feature Importance',
        'artifacts/model_results.json': 'Model Results',
        'artifacts/sample_data.pkl': 'Sample Data',
        'artifacts/predictions.pkl': 'Predictions',
        'artifacts/test_data.pkl': 'Test Data',
        'artifacts/feature_names.json': 'Feature Names'
    }
    
    # Check if all required files exist
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"{description} ({file_path})")
    
    if missing_files:
        st.error("Missing required files:")
        for file in missing_files:
            st.error(f"- {file}")
        st.error("""
        Please run train_models.py first to generate all required artifacts.
        
        Command:
        ```
        python train_models.py
        ```
        """)
        st.stop()
    
    try:
        artifacts = {}
        
        # Load all artifacts with proper error handling
        artifacts['sample_data'] = pd.read_pickle('artifacts/sample_data.pkl')
        artifacts['scaler'] = joblib.load('models/scaler.pkl')
        artifacts['label_encoders'] = joblib.load('models/label_encoders.pkl')
        artifacts['feature_importance'] = pd.read_pickle('artifacts/feature_importance.pkl')
        artifacts['predictions'] = pd.read_pickle('artifacts/predictions.pkl')
        artifacts['test_data'] = joblib.load('artifacts/test_data.pkl')
        
        # Load model results
        with open('artifacts/model_results.json', 'r') as f:
            artifacts['model_results'] = json.load(f)
        
        # Load feature names
        with open('artifacts/feature_names.json', 'r') as f:
            artifacts['feature_names'] = json.load(f)
        
        # Load models
        artifacts['models'] = {
            'Linear Regression': joblib.load('models/linear_regression_model.pkl'),
            'Random Forest': joblib.load('models/random_forest_model.pkl'),
            'XGBoost': joblib.load('models/xgboost_model.pkl'),
            'LightGBM': joblib.load('models/lightgbm_model.pkl')
        }
        
        print("All artifacts loaded successfully!")
        return artifacts
        
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        st.error("""
        There was an error loading the required files. Please ensure:
        1. You've run train_models.py first
        2. All files were generated successfully
        3. You have sufficient permissions to read the files
        """)
        st.stop()
# Cache visualization creation
@st.cache_data
def create_cached_visualizations(_df, model_results):
    """Create and cache essential visualizations"""
    visuals = {}
    
    # 1. Cycle Time Trends
    visuals['trends'] = px.line(
        _df, 
        x='Date', 
        y='Historical_Cycle_Time',
        color='Product_Type',
        title='Historical Cycle Time Trends by Product Type'
    ).update_layout(
        template='plotly_white',
        xaxis_title='Date',
        yaxis_title='Cycle Time (hours)',
        legend_title='Product Type'
    )
    
    # 2. Cycle Time Distribution by Product Type
    visuals['distribution'] = px.histogram(
        _df,
        x='Historical_Cycle_Time',
        color='Product_Type',
        nbins=30,
        title='Cycle Time Distribution by Product Type'
    ).update_layout(
        template='plotly_white',
        xaxis_title='Cycle Time (hours)',
        yaxis_title='Count'
    )
    
    # 3. Box Plot
    visuals['box_plot'] = px.box(
        _df, 
        x='Product_Type', 
        y='Historical_Cycle_Time',
        color='Season',
        title='Cycle Time Distribution by Product Type and Season'
    ).update_layout(
        template='plotly_white',
        xaxis_title='Product Type',
        yaxis_title='Cycle Time (hours)'
    )
    
    # 4. Model Performance Comparison
    metrics_df = pd.DataFrame(model_results).T
    visuals['model_comparison'] = px.bar(
        metrics_df,
        barmode='group',
        title='Model Performance Metrics'
    ).update_layout(
        template='plotly_white',
        xaxis_title='Metric',
        yaxis_title='Value'
    )
    
    return visuals

def style_metrics_table(metrics_df):
    """Style the metrics dataframe with highlights"""
    return metrics_df.style.background_gradient(cmap='YlOrRd', axis=0)\
                          .highlight_max(axis=0, color='lightgreen')\
                          .highlight_min(axis=0, color='lightpink')\
                          .format(precision=4)

def create_prediction_analysis(input_data, input_scaled, model, feature_names):
    """Create prediction analysis visualizations"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    
    contrib_df = pd.DataFrame({
        'Feature': feature_names,
        'Value': input_data.iloc[0],
        'Impact': shap_values[0]
    }).sort_values('Impact', key=abs, ascending=False)
    
    return contrib_df

def create_prediction_ui(artifacts):
    """Create the prediction interface"""
    with st.form("prediction_form"):
        cols = st.columns(3)
        
        with cols[0]:
            product_type = st.selectbox(
                'Product Type', 
                artifacts['label_encoders']['Product_Type'].classes_,
                help='Select the type of product to manufacture'
            )
            
            batch_size = st.slider(
                'Batch Size', 
                min_value=100, 
                max_value=1000, 
                value=500,
                help='Size of the production batch'
            )
            
            raw_material_purity = st.slider(
                'Raw Material Purity (%)', 
                min_value=90, 
                max_value=100, 
                value=95
            )
            
        with cols[1]:
            temperature = st.slider('Temperature (°C)', 150, 250, 200)
            pressure = st.slider('Pressure (bar)', 2, 10, 6)
            reactor_type = st.selectbox(
                'Reactor Type', 
                artifacts['label_encoders']['Reactor_Type'].classes_
            )
            
        with cols[2]:
            equipment_age = st.slider('Equipment Age (years)', 0, 10, 5)
            maintenance_status = st.slider('Days Since Maintenance', 0, 30, 15)
            operator_shift = st.selectbox(
                'Operator Shift', 
                artifacts['label_encoders']['Operator_Shift'].classes_
            )
        
        season = st.selectbox('Season', artifacts['label_encoders']['Season'].classes_)
        quality_score = st.slider('Quality Score', 70, 100, 85)
        reactor_capacity = st.slider('Reactor Capacity (%)', 60, 100, 80)
        
        submitted = st.form_submit_button("Predict Cycle Time")
        
    return (submitted, locals())

def main():
    st.title('🏭 Process Manufacturing Cycle Time Prediction')
    
    # Load all artifacts
    with st.spinner('Loading models and data...'):
        artifacts = load_models_and_artifacts()
    
    # Create cached visualizations
    visualizations = create_cached_visualizations(
        artifacts['sample_data'],
        artifacts['model_results']
    )
    
    # Create tabs
    tabs = st.tabs(['Prediction', 'Historical Analysis', 'Model Performance', 'Feature Analysis'])
    
    # Prediction Tab
    with tabs[0]:
        submitted, inputs = create_prediction_ui(artifacts)
        
        if submitted:
            current_date = datetime.now()
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Product_Type': [artifacts['label_encoders']['Product_Type'].transform([inputs['product_type']])[0]],
                'Batch_Size': [inputs['batch_size']],
                'Raw_Material_Purity': [inputs['raw_material_purity']],
                'Temperature_Setting': [inputs['temperature']],
                'Pressure_Setting': [inputs['pressure']],
                'Reactor_Type': [artifacts['label_encoders']['Reactor_Type'].transform([inputs['reactor_type']])[0]],
                'Equipment_Age': [inputs['equipment_age']],
                'Maintenance_Status': [inputs['maintenance_status']],
                'Operator_Shift': [artifacts['label_encoders']['Operator_Shift'].transform([inputs['operator_shift']])[0]],
                'Season': [artifacts['label_encoders']['Season'].transform([inputs['season']])[0]],
                'Initial_Quality_Score': [inputs['quality_score']],
                'Current_Reactor_Capacity': [inputs['reactor_capacity']],
                'Year': [current_date.year],
                'Month': [current_date.month],
                'DayOfWeek': [current_date.weekday()]
            })
            
            # Scale input
            input_scaled = artifacts['scaler'].transform(input_data)
            
            # Make predictions
            st.subheader('Model Predictions')
            pred_cols = st.columns(len(artifacts['models']))
            
            for idx, (name, model) in enumerate(artifacts['models'].items()):
                with pred_cols[idx]:
                    pred = model.predict(input_scaled)[0]
                    st.metric(
                        f"{name}",
                        f"{pred:.2f} hrs",
                        help=f"Prediction from {name} model"
                    )
            
            # Prediction Analysis
            with st.expander("📊 Prediction Analysis", expanded=True):
                # Get feature contributions
                contrib_df = create_prediction_analysis(
                    input_data,
                    input_scaled,
                    artifacts['models']['Random Forest'],
                    artifacts['feature_names']
                )
                
                # Display feature impacts
                st.dataframe(
                    contrib_df.style.background_gradient(subset=['Impact'], cmap='RdYlBu')
                )
                
                # Show recommendations
                st.subheader("Optimization Recommendations")
                for _, row in contrib_df.head(3).iterrows():
                    if row['Impact'] > 0:
                        st.warning(f"⬇️ Consider reducing {row['Feature']} (current: {row['Value']:.2f})")
                    else:
                        st.success(f"⬆️ Consider increasing {row['Feature']} (current: {row['Value']:.2f})")
    
    # Historical Analysis Tab
    with tabs[1]:
        st.plotly_chart(visualizations['trends'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(visualizations['distribution'], use_container_width=True)
        with col2:
            st.plotly_chart(visualizations['box_plot'], use_container_width=True)
    
    # Model Performance Tab
    with tabs[2]:
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame(artifacts['model_results']).T
        st.dataframe(style_metrics_table(metrics_df))
        st.plotly_chart(visualizations['model_comparison'], use_container_width=True)
    
    # Feature Analysis Tab
    with tabs[3]:
        st.subheader("Feature Importance")
        st.plotly_chart(
            px.bar(
                artifacts['feature_importance'],
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance (Random Forest)'
            ),
            use_container_width=True
        )
        
        # SHAP Analysis
        with st.expander("🔍 SHAP Analysis", expanded=True):
            st.write("Global Feature Impact Analysis")
            best_model = artifacts['models']['Random Forest']
            explainer = shap.TreeExplainer(best_model)
            
            # Use sample of test data for SHAP analysis
            sample_data = artifacts['test_data']['X_test'].sample(n=100, random_state=42)
            shap_values = explainer.shap_values(sample_data)
            
            st.pyplot(shap.summary_plot(shap_values, sample_data))

if __name__ == "__main__":
    main()
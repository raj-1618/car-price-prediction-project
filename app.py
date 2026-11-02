import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Page Configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ---------------------------------------------------------
# 1. Self-Training Logic (Runs once & caches in memory)
# ---------------------------------------------------------
@st.cache_resource
def build_and_train_models():
    np.random.seed(42)
    n_samples = 1500

    brands = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Hyundai']
    engines = ['petrol', 'diesel', 'hybrid', 'electric']

    brand_data = np.random.choice(brands, n_samples)
    engine_data = np.random.choice(engines, n_samples)
    mileage_data = np.random.randint(5000, 250000, n_samples)
    age_data = np.random.randint(1, 20, n_samples)

    # Synthetic target generation
    base_price = 25000
    brand_mult = {'Toyota': 1.0, 'Honda': 0.95, 'Ford': 0.9, 'BMW': 1.6, 'Mercedes': 1.8, 'Hyundai': 0.8}
    engine_mult = {'petrol': 1.0, 'diesel': 1.05, 'hybrid': 1.2, 'electric': 1.3}

    prices = []
    for b, e, m, a in zip(brand_data, engine_data, mileage_data, age_data):
        p = base_price * brand_mult[b] * engine_mult[e]
        p -= (a * 900)
        p -= (m * 0.04)
        p += np.random.normal(0, 1500)
        prices.append(max(2000, round(p, 2)))

    df = pd.DataFrame({
        'brand': brand_data,
        'engine_type': engine_data,
        'mileage': mileage_data,
        'age': age_data,
        'price': prices
    })

    X = df[['brand', 'mileage', 'engine_type', 'age']]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['mileage', 'age']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand', 'engine_type'])
        ]
    )

    # Model Pipelines
    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    # Evaluation
    lr_preds = lr_pipeline.predict(X_test)
    rf_preds = rf_pipeline.predict(X_test)

    metrics = {
        'Linear Regression': {
            'MAE': mean_absolute_error(y_test, lr_preds),
            'RMSE': root_mean_squared_error(y_test, lr_preds)
        },
        'Random Forest': {
            'MAE': mean_absolute_error(y_test, rf_preds),
            'RMSE': root_mean_squared_error(y_test, rf_preds)
        }
    }

    return lr_pipeline, rf_pipeline, metrics

# Train models
lr_model, rf_model, metrics = build_and_train_models()

# ---------------------------------------------------------
# 2. UI & Header Setup
# ---------------------------------------------------------
st.title("🚗 Used Car Price Prediction")
st.markdown("Predict fair resale values using real-time Machine Learning models.")

# Sidebar - Project Team
st.sidebar.title("👥 Project Team")
st.sidebar.info("""
**Team Members:**
- Raj Kashyap
- Kushagri Srivastava
- Abhinav Raj
- Deepali Choudhary
""")

# Main Tabs
tab1, tab2 = st.tabs(["💡 Price Estimator", "📊 Model Performance"])

with tab1:
    st.subheader("Vehicle Details")
    
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Hyundai'])
        engine_type = st.selectbox("Engine Type", ['petrol', 'diesel', 'hybrid', 'electric'])
    with col2:
        mileage = st.number_input("Mileage (in km)", min_value=0, max_value=500000, value=45000, step=5000)
        age = st.number_input("Age of Vehicle (in years)", min_value=0, max_value=30, value=4, step=1)

    model_choice = st.radio("Select Machine Learning Model", ['Random Forest Regressor', 'Linear Regression'], horizontal=True)

    if st.button("Predict Resale Price", type="primary"):
        input_data = pd.DataFrame({
            'brand': [brand],
            'mileage': [mileage],
            'engine_type': [engine_type],
            'age': [age]
        })

        if model_choice == 'Random Forest Regressor':
            prediction = rf_model.predict(input_data)[0]
        else:
            prediction = lr_model.predict(input_data)[0]

        st.metric(label=f"Estimated Price ({model_choice})", value=f"${prediction:,.2f}")

with tab2:
    st.subheader("Algorithm Evaluation & Metrics")
    
    results_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest Regressor'],
        'MAE ($)': [metrics['Linear Regression']['MAE'], metrics['Random Forest']['MAE']],
        'RMSE ($)': [metrics['Linear Regression']['RMSE'], metrics['Random Forest']['RMSE']]
    })
    
    st.table(results_df)
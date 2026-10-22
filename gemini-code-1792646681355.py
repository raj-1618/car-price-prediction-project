import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor | AI Applications",
    page_icon="🚗",
    layout="wide"
)

# -------------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists("used_cars.csv"):
        return pd.read_csv("used_cars.csv")
    elif os.path.exists("used cars.csv"):
        return pd.read_csv("used cars.csv")
    else:
        st.error("Dataset file not found! Please ensure 'used_cars.csv' is in the repository.")
        st.stop()

# -------------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("**Course:** AI Applications")
st.sidebar.markdown("**Problem Type:** Regression")
st.sidebar.markdown("**Dataset:** Kaggle – Used Car Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Custom CSV (Optional)", 
    type=["csv"],
    help="Columns required: Brand, Mileage, Engine_Type, Age, Price"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom CSV loaded!")
else:
    df = load_data()

# -------------------------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------------------------
@st.cache_resource
def train_models(data):
    X = data[['Brand', 'Mileage', 'Engine_Type', 'Age']]
    y = data['Price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = ['Brand', 'Engine_Type']
    numerical_features = ['Mileage', 'Age']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    lr_pipeline.fit(X_train, y_train)
    rf_pipeline.fit(X_train, y_train)

    lr_preds = lr_pipeline.predict(X_test)
    rf_preds = rf_pipeline.predict(X_test)

    metrics = {
        'Linear Regression': {
            'MAE': mean_absolute_error(y_test, lr_preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, lr_preds)),
            'R2': r2_score(y_test, lr_preds),
            'preds': lr_preds
        },
        'Random Forest Regressor': {
            'MAE': mean_absolute_error(y_test, rf_preds),
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_preds)),
            'R2': r2_score(y_test, rf_preds),
            'preds': rf_preds
        }
    }

    return lr_pipeline, rf_pipeline, metrics, y_test

lr_model, rf_model, metrics, y_test = train_models(df)

# -------------------------------------------------------------------------
# MAIN UI
# -------------------------------------------------------------------------
st.title("🚗 Used Car Price Prediction System")
st.markdown(
    """
    **Objective:** Predict the resale price of used cars using **Linear Regression** and **Random Forest Regressor** 
    to help buyers and sellers estimate fair market prices.
    """
)

tab1, tab2, tab3 = st.tabs([
    "💰 Price Predictor (Buyers & Sellers)", 
    "📊 Model Evaluation (MAE & RMSE)", 
    "🗂️ Dataset Explorer"
])

with tab1:
    st.subheader("Enter Vehicle Specifications")
    col1, col2 = st.columns(2)

    with col1:
        brand_input = st.selectbox("Select Car Brand", options=sorted(df['Brand'].unique()))
        engine_input = st.selectbox("Select Engine Type", options=sorted(df['Engine_Type'].unique()))

    with col2:
        age_input = st.slider("Age of Vehicle (Years)", min_value=1, max_value=15, value=4, step=1)
        mileage_input = st.number_input("Mileage / Distance Driven (km)", min_value=1000, max_value=250000, value=40000, step=1000)

    if st.button("🔍 Predict Resale Price", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            'Brand': brand_input,
            'Mileage': mileage_input,
            'Engine_Type': engine_input,
            'Age': age_input
        }])

        lr_pred = max(50000, lr_model.predict(input_df)[0])
        rf_pred = max(50000, rf_model.predict(input_df)[0])

        st.markdown("---")
        st.subheader("🎯 Estimated Resale Value")

        c1, c2, c3 = st.columns(3)
        c1.metric("Linear Regression Estimate", f"₹ {lr_pred:,.0f}")
        c2.metric("Random Forest Estimate", f"₹ {rf_pred:,.0f}")
        c3.metric("Recommended Fair Price Range", f"₹ {rf_pred*0.95:,.0f} - ₹ {rf_pred*1.05:,.0f}")

        st.info(
            f"💡 **Market Insight:** For a **{age_input}-year-old {brand_input} ({engine_input})** driven **{mileage_input:,} km**, "
            f"a fair deal for both buyer and seller is around **₹ {rf_pred:,.0f}**."
        )

with tab2:
    st.subheader("Model Performance Metrics (MAE & RMSE)")
    eval_df = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest Regressor'],
        'MAE (Mean Absolute Error)': [
            f"₹ {metrics['Linear Regression']['MAE']:,.2f}",
            f"₹ {metrics['Random Forest Regressor']['MAE']:,.2f}"
        ],
        'RMSE (Root Mean Squared Error)': [
            f"₹ {metrics['Linear Regression']['RMSE']:,.2f}",
            f"₹ {metrics['Random Forest Regressor']['RMSE']:,.2f}"
        ],
        'R² Score': [
            f"{metrics['Linear Regression']['R2']:.4f}",
            f"{metrics['Random Forest Regressor']['R2']:.4f}"
        ]
    })
    st.table(eval_df)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### MAE & RMSE Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        models = ['Linear Regression', 'Random Forest']
        mae_vals = [metrics['Linear Regression']['MAE'], metrics['Random Forest Regressor']['MAE']]
        rmse_vals = [metrics['Linear Regression']['RMSE'], metrics['Random Forest Regressor']['RMSE']]
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width/2, mae_vals, width, label='MAE', color='#3b82f6')
        ax.bar(x + width/2, rmse_vals, width, label='RMSE', color='#ef4444')
        ax.set_ylabel('Error (₹)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)

    with col_b:
        st.markdown("#### Actual vs. Predicted Prices")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(y_test, metrics['Linear Regression']['preds'], alpha=0.6, label='Linear Regression', color='#f59e0b')
        ax2.scatter(y_test, metrics['Random Forest Regressor']['preds'], alpha=0.7, label='Random Forest', color='#10b981')
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
        ax2.set_xlabel('Actual Price (₹)')
        ax2.set_ylabel('Predicted Price (₹)')
        ax2.legend()
        ax2.grid(linestyle='--', alpha=0.5)
        st.pyplot(fig2)

with tab3:
    st.subheader("Dataset Overview")
    st.write(f"Total Rows: **{len(df)}**")
    st.dataframe(df, use_container_width=True)
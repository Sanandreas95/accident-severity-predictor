import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Accident Severity Risk Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS FOR BETTER STYLING
# =====================================================

st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    h1 {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 10px;
    }
    .subheader {
        text-align: center;
        color: #666;
        font-size: 14px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL AND SCALER
# =====================================================

@st.cache_resource
def load_model_and_scaler():
    """Load pre-trained XGBoost model and StandardScaler"""
    try:
        # Try loading with joblib
        model = joblib.load('xgb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure xgb_model.pkl, scaler.pkl, and feature_columns.pkl are in the app directory.")
        st.stop()

model, scaler, feature_columns = load_model_and_scaler()

# =====================================================
# TITLE AND DESCRIPTION
# =====================================================

st.markdown("<h1>🚗 Accident Severity Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div class="subheader">
    Predict the risk of severe accidents based on time, weather, and road conditions.
    <br/>Built on 7.7M historical accident records | XGBoost ML Model
</div>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# INPUT SECTION
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🕐 Time")
    hour = st.slider(
        "Hour of Day",
        min_value=0, max_value=23,
        value=14,
        help="Select the hour (0-23). Rush hours: 7-9 AM, 4-6 PM"
    )
    
    day_type = st.selectbox(
        "Day Type",
        options=["Weekday", "Weekend"],
        help="Weekdays show slightly higher severity rates"
    )
    is_weekend = 1 if day_type == "Weekend" else 0
    
    is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0

with col2:
    st.subheader("🌤️ Weather")
    weather_condition = st.selectbox(
        "Weather Condition",
        options=["Clear", "Cloudy", "Rain", "Snow", "Fog", "Wind", "Other", "Unknown"],
        help="Snow/Rain are riskier. Clear is baseline."
    )
    
    temperature = st.slider(
        "Temperature (°F)",
        min_value=-20, max_value=120,
        value=70,
        help="Extreme cold (<32°F) increases severity"
    )
    
    visibility = st.slider(
        "Visibility (miles)",
        min_value=0.1, max_value=10.0,
        value=10.0,
        step=0.5,
        help="Low visibility (fog) increases risk. 10 = excellent."
    )

with col3:
    st.subheader("🛣️ Road Features")
    has_junction = st.checkbox("Junction", value=False, help="Intersection")
    has_traffic_signal = st.checkbox("Traffic Signal", value=False)
    has_crossing = st.checkbox("Crossing", value=False)
    has_bump = st.checkbox("Bump/Obstacle", value=False)

st.divider()

# =====================================================
# PREPARE FEATURES FOR PREDICTION
# =====================================================

# Create feature dictionary with default values
feature_dict = {}

# Time features
feature_dict['hour'] = hour
feature_dict['day_of_week'] = 5 if is_weekend else 2  # Saturday if weekend, Wed if weekday (midpoints)
feature_dict['month'] = 6  # Default to June (mid-year)
feature_dict['year'] = 2023  # Latest year
feature_dict['is_weekend'] = is_weekend
feature_dict['is_rush_hour'] = is_rush_hour
feature_dict['season_encoded'] = 2  # Summer (default)

# Weather numeric features
feature_dict['Temperature(F)'] = temperature
feature_dict['Humidity(%)'] = 50  # Default
feature_dict['Visibility(mi)'] = visibility
feature_dict['Wind_Speed(mph)'] = 10  # Default
feature_dict['Precipitation(in)'] = 0  # Default (no rain)
feature_dict['Pressure(in)'] = 29.9  # Default

# # Weather categorical features (one-hot encoded)
# weather_options = ["Clear", "Cloudy", "Fog", "Rain", "Snow", "Wind", "Other",
#     "Unknown"]
# for weather in weather_options:
#     feature_dict[f'weather_{weather}'] = 1 if weather == weather_condition else 0



# Weather categorical features (one-hot encoded)

# Initialize all weather columns expected by the model
for col in feature_columns:
    if col.startswith('weather_'):
        feature_dict[col] = 0

# Set the selected weather condition to 1
selected_weather = f'weather_{weather_condition}'

if selected_weather in feature_dict:
    feature_dict[selected_weather] = 1




# Time-of-day features
feature_dict['is_night'] = 1 if hour >= 20 or hour <= 6 else 0
feature_dict['is_civil_twilight'] = 1 if hour in [6, 7, 19, 20] else 0

# Road infrastructure features
feature_dict['Junction'] = 1 if has_junction else 0
feature_dict['Traffic_Signal'] = 1 if has_traffic_signal else 0
feature_dict['Crossing'] = 1 if has_crossing else 0
feature_dict['Amenity'] = 0  # Default
feature_dict['Bump'] = 1 if has_bump else 0
feature_dict['Give_Way'] = 0
feature_dict['No_Exit'] = 0
feature_dict['Railway'] = 0
feature_dict['Roundabout'] = 0
feature_dict['Station'] = 0
feature_dict['Stop'] = 0
feature_dict['Traffic_Calming'] = 0
feature_dict['Turning_Loop'] = 0

# Create DataFrame with features in correct order



# try:
#     X = pd.DataFrame([feature_dict])[feature_columns]
# except KeyError as e:
#     st.error(f"Missing feature column: {e}")
#     st.stop()





# Create DataFrame (later correction)
# =====================================================
# CREATE MODEL INPUT
# =====================================================

X = pd.DataFrame([feature_dict])

# Add missing columns
for col in feature_columns:
    if col not in X.columns:
        X[col] = 0

# Keep exact feature order
X = X.reindex(columns=feature_columns, fill_value=0)

# Convert all values to numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)






st.write("Feature count expected:", len(feature_columns))
st.write("Feature count supplied:", len(X.columns))


# =====================================================
# MAKE PREDICTION
# =====================================================

# =====================================================
# MAKE PREDICTION
# =====================================================

try:
    # Ensure numeric types
    X = X.astype(float)

    # Predict directly using XGBoost
    prediction_proba = model.predict_proba(X)[0]
    severity_prob = float(prediction_proba[1])

    prediction = model.predict(X)[0]
    severity_label = "SEVERE" if prediction == 1 else "NON-SEVERE"

except Exception as e:
    st.error(f"Prediction Error: {e}")

    st.write("Expected Features:", len(feature_columns))
    st.write("Input Shape:", X.shape)

    st.write("Missing Features:")
    st.write([c for c in feature_columns if c not in X.columns])

    st.stop()

# =====================================================
# DISPLAY RESULTS
# =====================================================

st.markdown("## 📊 Prediction Results")

col1, col2, col3 = st.columns(3)

with col1:
    # Risk probability meter
    if severity_prob >= 0.06:
        risk_color = "#f5576c"  # Red
        risk_text = "🔴 HIGH RISK"
    elif severity_prob >= 0.04:
        risk_color = "#fa709a"  # Orange
        risk_text = "🟠 MEDIUM RISK"
    else:
        risk_color = "#30cfd0"  # Green
        risk_text = "🟢 LOW RISK"
    
    st.markdown(f"""
    <div style="background: {risk_color}; color: white; padding: 20px; border-radius: 10px; text-align: center;">
        <h3 style="margin: 0; color: white;">{risk_text}</h3>
        <p style="margin: 10px 0 0 0; font-size: 24px; font-weight: bold;">{severity_prob*100:.2f}%</p>
        <p style="margin: 5px 0 0 0; font-size: 12px;">Probability of Severe Accident</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Prediction class
    st.metric(
        label="Predicted Class",
        value=severity_label,
        delta="High Alert" if severity_prob >= 0.06 else "Normal"
    )

with col3:
    # Comparison to baseline
    baseline_rate = 0.03  # 3% severe rate from training data
    relative_risk = severity_prob / baseline_rate
    st.metric(
        label="Relative Risk",
        value=f"{relative_risk:.1f}x",
        delta="vs. baseline" if relative_risk > 1 else "below baseline"
    )

st.divider()

# =====================================================
# VISUAL GAUGE
# =====================================================

# Create gauge chart
fig = go.Figure(data=[
    go.Indicator(
        mode="gauge+number+delta",
        value=severity_prob * 100,
        title={'text': "Severity Risk Score (%)"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 15]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 6], 'color': "lightyellow"},
                {'range': [6, 15], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 6
            }
        },
        delta={'reference': baseline_rate * 100}
    )
])

fig.update_layout(
    height=400,
    font={'size': 12},
    margin={'l': 50, 'r': 50, 't': 100, 'b': 50}
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# =====================================================
# INSIGHTS AND RECOMMENDATIONS
# =====================================================

st.markdown("## 💡 Key Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Risk Factors Present:")
    
    risk_factors = []
    
    if is_rush_hour:
        risk_factors.append("⚠️ **Rush Hour** - 20-30% higher severity")
    
    if weather_condition in ["Rain", "Snow"]:
        risk_factors.append(f"⚠️ **{weather_condition} Weather** - 2-3x higher severity")
    
    if visibility < 2:
        risk_factors.append("⚠️ **Low Visibility** - Poor driving conditions")
    
    if temperature < 32:
        risk_factors.append("⚠️ **Extreme Cold** - Icy roads, poor grip")
    
    if has_junction or has_traffic_signal:
        risk_factors.append("⚠️ **Complex Intersection** - Higher accident concentration")
    
    if not risk_factors:
        risk_factors.append("✅ No major risk factors detected")
    
    for factor in risk_factors:
        st.markdown(factor)

with col2:
    st.markdown("### Recommendations:")
    
    recommendations = []
    
    if severity_prob >= 0.06:
        recommendations.append("🚨 **AVOID or DELAY trip** if possible")
        recommendations.append("⏱️ Allow extra time for careful driving")
        recommendations.append("🔍 Increase vigilance and reduce speed")
    elif severity_prob >= 0.04:
        recommendations.append("⚠️ **Use caution** - higher risk conditions")
        recommendations.append("📍 Avoid complex intersections")
        recommendations.append("💧 Check vehicle maintenance (tires, brakes)")
    else:
        recommendations.append("✅ Safe conditions - normal precautions apply")
        recommendations.append("🚗 Maintain standard speed limits")
        recommendations.append("👀 Standard defensive driving practices")
    
    for rec in recommendations:
        st.markdown(rec)

st.divider()

# =====================================================
# FEATURE BREAKDOWN TABLE
# =====================================================

st.markdown("### 📈 Feature Values Entered:")

feature_display = pd.DataFrame({
    'Category': ['Time', 'Time', 'Time', 'Weather', 'Weather', 'Weather', 'Road Features', 'Road Features', 'Road Features', 'Road Features'],
    'Feature': ['Hour', 'Day Type', 'Rush Hour', 'Condition', 'Temperature', 'Visibility', 'Junction', 'Traffic Signal', 'Crossing', 'Bump'],
    'Value': [
        f"{hour}:00",
        day_type,
        "Yes" if is_rush_hour else "No",
        weather_condition,
        f"{temperature}°F",
        f"{visibility} mi",
        "Yes" if has_junction else "No",
        "Yes" if has_traffic_signal else "No",
        "Yes" if has_crossing else "No",
        "Yes" if has_bump else "No"
    ]
})

st.dataframe(feature_display, use_container_width=True, hide_index=True)

# =====================================================
# FOOTER
# =====================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📚 Model Details:**
    - Algorithm: XGBoost
    - Training Data: 595K accidents
    - Features: 36
    - Accuracy: 78% precision on severe
    """)

with col2:
    st.markdown("""
    **⚠️ Disclaimer:**
    This is a predictive model based on historical data. 
    It should not replace professional judgment or traffic laws.
    Always follow safety guidelines.
    """)

with col3:
    st.markdown("""
    **🔗 Resources:**
    - [GitHub Repo](#)
    - [Full Analysis](#)
    - [Deploy Guide](#)
    """)

st.markdown("""
---
*Built with Streamlit | Data: 7.7M US Accidents (2016-2023) | Last Updated: 2024*
""")

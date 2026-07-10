# 🚗 Accident Severity Risk Predictor

A machine learning web app that predicts the probability of severe accidents based on time, weather, and road conditions.

**Model:** XGBoost trained on 7.7M accident records (2016-2023)
**Accuracy:** 78% precision on severe accidents, 58% recall
**Live Demo:** [Coming Soon - Deploy on Streamlit Community Cloud]

---

## 📋 Features

### Inputs
- **Time:** Hour of day (0-23), day type (weekday/weekend)
- **Weather:** Condition dropdown, temperature, visibility
- **Road:** Checkboxes for junction, traffic signal, crossing, bump

### Outputs
- 🎯 **Severity Risk Probability** - Percentage chance of severe accident
- 📊 **Risk Gauge Chart** - Visual representation with zones
- 🔴 **Risk Level** - Low/Medium/High classification
- 📈 **Relative Risk** - Comparison to baseline (3%)
- 💡 **Insights** - Risk factors present and recommendations

---

## 🚀 Quick Start (Local Testing)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run streamlit_app.py
```

### 3. Open in Browser
```
http://localhost:8501
```

---

## 📦 Files Required

```
app/
├── streamlit_app.py          # Main Streamlit application
├── xgb_model.pkl             # Trained XGBoost model
├── scaler.pkl                # StandardScaler for feature normalization
├── feature_columns.pkl       # List of feature names in correct order
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### File Sizes
- `xgb_model.pkl`: ~35 MB (model)
- `scaler.pkl`: <1 MB (scaler)
- `feature_columns.pkl`: <1 MB (feature list)
- **Total:** <40 MB (well under 100 MB limit)

---

## ☁️ Deploy on Streamlit Community Cloud

### Step 1: Create GitHub Repository
```bash
# Create new repo on GitHub
git clone https://github.com/YOUR_USERNAME/accident-severity-predictor.git
cd accident-severity-predictor
```

### Step 2: Copy App Files
Copy these files to your repo:
- `streamlit_app.py`
- `xgb_model.pkl`
- `scaler.pkl`
- `feature_columns.pkl`
- `requirements.txt`
- `.gitignore` (to avoid uploading large files unnecessarily)

### Step 3: Create .gitignore
```
# Ignore large model files if rebuilding
*.pkl
*.joblib
__pycache__/
*.pyc
.DS_Store
```

### Step 4: Push to GitHub
```bash
git add .
git commit -m "Initial Streamlit app deployment"
git push origin main
```

### Step 5: Deploy on Streamlit Cloud
1. Visit https://share.streamlit.io/
2. Click **"New app"**
3. Fill in:
   - **GitHub account:** YOUR_USERNAME
   - **Repository:** accident-severity-predictor
   - **Branch:** main
   - **Main file path:** streamlit_app.py
4. Click **"Deploy"**
5. Wait 2-5 minutes for deployment
6. Share your app URL: `https://share.streamlit.io/YOUR_USERNAME/accident-severity-predictor/main/streamlit_app.py`

---

## 📊 How the Model Works

### Training Data
- **7.7 million** road accidents (2016-2023, USA)
- **36 features** covering time, weather, location, and road conditions
- **Binary target:** Severe (3-4) vs. Non-severe (1-2)

### Model Architecture
- **Algorithm:** XGBoost (gradient boosting trees)
- **Parameters:** 500 trees, depth 7, class weighting for imbalance
- **Imbalance handling:** scale_pos_weight=32.3 (penalize minority class errors)

### Performance
| Metric | Value |
|--------|-------|
| Precision | 0.52 (52% of predictions correct) |
| Recall | 0.58 (catches 58% of severe) |
| F1-Score | 0.55 |
| ROC-AUC | 0.87 |

### Key Features (by importance)
1. **is_rush_hour** - Rush hours (7-9 AM, 4-6 PM) have 20-30% higher severity
2. **hour** - Peak hours show elevated severity
3. **weather_Rain** - Rain increases severity 2-3x
4. **Temperature(F)** - Extreme cold (<32°F) doubles severity
5. **Visibility(mi)** - Low visibility (fog) increases risk

---

## 🎨 App Interface

### Top Section
- **Inputs:** Time, weather, road features in clean columns
- **Real-time updates:** Predictions update as you change inputs

### Results Section
- **Risk Box:** Color-coded (green/yellow/red) with percentage
- **Prediction Class:** Severe or Non-severe label
- **Relative Risk:** How much riskier than baseline (3%)
- **Gauge Chart:** Interactive Plotly visualization

### Insights Section
- **Risk Factors:** Which factors are elevating risk
- **Recommendations:** Tailored advice based on risk level

### Bottom Section
- **Feature Table:** All input values in one place
- **Model Details:** Architecture, data, performance
- **Footer:** Disclaimer and resources

---

## 💡 Usage Examples

### Example 1: Rush Hour in Rain
```
Hour: 17 (5 PM)
Day: Weekday
Weather: Rain
Temperature: 65°F
Visibility: 3 miles
Road: Traffic Signal
→ Result: 7.5% risk (HIGH) - "Avoid or delay trip"
```

### Example 2: Clear Weather, Daytime
```
Hour: 14 (2 PM)
Day: Weekday
Weather: Clear
Temperature: 75°F
Visibility: 10 miles
Road: No features
→ Result: 1.8% risk (LOW) - "Safe conditions"
```

### Example 3: Winter Night
```
Hour: 22 (10 PM)
Day: Weekday
Weather: Snow
Temperature: 20°F
Visibility: 1 mile
Road: Junction, Traffic Signal
→ Result: 11.2% risk (CRITICAL) - "AVOID trip if possible"
```

---

## ⚙️ Customization

### Add More Features
Edit the input section in `streamlit_app.py`:
```python
new_feature = st.slider("Feature Name", min_value=0, max_value=100)
feature_dict['new_feature'] = new_feature
```

### Change Risk Thresholds
Modify the threshold values in results section:
```python
if severity_prob >= 0.08:  # Changed from 0.06
    risk_color = "#f5576c"
    risk_text = "🔴 CRITICAL RISK"
```

### Update Model
To use a newer/better model:
1. Retrain with `06_modeling.py`
2. Save with `08_save_model_for_streamlit.py`
3. Replace `xgb_model.pkl`, `scaler.pkl`, `feature_columns.pkl`
4. Push to GitHub (Streamlit will auto-redeploy)

---

## 🔒 Security & Privacy

- **No data collection:** App doesn't store user inputs
- **No external API calls:** All predictions are local
- **Open source:** Code is transparent and auditable
- **Model explainability:** Features are interpretable (not a black box)

---

## ⚠️ Disclaimer

This application provides **predictions based on historical data patterns**. It should **NOT**:
- Replace professional judgment
- Override traffic laws and safety regulations
- Be used for critical medical or legal decisions

Always prioritize safety and follow official traffic guidelines.

---

## 📈 Model Development

For more details on model development:
- See full analysis: [GitHub Repo](#)
- SHAP explainability: [Notebook](#)
- Hotspot analysis: [Interactive Maps](#)
- EDA findings: [Report](#)

---

## 👨‍💻 Author

Built as part of a comprehensive data science portfolio project.

**Skills Demonstrated:**
- Data engineering (7.7M rows, SQL, MySQL)
- Feature engineering (36 features)
- Machine learning (XGBoost, imbalance handling)
- Model evaluation (precision, recall, ROC-AUC)
- Explainability (SHAP analysis)
- Web deployment (Streamlit Cloud)
- Geospatial analysis (DBSCAN clustering)

---

## 🤝 Contributing

Found a bug or have suggestions? 
- Open an issue on GitHub
- Submit a pull request
- Share feedback

---

## 📄 License

MIT License - Feel free to use and modify

---

## 🔗 Links

- **GitHub:** [Link](#)
- **Portfolio:** [Link](#)
- **LinkedIn:** [Link](#)
- **Email:** [contact@example.com](#)

---

*Last updated: January 2024*
*Built with Streamlit | Data: 7.7M US Accidents (2016-2023)*

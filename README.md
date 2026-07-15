# 🚗 Road Accident Severity & Hotspot Analyzer

System to predict accident severity and identify high-risk geographic hotspots*

---

## 📋 Table of Contents

1. [Project Title & Overview](#-project-title--overview)
2. [Problem Statement](#-problem-statement)
3. [Dataset & Data Source](#-dataset--data-source)
4. [Tools & Technologies](#-tools--technologies)
5. [Project Workflow](#-project-workflow)
6. [AI/ML Components](#-aiml-components)
7. [How to Run the Project](#-how-to-run-the-project)
8. [Demo & Screenshots](#-demo--screenshots)
9. [Results & Insights](#-results--insights)
10. [Limitations](#-limitations)
11. [Future Improvements](#-future-improvements)
12. [Team Members](#-team-members)

---

## 🎯 Project Title & Overview

### **Road Accident Severity & Hotspot Analyzer**

**A comprehensive end-to-end data science pipeline that leverages machine learning and geospatial analysis to:**

- ✅ **Predict accident severity** (severe vs. non-severe) based on time, weather, and road conditions
- ✅ **Identify geographic hotspots** where accidents are concentrated
- ✅ **Provide real-time risk assessments** through an interactive web application
- ✅ **Enable data-driven decision-making** for insurance, city planning, and traffic management

**Business Impact:**
-  **Insurance Industry:** Better risk pricing and fraud detection
- **City Planning:** Targeted safety interventions and resource allocation
-  **Safety Agencies:** Evidence-based traffic enforcement and education

---

## ⚠️ Problem Statement

### **The Challenge**

Road traffic accidents are a leading cause of injury and death, with billions of dollars spent annually on medical care, lost productivity, and property damage. Current approaches to accident prevention are largely reactive—responding after accidents occur rather than predicting and preventing them.

### **Key Questions We Address**

1. **Can we predict accident severity** before it happens using available contextual data?
2. **Which combinations of conditions** (time, weather, location) lead to severe accidents?
3. **Where are the geographic hotspots** where severe accidents are most likely to occur?
4. **How can we quantify and communicate risk** to stakeholders (drivers, insurers, city planners)?

### **Business Objectives**

- Reduce accident severity through predictive interventions
- Optimize insurance pricing models
- Guide traffic management and infrastructure improvements
- Enable data-driven public safety initiatives


## 📊 Dataset & Data Source

### **Data Overview**

| Metric | Value |
|--------|-------|
| **Total Records** | 1744915 Million |
| **Time Period** | 2016 - 2023 |
| **Geographic Focus** | United States (Primary: California) |
| **Features** | 43 engineered features |
| **Target Variable** | Accident Severity (Binary: Severe/Non-Severe) |
| **Class Distribution** | 96.7% Non-Severe, 3.3% Severe |

### **Data Source**

📍 **US Accident Dataset** - "A Countrywide Traffic Accident Dataset"
- **Available At:** [Kaggle - US Accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents?select=US_Accidents_March23.csv)


### **Data Features**

**Temporal Features (7):** Hour, Day of week, Month, Year, Weekend flag, Rush hour, Season

**Weather Features (14):** Temperature, Humidity, Visibility, Wind speed, Precipitation, Pressure, + 8 categorical conditions

**Location Features (15):** Latitude, Longitude, City, County, State, Zip, + road infrastructure (junctions, signals, crossings, etc.)

**Interaction Features (7):** Rain during rush, Night + bad weather, Cold + snow, Weather severity, Signal interactions, Friday night

---

## 🛠️ Tools & Technologies

### **Core Stack**

| Category | Tools |
|----------|-------|
| **Language** | Python |
| **Database** | MySQL |
| **ML Framework** | XGBoost, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Folium |
| **Web App** | Streamlit |
| **Explainability** | SHAP |
| **Geospatial** | DBSCAN, GeoPy |
| **Deployment** | Joblib, Docker-ready |

---

## 🔄 Project Workflow

### **5 Main Phases**

**Phase 1: Data Ingestion & Exploration**
- Load 1744915 records from Kaggle/MySQL
- SQL exploration and profiling
- Identify patterns and anomalies

**Phase 2: Data Cleaning & Engineering**
- Handle missing values (0 after treatment)
- Create 43 engineered features
- Stratified 80/20 train/test split
- EDA 

**Phase 3: Model Development**
- Train XGBoost and Logistic Regression
- Optimize hyperparameters
- Select balanced threshold (0.65)

**Phase 4: Geographic Analysis**
- DBSCAN clustering for hotspots (28 identified)
- Interactive Folium maps
- Risk intensity visualization

**Phase 5: Deployment**
- Streamlit web application
- Real-time predictions
- Model performance dashboard

---

## 🤖 AI/ML Components

### **XGBoost Classifier (Primary Model)**

- **Algorithm:** Gradient Boosting on 500 decision trees
- **Max Depth:** 7
- **Learning Rate:** 0.1
- **Class Weight:** 29.35 (handles 3.3% minority class)

### **Performance Metrics**

| Metric | Value |
|--------|-------|
| **Accuracy** | 79.99% |
| **Precision** | 43.01% |
| **Recall** | 66.79% |
| **F1-Score** | 52.33% |
| **ROC-AUC** | 83.48% |

### **Feature Importance (Top 5)**

1. Weather_Rain (12.3%)
2. Hour (10.5%)
3. Temperature (9.8%)
4. Is_rush_hour (8.2%)
5. Visibility (7.6%)

### **Hotspot Detection**

- **Algorithm:** DBSCAN Clustering
- **Eps:** 0.5 km
- **Min Samples:** 5
- **Result:** 28 geographic hotspots identified

---

## 🚀 How to Run the Project

### **Quick Start (5 steps)**

```bash
# 1. Clone repository
cd "Capstone 2"

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run pipeline
jupyter notebook
# Execute: 03_cleaning.py → 06_modeling.py → 08_save_model.py

# 5. Launch web app
cd accident-severity-predictor
streamlit run streamlit_app.py
```

**App opens at:** http://localhost:8507

### **Complete Directory Structure**

```
Capstone 2/
├── data/                              (Processed datasets)
├── notebooks/                         (9 Jupyter notebooks)
├── results/                           (24+ visualizations)
└── accident-severity-predictor/       (Streamlit app + model files)
```

---





### **Interactive Web App**

```
Accident Severity Risk Predictor
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIDEBAR METRICS:
Accuracy: 79.99%
Precision: 43.01%
Recall: 66.79%
ROC-AUC: 83.48%

INPUT CONTROLS:
Hour: [0-23 slider]
Weather: [Dropdown - Clear/Rain/Snow]
Temperature: [−20 to 120°F]
Road Features: [Checkboxes]

OUTPUT DISPLAY:
🟠 MEDIUM RISK
Probability: 5.2%
Prediction: NON-SEVERE ✅
Relative Risk: 1.7x

Risk Gauge Chart:
[Visual indicator showing probability]

Insights & Recommendations:
✓ Risk Factors
💡 Safety Tips
```

### **Hotspot Maps**

- Interactive Folium heatmaps
- 28 identified clusters
- Risk intensity overlays
- KPI cards for each hotspot

---

## 📊 Results & Insights

### **Key Findings**

**Temporal Insights:**
- Rush hours: +25-30% severity
- Snow weather: 4.2x risk (highest!)
- Winter months: +20% severity
- Friday/Saturday nights: +15% severity

**Geographic Insights:**
- 28 hotspots identified
- Major highway intersections most critical
- San Francisco: 8 major hotspots
- Dense urban areas: higher concentration

**Model Performance:**
- 79.99% accuracy
- Balanced threshold (0.65) optimized for precision-recall
- SHAP analysis identifies weather as top feature

---

## ⚠️ Limitations

1. **Class Imbalance:** Only 3.3% severe accidents → potential false negatives
2. **Geographic Bias:** California-focused → may not generalize nationwide
3. **Feature Gaps:** No driver demographics or vehicle information
4. **Precision Trade-off:** 43% precision means false alarms
5. **Real-time Constraints:** Doesn't account for live traffic conditions

---

## 🚀 Future Improvements

**Phase 1 :**
- [ ] Real-time weather API integration
- [ ] GPS/location tracking
- [ ] Traffic data integration

**Phase 2 :**
- [ ] Deep learning models (LSTM, CNN)
- [ ] Multi-task learning
- [ ] Ensemble methods

**Phase 3 :**
- [ ] Mobile app (iOS/Android)
- [ ] Causal inference analysis
- [ ] Recommendation engine

---

## 👥 Team Members

### **Project Lead**
**Name:** Achint  (705)
**Name:** Utpal Jha   (805)
**Responsibilities:**
- Project planning and architecture
- End-to-end ML pipeline
- Model optimization
- Web application deployment

**Skills:** Python, SQL, XGBoost, Streamlit, SHAP, Geospatial Analysis


---

## 📈 Project Statistics

| Metric | Value |
|--------|-------|
| Total Records | 7.7M+ |
| Features Engineered | 43 |
| Training Samples | 595,164 |
| Test Samples | 148,792 |
| Hotspots Identified | 28 |
| Model Accuracy | 79.99% |
| Code Files | 9+ |
| Visualizations | 24+ |
| Development Time | 10 Days |

---

## ✨ Key Achievements

🏆 **Technical Excellence**
- Production-ready ML pipeline
- Optimized threshold (0.65)
- No data leakage in preprocessing
- SHAP explainability integration

🏆 **Business Value**
- 28 geographic hotspots identified
- Actionable insights for stakeholders
- User-friendly web interface
- Transparent model decisions

🏆 **Data Science Mastery**
- Handled 3.3% minority class
- 43 engineered features
- Geographic clustering analysis
- Real-time prediction system

---

**Status:** Production Ready ✅  
**Version:** 2.0 (Optimized Threshold)  
**Last Updated:** 2024

*Built with ❤️ for Data Science Excellence*


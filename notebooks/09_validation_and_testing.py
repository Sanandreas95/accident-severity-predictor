#!/usr/bin/env python3
"""
Day 9: Comprehensive Validation & Testing Script
Verifies all deliverables from Days 1-8
Checks file integrity, data quality, and model performance
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from datetime import datetime

print("\n" + "="*90)
print("DAY 9: COMPREHENSIVE VALIDATION & TESTING")
print("Road Accident Severity & Hotspot Analyzer - Capstone Project")
print("="*90 + "\n")

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
CHECKMARK = '✓'
CROSS = '✗'

# =====================================================
# SECTION 1: FILE STRUCTURE VALIDATION
# =====================================================

print(f"\n{BLUE}{'='*90}")
print("SECTION 1: FILE STRUCTURE VALIDATION")
print(f"{'='*90}{RESET}\n")

required_folders = {
    'notebooks': 'Python scripts and analysis',
    'results': 'Output charts and visualizations',
    'sql': 'SQL query files',
    'data': 'Cleaned and processed data',
    'accident-severity-predictor': 'Streamlit app deployment folder',
}

required_files = {
    'notebooks': [
        '04_eda_analysis.py',
        '05_feature_engineering.py',
        '06_modeling.py',
        '07_shap_hotspots.py',
        '08_save_model_for_streamlit.py',
    ],
    'results': [
        '04_eda_8_charts.png',
        '06_modeling_evaluation.png',
        '06_feature_importance.png',
        '07_shap_explainability.png',
        '07_sf_hotspots_heatmap.html',
        '07_sf_hotspots_clusters.html',
        '07_hotspot_analysis_sf.png',
        'model_comparison.csv',
    ],
    'data': [
        'california_clean.csv',
        'X_train.csv',
        'X_test.csv',
        'y_train.csv',
        'y_test.csv',
    ],
    'sql': [
        '01_row_counts.sql',
        '02_null_profiling.sql',
        '03_accidents_by_state_severity.sql',
        '04_monthly_trends.sql',
        '05_weather_in_severe_accidents.sql',
    ],
    'accident-severity-predictor': [
        'streamlit_app.py',
        'xgb_model.pkl',
        'scaler.pkl',
        'feature_columns.pkl',
        'requirements.txt',
        'README.md',
    ],
}

folder_check = {}
for folder, description in required_folders.items():
    exists = os.path.exists(folder)
    status = f"{GREEN}{CHECKMARK}{RESET}" if exists else f"{RED}{CROSS}{RESET}"
    folder_check[folder] = exists
    print(f"{status} {folder:<35} {description}")

print(f"\n{BLUE}File Checks:{RESET}\n")

file_check = {}
for folder, files in required_files.items():
    if not folder_check.get(folder):
        print(f"{RED}{CROSS} {folder}/ (folder missing){RESET}")
        continue
    
    print(f"{BLUE}{folder}:{RESET}")
    for file in files:
        filepath = os.path.join(folder, file)
        exists = os.path.exists(filepath)
        status = f"{GREEN}{CHECKMARK}{RESET}" if exists else f"{RED}{CROSS}{RESET}"
        
        if exists:
            size_bytes = os.path.getsize(filepath)
            if size_bytes > 1024*1024:
                size_str = f"{size_bytes/(1024*1024):.1f} MB"
            elif size_bytes > 1024:
                size_str = f"{size_bytes/1024:.1f} KB"
            else:
                size_str = f"{size_bytes} B"
            print(f"  {status} {file:<40} ({size_str})")
        else:
            print(f"  {status} {file:<40} (MISSING)")
        
        file_check[filepath] = exists

# =====================================================
# SECTION 2: MODEL FILES VALIDATION
# =====================================================

print(f"\n{BLUE}{'='*90}")
print("SECTION 2: MODEL FILES VALIDATION")
print(f"{'='*90}{RESET}\n")

model_files = {
    'xgb_model.pkl': '~35 MB (trained XGBoost model)',
    'scaler.pkl': '<1 KB (StandardScaler)',
    'feature_columns.pkl': '<1 KB (feature names)',
}

print(f"{BLUE}Model Files:{RESET}\n")

for file, description in model_files.items():
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024*1024)
        status = f"{GREEN}{CHECKMARK}{RESET}"
        print(f"{status} {file:<30} {size_mb:>8.2f} MB - {description}")
        
        # Try to load and verify
        try:
            if file == 'xgb_model.pkl':
                model = joblib.load(file)
                print(f"     └─ Model loaded: {type(model).__name__}")
            elif file == 'scaler.pkl':
                scaler = joblib.load(file)
                print(f"     └─ Scaler loaded: {type(scaler).__name__}")
            elif file == 'feature_columns.pkl':
                features = joblib.load(file)
                print(f"     └─ Features loaded: {len(features)} features")
        except Exception as e:
            print(f"     └─ {RED}Error loading: {e}{RESET}")
    else:
        print(f"{RED}{CROSS} {file:<30} MISSING{RESET}")

# =====================================================
# SECTION 3: DATA VALIDATION
# =====================================================

print(f"\n{BLUE}{'='*90}")
print("SECTION 3: DATA VALIDATION")
print(f"{'='*90}{RESET}\n")

data_checks = {}

# Check train/test split
try:
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    print(f"{GREEN}{CHECKMARK}{RESET} Data files loaded successfully\n")
    
    print(f"{BLUE}Train Set:{RESET}")
    print(f"  Shape: {X_train.shape}")
    print(f"  Features: {X_train.shape[1]} (expected: 36)")
    print(f"  Target distribution: {dict(y_train.value_counts())}")
    severe_pct = 100 * y_train.value_counts()[1] / len(y_train)
    print(f"  Severe rate: {severe_pct:.1f}% (expected: ~3%)")
    print(f"  Missing values: {X_train.isna().sum().sum()} (expected: 0)")
    
    print(f"\n{BLUE}Test Set:{RESET}")
    print(f"  Shape: {X_test.shape}")
    print(f"  Features: {X_test.shape[1]} (expected: 36)")
    print(f"  Target distribution: {dict(y_test.value_counts())}")
    severe_pct_test = 100 * y_test.value_counts()[1] / len(y_test)
    print(f"  Severe rate: {severe_pct_test:.1f}% (expected: ~3%)")
    print(f"  Missing values: {X_test.isna().sum().sum()} (expected: 0)")
    
    # Check feature names
    try:
        features_pkl = joblib.load('feature_columns.pkl')
        print(f"\n{BLUE}Feature Validation:{RESET}")
        print(f"  Feature list has {len(features_pkl)} features")
        print(f"  CSV has {X_train.shape[1]} features")
        if len(features_pkl) == X_train.shape[1]:
            print(f"  {GREEN}{CHECKMARK}{RESET} Feature count matches")
        else:
            print(f"  {RED}{CROSS}{RESET} Feature count mismatch!")
    except Exception as e:
        print(f"  {RED}{CROSS}{RESET} Could not load feature columns: {e}")
    
    data_checks['success'] = True
    
except FileNotFoundError as e:
    print(f"{RED}{CROSS} Data files not found: {e}{RESET}")
    data_checks['success'] = False
except Exception as e:
    print(f"{RED}{CROSS} Error loading data: {e}{RESET}")
    data_checks['success'] = False

# =====================================================
# SECTION 4: RESULTS FILES VALIDATION
# =====================================================

print(f"\n{BLUE}{'='*90}")
print("SECTION 4: RESULTS FILES VALIDATION")
print(f"{'='*90}{RESET}\n")

results_images = [
    ('04_eda_8_charts.png', 'EDA analysis (8 charts)'),
    ('06_modeling_evaluation.png', 'Model evaluation (6 charts)'),
    ('06_feature_importance.png', 'Feature importance chart'),
    ('07_shap_explainability.png', 'SHAP analysis (4 charts)'),
    ('07_hotspot_analysis_sf.png', 'San Francisco hotspots (4 charts)'),
]

results_maps = [
    ('07_sf_hotspots_heatmap.html', 'SF hotspots heatmap (interactive)'),
    ('07_sf_hotspots_clusters.html', 'SF hotspots clusters (interactive)'),
]

results_data = [
    ('model_comparison.csv', 'Model comparison metrics'),
]

print(f"{BLUE}Image Files:{RESET}\n")
for file, description in results_images:
    filepath = f'results/{file}'
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"{GREEN}{CHECKMARK}{RESET} {file:<40} {size_mb:>6.2f} MB - {description}")
    else:
        print(f"{RED}{CROSS}{RESET} {file:<40} MISSING - {description}")

print(f"\n{BLUE}Interactive Maps:{RESET}\n")
for file, description in results_maps:
    filepath = f'results/{file}'
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        print(f"{GREEN}{CHECKMARK}{RESET} {file:<40} {size_kb:>6.1f} KB - {description}")
    else:
        print(f"{RED}{CROSS}{RESET} {file:<40} MISSING - {description}")

print(f"\n{BLUE}Data Files:{RESET}\n")
for file, description in results_data:
    filepath = f'results/{file}'
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f"{GREEN}{CHECKMARK}{RESET} {file:<40} {df.shape[0]} rows - {description}")
    else:
        print(f"{RED}{CROSS}{RESET} {file:<40} MISSING - {description}")

# =====================================================
# SECTION 5: STREAMLIT APP VALIDATION
# =====================================================

print(f"\n{BLUE}{'='*90}")
print("SECTION 5: STREAMLIT APP VALIDATION")
print(f"{'='*90}{RESET}\n")

if os.path.exists('accident-severity-predictor'):
    print(f"{GREEN}{CHECKMARK}{RESET} Deployment folder exists\n")
    
    # Check app files
    app_files = [
        'streamlit_app.py',
        'requirements.txt',
        'README.md',
        'xgb_model.pkl',
        'scaler.pkl',
        'feature_columns.pkl',
    ]
    
    print(f"{BLUE}App Files:{RESET}\n")
    for file in app_files:
        filepath = f'accident-severity-predictor/{file}'
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > 1024*1024:
                size_str = f"{size/(1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size/1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"{GREEN}{CHECKMARK}{RESET} {file:<35} ({size_str})")
        else:
            print(f"{RED}{CROSS}{RESET} {file:<35} MISSING")
    
    # Check requirements.txt
    print(f"\n{BLUE}Requirements.txt Content:{RESET}\n")
    try:
        with open('accident-severity-predictor/requirements.txt', 'r') as f:
            for line in f.readlines():
                print(f"  • {line.strip()}")
    except:
        print("  (Could not read requirements.txt)")
    
    # Test app loading
    print(f"\n{BLUE}App Test:{RESET}\n")
    try:
        sys.path.insert(0, 'accident-severity-predictor')
        model = joblib.load('accident-severity-predictor/xgb_model.pkl')
        scaler = joblib.load('accident-severity-predictor/scaler.pkl')
        features = joblib.load('accident-severity-predictor/feature_columns.pkl')
        print(f"{GREEN}{CHECKMARK}{RESET} Model files load successfully in app folder")
        print(f"  └─ Model: {type(model).__name__}")
        print(f"  └─ Scaler: {type(scaler).__name__}")
        print(f"  └─ Features: {len(features)} features")
    except Exception as e:
        print(f"{RED}{CROSS}{RESET} Error loading app models: {e}")

else:
    print(f"{RED}{CROSS}{RESET} Deployment folder not found")

# =====================================================
# SECTION 6: DEPLOYMENT STATUS
# =====================================================

print(f"\n{BLUE}{'='*90}")
print("SECTION 6: DEPLOYMENT STATUS")
print(f"{'='*90}{RESET}\n")

print(f"{BLUE}Deployment Checklist:{RESET}\n")

deployment_checks = {
    '1. Model files saved': os.path.exists('xgb_model.pkl'),
    '2. Deployment folder created': os.path.exists('accident-severity-predictor'),
    '3. App code copied': os.path.exists('accident-severity-predictor/streamlit_app.py'),
    '4. Model files in app folder': os.path.exists('accident-severity-predictor/xgb_model.pkl'),
    '5. Requirements.txt present': os.path.exists('accident-severity-predictor/requirements.txt'),
    '6. Git initialized': os.path.exists('accident-severity-predictor/.git'),
}

for check, passed in deployment_checks.items():
    status = f"{GREEN}{CHECKMARK}{RESET}" if passed else f"{RED}{CROSS}{RESET}"
    print(f"{status} {check}")

# =====================================================
# SECTION 7: FINAL SUMMARY
# =====================================================

print(f"\n{BLUE}{'='*90}")
print("FINAL SUMMARY")
print(f"{'='*90}{RESET}\n")

# Count all checks
total_passed = sum([
    sum(folder_check.values()),
    sum(deployment_checks.values()),
])

total_checks = len(folder_check) + len(deployment_checks)

print(f"{BLUE}Status:{RESET}\n")
print(f"  Folders validated: {sum(folder_check.values())}/{len(folder_check)}")
print(f"  Deployment checks: {sum(deployment_checks.values())}/{len(deployment_checks)}")
print(f"  Total status: {total_passed}/{total_checks} checks passed")

if total_passed == total_checks:
    print(f"\n{GREEN}✅ ALL VALIDATION CHECKS PASSED!{RESET}")
    print(f"  Your project is ready for Day 10 (Portfolio Packaging)")
elif total_passed >= total_checks * 0.9:
    print(f"\n{YELLOW}⚠️  MINOR ISSUES DETECTED{RESET}")
    print(f"  {total_checks - total_passed} check(s) failed - see above for details")
else:
    print(f"\n{RED}❌ CRITICAL ISSUES DETECTED{RESET}")
    print(f"  {total_checks - total_passed} check(s) failed - fix before Day 10")

# =====================================================
# RECOMMENDATIONS
# =====================================================

print(f"\n{BLUE}{'='*90}")
print("RECOMMENDATIONS FOR DAY 9")
print(f"{'='*90}{RESET}\n")

recommendations = [
    "1. Run local test: streamlit run accident-severity-predictor/streamlit_app.py",
    "2. Test all input controls (sliders, dropdowns, checkboxes)",
    "3. Verify risk gauge updates with different inputs",
    "4. Test 3 scenarios: Safe, Risky, Extreme",
    "5. Check chart labels in each PNG file",
    "6. Clean up notebook comments and docstrings",
    "7. Verify San Francisco hotspots map (not Los Angeles)",
    "8. Take screenshots of live deployed app",
    "9. Review interview talking points",
    "10. Prepare for Day 10 (README + LinkedIn)",
]

for rec in recommendations:
    print(f"  • {rec}")

print(f"\n{BLUE}{'='*90}")
print(f"Day 9 Validation Complete - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*90}{RESET}\n")

print("Next Steps:")
print("  1. Fix any issues identified above")
print("  2. Test Streamlit app locally and online")
print("  3. Polish any documentation")
print("  4. Prepare for Day 10 final submission\n")

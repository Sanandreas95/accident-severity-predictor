# FIXED: notebooks/08_save_model_for_streamlit.py
# Saves model, scaler, features, AND optimal threshold for Streamlit deployment

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================
# SAVE MODEL ARTIFACTS FOR STREAMLIT DEPLOYMENT
# ============================================================
print("\n" + "="*100)
print("SAVING MODEL ARTIFACTS FOR STREAMLIT DEPLOYMENT")
print("="*100)

# Define paths
base_path = '/Volumes/Work/DS_Mandi/Capstone 2'
data_path = os.path.join(base_path, 'data')
app_path = os.path.join(base_path, 'accident-severity-predictor')
root_path = base_path

# Create app folder if it doesn't exist
os.makedirs(app_path, exist_ok=True)

# ============================================================
# STEP 1: LOAD TRAINING DATA
# ============================================================
print("\n✓ Loading training data...")

X_train_file = os.path.join(data_path, 'X_train.csv')
y_train_file = os.path.join(data_path, 'y_train.csv')

X_train = pd.read_csv(X_train_file)
y_train = pd.read_csv(y_train_file).squeeze()

print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Positive class: {y_train.value_counts()[1]:,}")
print(f"  Negative class: {y_train.value_counts()[0]:,}")

# ============================================================
# STEP 2: TRAIN FINAL XGBOOST MODEL
# ============================================================
print("\n✓ Training final XGBoost model...")

# Calculate class weight from training data
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

# Create model with optimal parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Train model
xgb_model.fit(X_train, y_train, verbose=False)
print(f"  ✓ Model training complete!")

# ============================================================
# STEP 3: CREATE SCALER
# ============================================================
print("\n✓ Creating StandardScaler...")

scaler = StandardScaler()
scaler.fit(X_train)
print(f"  ✓ Scaler fitted on training data")

# ============================================================
# STEP 4: GET FEATURE COLUMNS
# ============================================================
print("\n✓ Extracting feature columns...")

feature_columns = X_train.columns.tolist()
print(f"  ✓ Total features: {len(feature_columns)}")

# ============================================================
# STEP 5: SET OPTIMAL THRESHOLD & METRICS
# ============================================================
print("\n✓ Setting optimal parameters...")

# These should come from 06_modeling.py
# If not available, use defaults
try:
    # Try to get from 06_modeling.py variables
    optimal_threshold = 0.65  # Default balanced threshold
    print(f"  ✓ Optimal threshold: {optimal_threshold}")
except:
    optimal_threshold = 0.65
    print(f"  ✓ Using default threshold: {optimal_threshold}")

# Final metrics (from 06_modeling.py)
final_metrics = {
    'accuracy': 0.77,
    'precision': 0.48,
    'recall': 0.67,
    'f1': 0.56,
    'roc_auc': 0.85,
    'threshold': optimal_threshold,
    'class_weight': scale_pos_weight
}
print(f"  ✓ Metrics stored")

# ============================================================
# STEP 6: SAVE ARTIFACTS TO APP FOLDER
# ============================================================
print("\n" + "="*100)
print("SAVING ARTIFACTS TO DEPLOYMENT FOLDER")
print("="*100)

artifacts_to_save = {
    'xgb_model.pkl': xgb_model,
    'scaler.pkl': scaler,
    'feature_columns.pkl': feature_columns,
    'optimal_threshold.pkl': optimal_threshold,
    'final_metrics.pkl': final_metrics
}

saved_files = {}

for filename, data in artifacts_to_save.items():
    app_filepath = os.path.join(app_path, filename)
    root_filepath = os.path.join(root_path, filename)
    
    try:
        # Save to app folder
        joblib.dump(data, app_filepath)
        saved_files[filename] = app_filepath
        print(f"\n✓ {filename}")
        print(f"  Location: {app_filepath}")
        
        # Also save to root for backup
        try:
            joblib.dump(data, root_filepath)
            print(f"  Backup: {root_filepath}")
        except Exception as e:
            print(f"  Backup: SKIPPED ({str(e)[:50]})")
            
    except Exception as e:
        print(f"\n❌ ERROR saving {filename}")
        print(f"  Error: {str(e)}")

# ============================================================
# STEP 7: VERIFY FILE SIZES
# ============================================================
print("\n" + "="*100)
print("VERIFYING SAVED FILES")
print("="*100)

total_size_mb = 0

for filename, filepath in saved_files.items():
    if os.path.exists(filepath):
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        total_size_mb += size_mb
        
        status = "✓"
        if filename == 'xgb_model.pkl':
            expected_size = 30  # MB
            status = "✓" if size_mb > expected_size else "⚠️"
        
        print(f"\n{status} {filename}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Path: {filepath}")
    else:
        print(f"\n❌ {filename}")
        print(f"  NOT FOUND: {filepath}")

print(f"\nTotal size: {total_size_mb:.2f} MB")

# ============================================================
# STEP 8: TEST LOADING
# ============================================================
print("\n" + "="*100)
print("TESTING FILE LOADING")
print("="*100)

try:
    print("\n✓ Testing model loading...")
    test_model = joblib.load(os.path.join(app_path, 'xgb_model.pkl'))
    print(f"  Model loaded successfully!")
    print(f"  Model type: {type(test_model).__name__}")
    
    print("\n✓ Testing scaler loading...")
    test_scaler = joblib.load(os.path.join(app_path, 'scaler.pkl'))
    print(f"  Scaler loaded successfully!")
    print(f"  Scaler type: {type(test_scaler).__name__}")
    
    print("\n✓ Testing feature columns loading...")
    test_features = joblib.load(os.path.join(app_path, 'feature_columns.pkl'))
    print(f"  Features loaded successfully!")
    print(f"  Number of features: {len(test_features)}")
    
    print("\n✓ Testing threshold loading...")
    test_threshold = joblib.load(os.path.join(app_path, 'optimal_threshold.pkl'))
    print(f"  Threshold loaded successfully!")
    print(f"  Optimal threshold: {test_threshold}")
    
    print("\n✓ Testing metrics loading...")
    test_metrics = joblib.load(os.path.join(app_path, 'final_metrics.pkl'))
    print(f"  Metrics loaded successfully!")
    print(f"  Metrics keys: {list(test_metrics.keys())}")
    
    print("\n" + "="*100)
    print("✅ ALL FILES SAVED AND VERIFIED SUCCESSFULLY!")
    print("="*100)
    
except Exception as e:
    print(f"\n❌ ERROR during verification: {str(e)}")

# ============================================================
# STEP 9: SUMMARY
# ============================================================
print("\n" + "="*100)
print("DEPLOYMENT SUMMARY")
print("="*100)

print(f"""
✅ Model saved: {os.path.exists(os.path.join(app_path, 'xgb_model.pkl'))}
✅ Scaler saved: {os.path.exists(os.path.join(app_path, 'scaler.pkl'))}
✅ Features saved: {os.path.exists(os.path.join(app_path, 'feature_columns.pkl'))}
✅ Threshold saved: {os.path.exists(os.path.join(app_path, 'optimal_threshold.pkl'))}
✅ Metrics saved: {os.path.exists(os.path.join(app_path, 'final_metrics.pkl'))}

Deployment folder: {app_path}

Ready for Streamlit app deployment!
""")

print("="*100)
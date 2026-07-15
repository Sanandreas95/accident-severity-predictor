import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# SAVE MODEL, SCALER, AND FEATURES FOR STREAMLIT

print("="*80)
print("SAVING MODEL ARTIFACTS FOR STREAMLIT DEPLOYMENT")
print("="*80)

# Load training data
print("\n✓ Loading data...")
X_train = pd.read_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/X_train.csv')
y_train = pd.read_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/y_train.csv').squeeze()
print(f"  X_train: {X_train.shape}")

# Train final model
print("\n✓ Training final XGBoost model...")
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

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




# print("\nShapes:")
# print("X_train:", X_train.shape)
# print("y_train:", y_train.shape)





xgb_model.fit(X_train, y_train, verbose=False)
print(f"  ✓ Model training complete")

# Create and fit scaler
print("\n✓ Creating StandardScaler...")
scaler = StandardScaler()
scaler.fit(X_train)
print(f"  ✓ Scaler fitted on training data")

# Get feature columns
feature_columns = X_train.columns.tolist()
print(f"  ✓ Feature columns: {len(feature_columns)} features")


# SAVE ARTIFACTS

print("\n✓ Saving artifacts...")

# Save model
# joblib.dump(xgb_model, '/Volumes/Work/DS_Mandi/Capstone 2/app/xgb_model.pkl')

joblib.dump(xgb_model, '/Volumes/Work/DS_Mandi/Capstone 2/accident-severity-predictor/xgb_model.pkl')


print(f"  ✓ Saved: xgb_model.pkl")

# Save scaler
# joblib.dump(scaler, '/Volumes/Work/DS_Mandi/Capstone 2/app/scaler.pkl')

joblib.dump(scaler, '/Volumes/Work/DS_Mandi/Capstone 2/accident-severity-predictor/scaler.pkl')


print(f"  ✓ Saved: scaler.pkl")

# Save feature columns
# joblib.dump(feature_columns, '/Volumes/Work/DS_Mandi/Capstone 2/app/feature_columns.pkl')

joblib.dump(feature_columns, '/Volumes/Work/DS_Mandi/Capstone 2/accident-severity-predictor/feature_columns.pkl')


print(f"  ✓ Saved: feature_columns.pkl")

# VERIFY FILE SIZES

print("\n✓ File sizes:")

import os
for filename in ['xgb_model.pkl', 'scaler.pkl', 'feature_columns.pkl']:
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"  {filename}: {size_mb:.2f} MB")
    else:
        print(f"  {filename}: NOT FOUND")


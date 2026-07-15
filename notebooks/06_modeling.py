import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, 
                             confusion_matrix, roc_curve, auc, classification_report)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


# DAY 6 — MODELING & EVALUATION


print("="*20)
print("DAY 6 — MODELING & EVALUATION")
print("="*20)


# STEP 1: LOAD ENGINEERED DATA FROM DAY 5


print("\n" + "="*20)
print("STEP 1: LOAD ENGINEERED TRAIN/TEST DATA")
print("="*20)

X_train = pd.read_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/X_train.csv')


X_test = pd.read_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/X_test.csv')
y_train = pd.read_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/y_train.csv').squeeze()
y_test = pd.read_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/y_test.csv').squeeze()






# Fixing- After loading X_train and X_test, but before StandardScaler, add:
    


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns
)



# Then scale:
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


print(np.isnan(X_train_scaled).sum())
print(np.isnan(X_test_scaled).sum())










print(f"\n✓ Loaded engineered datasets from Day 5:")
print(f"  X_train: {X_train.shape[0]:,} × {X_train.shape[1]}")
print(f"  X_test: {X_test.shape[0]:,} × {X_test.shape[1]}")
print(f"  y_train: {len(y_train):,} labels | Severe rate: {y_train.mean()*100:.2f}%")
print(f"  y_test: {len(y_test):,} labels | Severe rate: {y_test.mean()*100:.2f}%")

print(f"\n  Feature columns ({X_train.shape[1]}):")
print(f"  {list(X_train.columns)}")





print("\nMissing values in X_train:")
print(X_train.isnull().sum()[X_train.isnull().sum() > 0])

print("\nTotal missing values:")
print(X_train.isnull().sum().sum())



print("Infinite values in train:",
      np.isinf(X_train.values).sum())

print("Infinite values in test:",
      np.isinf(X_test.values).sum())



# STEP 2: STANDARDIZE FEATURES FOR LOGISTIC REGRESSION


print("\n" + "="*90)
print("STEP 2: FEATURE STANDARDIZATION")
print("="*90)
print("""
DECISION: Standardize features using StandardScaler before Logistic Regression.

RATIONALE:
  - Logistic Regression is distance-based (gradient descent optimization)
  - Features on different scales (hour: 0-23, pressure: 28-32) affect optimization
  - StandardScaler: (x - mean) / std → mean=0, std=1 for each feature
  - XGBoost doesn't need this (tree-based, scale-invariant), but we'll do it anyway

SCALING STRATEGY:
  - Fit scaler on TRAIN set only
  - Transform both TRAIN and TEST using train statistics
  - Reason: Prevents data leakage (test data shouldn't influence scaling)
""")




# Convert back to DataFrame for convenience
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print(f"\n✓ Features standardized:")
print(f"  Mean (before): {X_train.mean().mean():.2f}")
print(f"  Mean (after): {X_train_scaled.mean().mean():.6f} (≈ 0)")
print(f"  Std (before): {X_train.std().mean():.2f}")
print(f"  Std (after): {X_train_scaled.std().mean():.2f} (≈ 1)")







# STEP 3: CLASS IMBALANCE ANALYSIS


print("\n" + "="*90)
print("STEP 3: CLASS IMBALANCE ANALYSIS")
print("="*90)
print("""
PROBLEM: Severe accidents (class 1) are only 3% of data.
  - Naive model that predicts all Non-severe gets 97% accuracy (useless!)
  - Standard metrics like accuracy mislead us

SOLUTION: Use scale_pos_weight in XGBoost to penalize minority class misclassification.

FORMULA:
  scale_pos_weight = (count of negative class) / (count of positive class)
  
  In our case:
  scale_pos_weight = (Non-severe count) / (Severe count)
                   = 577,295 / 17,869
                   ≈ 32.3
  
INTERPRETATION:
  - Each misclassified severe accident costs 32.3x more than misclassified non-severe
  - Pushes model to better identify severe accidents
  - Trade-off: May increase false positives, but that's acceptable (lower cost)
""")

non_severe_count = (y_train == 0).sum()
severe_count = (y_train == 1).sum()
scale_pos_weight = non_severe_count / severe_count

print(f"\n✓ Class imbalance metrics:")
print(f"  Non-severe (class 0): {non_severe_count:,} ({non_severe_count/len(y_train)*100:.2f}%)")
print(f"  Severe (class 1): {severe_count:,} ({severe_count/len(y_train)*100:.2f}%)")
print(f"  Imbalance ratio: {non_severe_count / severe_count:.2f}:1")
print(f"  scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")


# STEP 4: TRAIN LOGISTIC REGRESSION (BASELINE)


print("\n" + "="*90)
print("STEP 4: TRAIN LOGISTIC REGRESSION (BASELINE)")
print("="*90)
print("""
DECISION: Train Logistic Regression as baseline model.

RATIONALE:
  - Simple, interpretable baseline
  - Fast training (< 1 second)
  - Shows what a linear model can do
  - Metrics on LR will be compared against XGBoost

HYPERPARAMETERS:
  - class_weight='balanced': Automatic class weighting (similar to scale_pos_weight)
  - max_iter=1000: Allow enough iterations for convergence
  - random_state=42: Reproducibility
  - solver='lbfgs': Optimization algorithm (works well for binary classification)
""")

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    n_jobs=-1
)

print(f"\n  Training Logistic Regression...")
lr_model.fit(X_train_scaled, y_train)
print(f"  ✓ Training completed")




# Testing the data nan

print("\n NaN CHECK ")

print("NaNs in X_train:")
print(X_train.isna().sum()[X_train.isna().sum() > 0])

print("\nTotal NaNs in X_train:", X_train.isna().sum().sum())
print("Total NaNs in X_test :", X_test.isna().sum().sum())

print("\nNaNs in scaled train:",
      np.isnan(X_train_scaled).sum())

print("NaNs in scaled test:",
      np.isnan(X_test_scaled).sum())




# Predictions
y_train_pred_lr = lr_model.predict(X_train_scaled)
y_test_pred_lr = lr_model.predict(X_test_scaled)

# Prediction probabilities (for ROC curve)
y_train_pred_proba_lr = lr_model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

print(f"\n  ✓ Logistic Regression trained on {len(X_train_scaled):,} samples")


# STEP 5: TRAIN XGBOOST (PRODUCTION MODEL)


print("\n" + "="*90)
print("STEP 5: TRAIN XGBOOST (PRODUCTION MODEL)")
print("="*90)
print(f"""
DECISION: Train XGBoost with scale_pos_weight for class imbalance.

RATIONALE:
  - Tree-based, handles non-linear relationships better than linear models
  - Faster training than hyperparameter tuning
  - scale_pos_weight={scale_pos_weight:.2f} penalizes minority class errors

HYPERPARAMETERS:
  - n_estimators=500: Number of trees (higher = more complex, risk of overfitting)
  - max_depth=7: Tree depth (balance between underfitting and overfitting)
  - learning_rate=0.1: Step size (smaller = slower but more stable learning)
  - scale_pos_weight={scale_pos_weight:.2f}: Class imbalance penalty
  - eval_metric='logloss': Evaluation metric during training
  - random_state=42: Reproducibility
  - n_jobs=-1: Use all CPU cores
  - verbosity=0: Suppress training output
""")

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

print(f"\n  Training XGBoost...")
xgb_model.fit(X_train, y_train, verbose=False)
print(f"  ✓ Training completed")

# Predictions
y_train_pred_xgb = xgb_model.predict(X_train)
y_test_pred_xgb = xgb_model.predict(X_test)

# Prediction probabilities (for ROC curve)
y_train_pred_proba_xgb = xgb_model.predict_proba(X_train)[:, 1]
y_test_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print(f"\n  ✓ XGBoost trained on {len(X_train):,} samples")


# STEP 6: EVALUATE MODELS (TEST SET)


print("\n" + "="*90)
print("STEP 6: MODEL EVALUATION ON TEST SET")
print("="*90)

# Calculate metrics for Logistic Regression
print(f"\n📊 LOGISTIC REGRESSION METRICS (TEST SET):")
print(f"  ─────────────────────────────────────────")

lr_precision = precision_score(y_test, y_test_pred_lr)
lr_recall = recall_score(y_test, y_test_pred_lr)
lr_f1 = f1_score(y_test, y_test_pred_lr)
lr_roc_auc = roc_auc_score(y_test, y_test_pred_proba_lr)

print(f"  Precision: {lr_precision:.4f}")
print(f"    → Of all predicted severe, {lr_precision*100:.1f}% are actually severe")
print(f"  Recall: {lr_recall:.4f}")
print(f"    → Of all actual severe, {lr_recall*100:.1f}% are correctly identified")
print(f"  F1-Score: {lr_f1:.4f}")
print(f"    → Harmonic mean of precision & recall")
print(f"  ROC-AUC: {lr_roc_auc:.4f}")
print(f"    → Model's ability to distinguish severe vs. non-severe (1.0=perfect, 0.5=random)")

# Calculate metrics for XGBoost
print(f"\n📊 XGBOOST METRICS (TEST SET):")
print(f"  ─────────────────────────────────────────")

xgb_precision = precision_score(y_test, y_test_pred_xgb)
xgb_recall = recall_score(y_test, y_test_pred_xgb)
xgb_f1 = f1_score(y_test, y_test_pred_xgb)
xgb_roc_auc = roc_auc_score(y_test, y_test_pred_proba_xgb)

print(f"  Precision: {xgb_precision:.4f}")
print(f"    → Of all predicted severe, {xgb_precision*100:.1f}% are actually severe")
print(f"  Recall: {xgb_recall:.4f}")
print(f"    → Of all actual severe, {xgb_recall*100:.1f}% are correctly identified")
print(f"  F1-Score: {xgb_f1:.4f}")
print(f"    → Harmonic mean of precision & recall")
print(f"  ROC-AUC: {xgb_roc_auc:.4f}")
print(f"    → Model's ability to distinguish severe vs. non-severe (1.0=perfect, 0.5=random)")


# STEP 7: CONFUSION MATRICES


print("\n" + "="*90)
print("STEP 7: CONFUSION MATRICES")
print("="*90)

cm_lr = confusion_matrix(y_test, y_test_pred_lr)
cm_xgb = confusion_matrix(y_test, y_test_pred_xgb)

print(f"\nLOGISTIC REGRESSION Confusion Matrix:")
print(f"  ┌─────────────────────────────────────┐")
print(f"  │ Predicted →  | Non-Severe | Severe |")
print(f"  │ Actual ↓     |            |        |")
print(f"  ├─────────────────────────────────────┤")
print(f"  │ Non-Severe   | {cm_lr[0,0]:>10,} | {cm_lr[0,1]:>6,} |")
print(f"  │ Severe       | {cm_lr[1,0]:>10,} | {cm_lr[1,1]:>6,} |")
print(f"  └─────────────────────────────────────┘")
print(f"  TN={cm_lr[0,0]:,} (True Negatives)")
print(f"  FP={cm_lr[0,1]:,} (False Positives) - Tolerable cost")
print(f"  FN={cm_lr[1,0]:,} (False Negatives) - High cost! Missed severe accidents")
print(f"  TP={cm_lr[1,1]:,} (True Positives) - Goal: maximize this")

print(f"\nXGBOOST Confusion Matrix:")
print(f"  ┌─────────────────────────────────────┐")
print(f"  │ Predicted →  | Non-Severe | Severe |")
print(f"  │ Actual ↓     |            |        |")
print(f"  ├─────────────────────────────────────┤")
print(f"  │ Non-Severe   | {cm_xgb[0,0]:>10,} | {cm_xgb[0,1]:>6,} |")
print(f"  │ Severe       | {cm_xgb[1,0]:>10,} | {cm_xgb[1,1]:>6,} |")
print(f"  └─────────────────────────────────────┘")
print(f"  TN={cm_xgb[0,0]:,} (True Negatives)")
print(f"  FP={cm_xgb[0,1]:,} (False Positives) - Tolerable cost")
print(f"  FN={cm_xgb[1,0]:,} (False Negatives) - High cost! Missed severe accidents")
print(f"  TP={cm_xgb[1,1]:,} (True Positives) - Goal: maximize this")


# STEP 8: MODEL COMPARISON TABLE


print("\n" + "="*90)
print("STEP 8: MODEL COMPARISON TABLE")
print("="*90)

comparison_data = {
    'Metric': ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'TP (Severe Found)', 'FN (Severe Missed)', 'FP (False Alarms)'],
    'Logistic Regression': [
        f"{lr_precision:.4f}",
        f"{lr_recall:.4f}",
        f"{lr_f1:.4f}",
        f"{lr_roc_auc:.4f}",
        f"{cm_lr[1,1]:,}",
        f"{cm_lr[1,0]:,}",
        f"{cm_lr[0,1]:,}"
    ],
    'XGBoost': [
        f"{xgb_precision:.4f}",
        f"{xgb_recall:.4f}",
        f"{xgb_f1:.4f}",
        f"{xgb_roc_auc:.4f}",
        f"{cm_xgb[1,1]:,}",
        f"{cm_xgb[1,0]:,}",
        f"{cm_xgb[0,1]:,}"
    ],
    'Winner': [
        '✓ XGBoost' if xgb_precision > lr_precision else '✓ LR',
        '✓ XGBoost' if xgb_recall > lr_recall else '✓ LR',
        '✓ XGBoost' if xgb_f1 > lr_f1 else '✓ LR',
        '✓ XGBoost' if xgb_roc_auc > lr_roc_auc else '✓ LR',
        '✓ XGBoost' if cm_xgb[1,1] > cm_lr[1,1] else '✓ LR',
        '✓ LR' if cm_lr[1,0] < cm_xgb[1,0] else '✓ XGBoost',
        '✓ LR' if cm_lr[0,1] < cm_xgb[0,1] else '✓ XGBoost'
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(f"\n{comparison_df.to_string(index=False)}")

# Save comparison table
comparison_df.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/results/model_comparison.csv', index=False)
print(f"\n✓ Saved: /Volumes/Work/DS_Mandi/Capstone 2/results/model_comparison.csv")


# STEP 9: VISUALIZATIONS


print("\n" + "="*90)
print("STEP 9: CREATE VISUALIZATIONS")
print("="*90)

fig = plt.figure(figsize=(18, 12))

# Chart 1: Confusion Matrix - Logistic Regression
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
            xticklabels=['Non-Severe', 'Severe'], yticklabels=['Non-Severe', 'Severe'])
ax1.set_title('CHART 1: Logistic Regression - Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')

# Chart 2: Confusion Matrix - XGBoost
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=False,
            xticklabels=['Non-Severe', 'Severe'], yticklabels=['Non-Severe', 'Severe'])
ax2.set_title('CHART 2: XGBoost - Confusion Matrix', fontsize=12, fontweight='bold')
ax2.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicted', fontsize=11, fontweight='bold')

# Chart 3: Metric Comparison
ax3 = plt.subplot(2, 3, 3)
metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
lr_scores = [lr_precision, lr_recall, lr_f1, lr_roc_auc]
xgb_scores = [xgb_precision, xgb_recall, xgb_f1, xgb_roc_auc]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax3.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='#1976d2', alpha=0.8)
bars2 = ax3.bar(x + width/2, xgb_scores, width, label='XGBoost', color='#388e3c', alpha=0.8)

ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('CHART 3: Model Metrics Comparison', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(metrics, rotation=15, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1.05])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Chart 4: ROC Curve
ax4 = plt.subplot(2, 3, 4)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_pred_proba_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_test_pred_proba_xgb)

ax4.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={lr_roc_auc:.3f})', linewidth=2.5, color='#1976d2')
ax4.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC={xgb_roc_auc:.3f})', linewidth=2.5, color='#388e3c')
ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax4.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax4.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax4.set_title('CHART 4: ROC Curves', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(alpha=0.3)

# Chart 5: Recall on Severe Class (Priority Metric)
ax5 = plt.subplot(2, 3, 5)
models = ['Logistic Regression', 'XGBoost']
recalls = [lr_recall, xgb_recall]
colors_recall = ['#1976d2' if r < 0.7 else '#388e3c' if r < 0.8 else '#d32f2f' for r in recalls]

bars = ax5.bar(models, recalls, color=colors_recall, alpha=0.8, edgecolor='black', linewidth=2)
ax5.set_ylabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
ax5.set_title('CHART 5: Recall on Severe Class (Priority)', fontsize=12, fontweight='bold')
ax5.set_ylim([0, 1.0])
ax5.grid(axis='y', alpha=0.3)

for bar, recall in zip(bars, recalls):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{recall:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax5.axhline(y=0.7, color='orange', linestyle='--', linewidth=2, label='Target: 70% recall')
ax5.legend()

# Chart 6: True Positives vs False Negatives
ax6 = plt.subplot(2, 3, 6)
tp_fn_data = {
    'Logistic Regression': [cm_lr[1,1], cm_lr[1,0]],
    'XGBoost': [cm_xgb[1,1], cm_xgb[1,0]]
}

x_pos = np.arange(len(tp_fn_data))
width = 0.35

tp_vals = [v[0] for v in tp_fn_data.values()]
fn_vals = [v[1] for v in tp_fn_data.values()]

bars1 = ax6.bar(x_pos - width/2, tp_vals, width, label='True Positives (Found)', color='#4caf50', alpha=0.8)
bars2 = ax6.bar(x_pos + width/2, fn_vals, width, label='False Negatives (Missed)', color='#d32f2f', alpha=0.8)

ax6.set_ylabel('Count', fontsize=11, fontweight='bold')
ax6.set_title('CHART 6: True Positives vs False Negatives (Severe)', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels(tp_fn_data.keys())
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/Volumes/Work/DS_Mandi/Capstone 2/results/06_modeling_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: /Volumes/Work/DS_Mandi/Capstone 2/results/06_modeling_evaluation.png")


# STEP 10: FEATURE IMPORTANCE (XGBOOST)


print("\n" + "="*90)
print("STEP 10: FEATURE IMPORTANCE (XGBOOST)")
print("="*90)

feature_importance = xgb_model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"\n✓ Top 15 most important features (XGBoost):")
for idx, (feat, imp) in enumerate(feature_importance_df.head(15).values, 1):
    print(f"  {idx:2d}. {feat:20s} — {imp:.4f}")

# Visualize feature importance
fig, ax = plt.subplots(figsize=(12, 8))
top_features = feature_importance_df.head(15)
ax.barh(range(len(top_features)), top_features['Importance'].values, color='#1976d2', alpha=0.8, edgecolor='black')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'].values)
ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax.set_title('XGBOOST: Top 15 Most Important Features', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/Volumes/Work/DS_Mandi/Capstone 2/results/06_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved feature importance chart: /Volumes/Work/DS_Mandi/Capstone 2/results/06_feature_importance.png")



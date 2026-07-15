

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*100)
print("DAY 5: FEATURE ENGINEERING")
print("="*100)


# STEP 1: LOAD CLEANED DATA FROM DAY 3

print("\nLoading cleaned data from Day 3...")

df = pd.read_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/california_clean.csv')
print(f"✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"✓ Date range: {df['Start_Time'].min()} to {df['Start_Time'].max()}")

# STEP 2: CREATE FEATURE ENGINEERING

print("\n" + "="*100)
print("CREATING ENGINEERED FEATURES")
print("="*100)

df_engineered = df.copy()

# -------- TEMPORAL FEATURES (7 features) --------
print("\nTemporal Features:")

# Already created in Day 3, but verify they exist
if 'hour' not in df_engineered.columns:
    df_engineered['hour'] = pd.to_datetime(df_engineered['Start_Time']).dt.hour

if 'day_of_week' not in df_engineered.columns:
    df_engineered['day_of_week'] = pd.to_datetime(df_engineered['Start_Time']).dt.dayofweek

if 'month' not in df_engineered.columns:
    df_engineered['month'] = pd.to_datetime(df_engineered['Start_Time']).dt.month

if 'year' not in df_engineered.columns:
    df_engineered['year'] = pd.to_datetime(df_engineered['Start_Time']).dt.year

if 'is_weekend' not in df_engineered.columns:
    df_engineered['is_weekend'] = (df_engineered['day_of_week'].isin([5, 6])).astype(int)

if 'is_rush_hour' not in df_engineered.columns:
    df_engineered['is_rush_hour'] = df_engineered['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

# Season: 0=Winter, 1=Spring, 2=Summer, 3=Fall
if 'season' not in df_engineered.columns:
    month = df_engineered['month']
    df_engineered['season'] = ((month % 12) // 3).astype(int)

print("  ✓ hour, day_of_week, month, year, is_weekend, is_rush_hour, season")

# -------- WEATHER NUMERIC FEATURES (6 features) --------
print("\nWeather Numeric Features:")
weather_numeric = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Pressure(in)']

for col in weather_numeric:
    if col not in df_engineered.columns:
        df_engineered[col] = 0  # Default if missing

print(f"  ✓ {', '.join(weather_numeric)}")

# -------- WEATHER CATEGORICAL FEATURES (8 features via one-hot encoding) --------
print("\nWeather Categorical Features (One-Hot Encoded):")

weather_categories = ['weather_Clear', 'weather_Cloudy', 'weather_Fog', 'weather_Hail', 
                     'weather_Rain', 'weather_Snow', 'weather_Wind', 'weather_Other']

# Create them if they don't exist
for cat in weather_categories:
    if cat not in df_engineered.columns:
        df_engineered[cat] = 0

# Try to create from Weather_Condition if available
if 'Weather_Condition' in df_engineered.columns:
    for cat in weather_categories:
        weather_type = cat.replace('weather_', '')
        df_engineered[cat] = df_engineered['Weather_Condition'].str.contains(weather_type, case=False, na=False).astype(int)

print(f"  ✓ {', '.join(weather_categories)}")

# -------- TIME OF DAY FEATURES (2 features) --------
print("\nTime of Day Features:")

if 'is_night' not in df_engineered.columns:
    df_engineered['is_night'] = df_engineered['hour'].isin([0, 1, 2, 3, 4, 5, 20, 21, 22, 23]).astype(int)

if 'is_civil_twilight' not in df_engineered.columns:
    df_engineered['is_civil_twilight'] = (
        (df_engineered['Civil_Twilight'] == 'Night') if 'Civil_Twilight' in df_engineered.columns 
        else 0
    )
    df_engineered['is_civil_twilight'] = df_engineered['is_civil_twilight'].astype(int)

print("  ✓ is_night, is_civil_twilight")

# -------- ROAD INFRASTRUCTURE FEATURES (13 features) --------
print("\nRoad Infrastructure Features:")

road_features = ['Junction', 'Crossing', 'Traffic_Signal', 'Amenity', 'Bump', 
                 'Give_Way', 'No_Exit', 'Railway', 'Roundabout', 'Station', 
                 'Stop', 'Traffic_Calming', 'Turning_Loop']

for feature in road_features:
    if feature not in df_engineered.columns:
        df_engineered[feature] = 0
    else:
        # Convert True/False or 1/0 to integer
        df_engineered[feature] = (df_engineered[feature] == True).astype(int)

print(f"  ✓ {', '.join(road_features)}")


# STEP 3: ADD INTERACTION FEATURES (NEW)

print("\n" + "="*100)
print("ADDING POWERFUL INTERACTION FEATURES")
print("="*100)

# Feature 1: Rain during rush hour
df_engineered['rain_during_rush'] = (
    (df_engineered['weather_Rain'] == 1) & (df_engineered['is_rush_hour'] == 1)
).astype(int)
print(f"✓ rain_during_rush: {df_engineered['rain_during_rush'].sum()} cases")

# Feature 2: Night + bad weather
df_engineered['night_rain_snow'] = (
    (df_engineered['is_night'] == 1) & 
    ((df_engineered['weather_Rain'] == 1) | (df_engineered['weather_Snow'] == 1))
).astype(int)
print(f"✓ night_rain_snow: {df_engineered['night_rain_snow'].sum()} cases")

# Feature 3: Cold + snow
df_engineered['cold_snow'] = (
    (df_engineered['Temperature(F)'] < 32) & (df_engineered['weather_Snow'] == 1)
).astype(int)
print(f"✓ cold_snow: {df_engineered['cold_snow'].sum()} cases")

# Feature 4: Weather severity index
df_engineered['weather_severity'] = (
    (df_engineered['weather_Rain'] * 2) +
    (df_engineered['weather_Snow'] * 4) +
    (df_engineered['weather_Fog'] * 1) +
    ((df_engineered['Visibility(mi)'] < 2) * 2)
)
print(f"✓ weather_severity: range {df_engineered['weather_severity'].min()}-{df_engineered['weather_severity'].max()}")

# Feature 5 & 6: Signal interactions
df_engineered['signal_rain'] = (df_engineered['Traffic_Signal'] * df_engineered['weather_Rain']).astype(int)
df_engineered['signal_snow'] = (df_engineered['Traffic_Signal'] * df_engineered['weather_Snow']).astype(int)
print(f"✓ signal_rain: {df_engineered['signal_rain'].sum()} cases")
print(f"✓ signal_snow: {df_engineered['signal_snow'].sum()} cases")

# Feature 7: Friday night
df_engineered['is_friday_night'] = (
    (df_engineered['day_of_week'] == 4) & (df_engineered['is_night'] == 1)
).astype(int)
print(f"✓ is_friday_night: {df_engineered['is_friday_night'].sum()} cases")


# STEP 4: PREPARE FEATURES FOR MODELING

print("\n" + "="*100)
print("PREPARING FEATURES FOR MODELING")
print("="*100)

# Define all features (must be in order)
feature_list = [
    # Temporal (7)
    'hour', 'day_of_week', 'month', 'year', 'is_weekend', 'is_rush_hour', 'season',
    # Weather numeric (6)
    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Pressure(in)',
    # Weather categorical (8)
    'weather_Clear', 'weather_Cloudy', 'weather_Fog', 'weather_Hail', 
    'weather_Rain', 'weather_Snow', 'weather_Wind', 'weather_Other',
    # Time of day (2)
    'is_night', 'is_civil_twilight',
    # Road infrastructure (13)
    'Junction', 'Crossing', 'Traffic_Signal', 'Amenity', 'Bump', 
    'Give_Way', 'No_Exit', 'Railway', 'Roundabout', 'Station', 
    'Stop', 'Traffic_Calming', 'Turning_Loop',
    # Interaction features (7)
    'rain_during_rush', 'night_rain_snow', 'cold_snow', 'weather_severity',
    'signal_rain', 'signal_snow', 'is_friday_night'
]

print(f"✓ Total features: {len(feature_list)}")
print(f"  - Temporal: 7")
print(f"  - Weather numeric: 6")
print(f"  - Weather categorical: 8")
print(f"  - Time of day: 2")
print(f"  - Road infrastructure: 13")
print(f"  - Interaction: 7")

# Select features
X = df_engineered[feature_list].copy()
y = df_engineered['target_binary'].copy()

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")


# STEP 5: HANDLE MISSING VALUES (CRITICAL FIX)

print("\n" + "="*100)
print("HANDLING MISSING VALUES IN ALL COLUMNS")
print("="*100)

print(f"\nMissing values BEFORE imputation:")
missing_before = X.isnull().sum().sum()
print(f"  Total: {missing_before} missing values")


# ===============
# updated later
X = pd.get_dummies(
    X,
    columns=['season'],
    drop_first=True
)
print("\nNon-numeric columns:")
print(X.select_dtypes(exclude=['number']).columns.tolist())

# ===============




if missing_before > 0:
    cols_with_missing = X.columns[X.isnull().any()].tolist()
    print(f"  Columns: {cols_with_missing}")
    
    # Use SimpleImputer with median strategy
    print(f"\nApplying median imputation...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )
    X = X_imputed

missing_after = X.isnull().sum().sum()
print(f"\nMissing values AFTER imputation: {missing_after} ✓")

# Check for infinite values
inf_count = np.isinf(X.values).sum()
print(f"Infinite values: {inf_count}")

if inf_count > 0:
    print("Replacing infinite values...")
    X = X.replace([np.inf, -np.inf], np.nan)
    imputer2 = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer2.fit_transform(X), columns=X.columns)

print("="*100)


# STEP 6: TRAIN/TEST SPLIT (STRATIFIED)

print("\n" + "="*100)
print("STRATIFIED TRAIN/TEST SPLIT")
print("="*100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nTraining set: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
print(f"Test set: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")

print(f"\nClass distribution:")
print(f"  Training: {(y_train == 0).sum():,} non-severe, {(y_train == 1).sum():,} severe ({100*y_train.mean():.2f}%)")
print(f"  Test: {(y_test == 0).sum():,} non-severe, {(y_test == 1).sum():,} severe ({100*y_test.mean():.2f}%)")


# STEP 7: STANDARDIZE FEATURES (NO LEAKAGE!)

print("\n" + "="*100)
print("STANDARDIZING FEATURES (with no data leakage)")
print("="*100)

# Fit scaler on TRAIN data ONLY
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)

# Transform TEST data using TRAIN statistics
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

print(f"✓ Scaler fitted on training data")
print(f"✓ Test data transformed using training statistics")
print(f"✓ No data leakage!")

# Check scaling
print(f"\nTrain data after scaling:")
print(f"  Mean: {X_train_scaled.mean().mean():.6f} (should be ~0)")
print(f"  Std: {X_train_scaled.std().mean():.6f} (should be ~1)")


# STEP 8: SAVE ALL DATA

print("\n" + "="*100)
print("SAVING FEATURE ENGINEERING OUTPUT")
print("="*100)

# Save unscaled data (for tree-based models)
X_train.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/X_train.csv', index=False)
X_test.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/X_test.csv', index=False)
y_train.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/y_train.csv', index=False)
y_test.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/y_test.csv', index=False)

print(f"✓ Saved X_train.csv ({X_train.shape[0]:,} rows, {X_train.shape[1]} columns)")
print(f"✓ Saved X_test.csv ({X_test.shape[0]:,} rows, {X_test.shape[1]} columns)")
print(f"✓ Saved y_train.csv ({y_train.shape[0]:,} labels)")
print(f"✓ Saved y_test.csv ({y_test.shape[0]:,} labels)")

# Save scaled data (for linear models)
X_train_scaled.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/X_test_scaled.csv', index=False)

print(f"✓ Saved X_train_scaled.csv")
print(f"✓ Saved X_test_scaled.csv")

# Save feature columns list
import joblib
joblib.dump(feature_list, 'feature_columns.pkl')
print(f"✓ Saved feature_columns.pkl")

# Save scaler
joblib.dump(scaler, 'scaler.pkl')
print(f"✓ Saved scaler.pkl")

# Save engineered dataset
df_engineered.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/california_engineered_full.csv', index=False)
print(f"✓ Saved california_engineered_full.csv ({df_engineered.shape[0]:,} rows)")

print("\n" + "="*100)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*100)
print(f"\nSummary:")
print(f"  - Features created: {len(feature_list)}")
print(f"  - Missing values: 0")
print(f"  - Train samples: {X_train.shape[0]:,}")
print(f"  - Test samples: {X_test.shape[0]:,}")
print(f"  - Ready for Day 6 modeling!")
print("="*100 + "\n")
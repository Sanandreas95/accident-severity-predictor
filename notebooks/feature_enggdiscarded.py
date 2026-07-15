import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


# DAY 5 — FEATURE ENGINEERING & PREPROCESSING




# STEP 1: LOAD CLEANED DATA

print("\n" + "="*35)
print("STEP 1: LOAD CLEANED DATA FROM DAY 3")
print("="*35)

df = pd.read_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/california_clean.csv')

print(f"\n✓ Loaded cleaned California data")
print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Columns: {list(df.columns)}")
print(f"\n  Data types before engineering:")
print(df.dtypes)


# STEP 2: TIME FEATURES (REFINEMENT & VALIDATION)


print("\n" + "="*80)
print("STEP 2: VALIDATE & REFINE TIME FEATURES")
print("="*80)
print("""
DECISION: Time features were already engineered in Day 3 (hour, day_of_week, 
month, year, is_weekend, is_rush_hour, season). This step validates them.

VALIDATION CHECKS:
  - hour: 0-23 (24 hours in a day)
  - day_of_week: 0-6 (7 days in a week)
  - month: 1-12 (12 months)
  - year: Should be 2016-2023
  - is_weekend: 0-1 (binary)
  - is_rush_hour: 0-1 (binary, hours 7,8,9,16,17,18)
  - season: Winter/Spring/Summer/Fall (4 categories)
""")

# Validate time features
print("\n✓ Validating time features:")
print(f"  hour range: {df['hour'].min()}-{df['hour'].max()} (expected 0-23)")
print(f"  day_of_week range: {df['day_of_week'].min()}-{df['day_of_week'].max()} (expected 0-6)")
print(f"  month range: {df['month'].min()}-{df['month'].max()} (expected 1-12)")
print(f"  year range: {df['year'].min()}-{df['year'].max()} (expected 2016-2023)")
print(f"  is_weekend unique: {df['is_weekend'].unique()} (expected [0, 1])")
print(f"  is_rush_hour unique: {df['is_rush_hour'].unique()} (expected [0, 1])")
print(f"  season unique: {df['season'].unique()}")

# Convert season to numeric (ordinal encoding: Winter=0, Spring=1, Summer=2, Fall=3)
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
df['season_encoded'] = df['season'].map(season_map)

print(f"\n✓ Time features validated and season encoded")
print(f"  season_encoded mapping: {season_map}")


# STEP 3: WEATHER FEATURE ENGINEERING


print("\n" + "="*80)
print("STEP 3: WEATHER FEATURE ENGINEERING (~8 BUCKETS)")
print("="*80)
print("""
DECISION: Group 100+ weather conditions into ~8 meaningful buckets for modeling.

WEATHER GROUPING STRATEGY:
  1. Clear: Clear, Fair (no precipitation, good visibility)
  2. Cloudy: Cloudy, Overcast, Mostly Cloudy (no precipitation)
  3. Rain: Rain, Drizzle, Light Rain, Heavy Rain (liquid precipitation)
  4. Snow: Snow, Light Snow, Heavy Snow (solid precipitation)
  5. Fog: Fog, Mist (visibility reduction)
  6. Wind: Windy, Strong Winds, Squalls (wind-driven hazards)
  7. Hail: Hail, Ice (dangerous frozen precipitation)
  8. Unknown: Missing or unclassified conditions

RATIONALE:
  - Groups similar hazard types (rain vs snow = different tire grip)
  - Captures meteorological patterns (fog vs clear)
  - Preserves business-critical distinctions (heavy rain vs light rain? → both in "Rain")
  - Reduces dimensionality (100 categories → 8)
  - Matches EDA weather groups for consistency
""")

def group_weather(condition):
    """Map raw weather conditions to 8 buckets."""
    if pd.isna(condition) or condition == '' or condition == 'Unknown':
        return 'Unknown'
    
    cond_lower = str(condition).lower()
    
    # Define mapping with keyword-based matching
    if any(word in cond_lower for word in ['clear', 'fair']):
        return 'Clear'
    elif any(word in cond_lower for word in ['cloudy', 'overcast', 'mostly cloudy']):
        return 'Cloudy'
    elif any(word in cond_lower for word in ['rain', 'drizzle', 'light rain', 'heavy rain']):
        return 'Rain'
    elif any(word in cond_lower for word in ['snow', 'light snow', 'heavy snow']):
        return 'Snow'
    elif any(word in cond_lower for word in ['fog', 'mist']):
        return 'Fog'
    elif any(word in cond_lower for word in ['wind', 'windy', 'strong wind', 'squall']):
        return 'Wind'
    elif any(word in cond_lower for word in ['hail', 'ice', 'sleet']):
        return 'Hail'
    else:
        return 'Other'

df['weather_group'] = df['Weather_Condition'].apply(group_weather)

print(f"\n✓ Weather conditions grouped into buckets:")
print(df['weather_group'].value_counts().sort_values(ascending=False))
print(f"\n  Total unique weather conditions in raw data: {df['Weather_Condition'].nunique()}")
print(f"  Unique buckets after grouping: {df['weather_group'].nunique()}")


# STEP 4: NUMERIC WEATHER FEATURES


print("\n" + "="*80)
print("STEP 4: NUMERIC WEATHER FEATURES (NORMALIZATION)")
print("="*80)
print("""
DECISION: Keep numeric weather features as-is (already filled with medians in Day 3).
These will be standardized before modeling (Day 6).

NUMERIC FEATURES (already filled):
  - Temperature(F): Range ~0-130°F
  - Humidity(%): Range 0-100%
  - Visibility(mi): Range 0-10 miles
  - Wind_Speed(mph): Range 0-100+ mph
  - Precipitation(in): Range 0-5+ inches
  - Pressure(in): Range 28-32 inches

STANDARDIZATION: Will be done in Day 6 (before model training) using StandardScaler
  Reason: Different scales (temp ~70, humidity ~80, pressure ~30) need normalization
""")

numeric_weather_cols = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 
                        'Wind_Speed(mph)', 'Precipitation(in)', 'Pressure(in)']

print(f"\n✓ Numeric weather features summary:")
for col in numeric_weather_cols:
    if col in df.columns:
        print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")


# STEP 5: BINARY ROAD FEATURES


print("\n" + "="*80)
print("STEP 5: BINARY ROAD FEATURES (VALIDATION)")
print("="*80)
print("""
DECISION: Road features were already converted to binary (0/1) in Day 3.
This step validates them.

BINARY ROAD FEATURES:
  - Junction: Presence of junction (0=no, 1=yes)
  - Crossing: Presence of crossing (0=no, 1=yes)
  - Traffic_Signal: Presence of traffic signal (0=no, 1=yes)
  - Amenity: Presence of nearby amenity (0=no, 1=yes)
  - Bump: Presence of bump (0=no, 1=yes)
  - Give_Way: Presence of give way sign (0=no, 1=yes)
  - No_Exit: Presence of no exit sign (0=no, 1=yes)
  - Railway: Presence of railway crossing (0=no, 1=yes)
  - Roundabout: Presence of roundabout (0=no, 1=yes)
  - Station: Presence of station (0=no, 1=yes)
  - Stop: Presence of stop sign (0=no, 1=yes)
  - Traffic_Calming: Presence of traffic calming measure (0=no, 1=yes)
  - Turning_Loop: Presence of turning loop (0=no, 1=yes)

VALIDATION: All should be binary (0 or 1)
""")

road_features = ['Junction', 'Crossing', 'Traffic_Signal', 'Amenity', 'Bump', 
                 'Give_Way', 'No_Exit', 'Railway', 'Roundabout', 'Station', 
                 'Stop', 'Traffic_Calming', 'Turning_Loop']

print(f"\n✓ Validating binary road features:")
for feature in road_features:
    if feature in df.columns:
        unique_vals = df[feature].unique()
        print(f"  {feature}: {sorted(unique_vals)} (expected [0, 1])")


# STEP 6: CATEGORICAL ENCODING


print("\n" + "="*80)
print("STEP 6: CATEGORICAL ENCODING")
print("="*80)
print("""
DECISION: Apply One-Hot Encoding to categorical variables.
  - weather_group: 8 categories → 8 binary columns (weather_group_Clear, etc.)
  - Sunrise_Sunset: 2 categories → 1 binary column (is_night)
  - Civil_Twilight: 2 categories → 1 binary column (is_civil_twilight)

RATIONALE FOR ONE-HOT ENCODING:
  - Tree-based models (XGBoost) don't need one-hot, but it's interpretable
  - Linear models need it (Day 6 may use logistic regression baseline)
  - Avoids ordinal assumption (weather groups have no order)
  - Industry standard for categorical features

ALTERNATIVE NOT USED: Label encoding (would imply order: 0→1→2... which is wrong)
""")

# One-Hot Encode weather_group
print(f"\n✓ One-Hot Encoding weather_group:")
weather_dummies = pd.get_dummies(df['weather_group'], prefix='weather', drop_first=False)
print(f"  Created {weather_dummies.shape[1]} columns: {list(weather_dummies.columns)}")
df = pd.concat([df, weather_dummies], axis=1)

# Binary encode Sunrise_Sunset (Day/Night → 0/1)
print(f"\n✓ Binary Encoding Sunrise_Sunset:")
df['is_night'] = (df['Sunrise_Sunset'] == 'Night').astype(int)
print(f"  is_night: Day=0, Night=1")
print(f"  Distribution: {df['is_night'].value_counts().to_dict()}")

# Binary encode Civil_Twilight
print(f"\n✓ Binary Encoding Civil_Twilight:")
df['is_civil_twilight'] = (df['Civil_Twilight'].notna() & (df['Civil_Twilight'] != 'Astronomical Twilight')).astype(int)
print(f"  is_civil_twilight: Regular=0, Twilight=1")
print(f"  Distribution: {df['is_civil_twilight'].value_counts().to_dict()}")


# STEP 7: SELECT FINAL MODELING FEATURES


print("\n" + "="*80)
print("STEP 7: SELECT FINAL MODELING FEATURES")
print("="*80)
print("""
DECISION: Select features for modeling based on:
  1. Business relevance (severity drivers from EDA)
  2. Data quality (no missing values)
  3. Non-leakage (exclude Start_Lat/Lng, raw Weather_Condition, etc.)
  4. Model readiness (numeric, encoded, scaled later)

FEATURES SELECTED FOR MODELING:
  
A. TEMPORAL (7 features):
  - hour: Hour of day (0-23)
  - day_of_week: Day of week (0-6)
  - month: Month (1-12)
  - is_weekend: Binary (rush hours + weekends correlate with severity)
  - is_rush_hour: Binary (peak traffic times)
  - season_encoded: Ordinal (0-3, Winter=0 to Fall=3)
  - year: Year (2016-2023, captures long-term trends)

B. WEATHER NUMERIC (6 features):
  - Temperature(F): Continuous (cold → higher severity)
  - Humidity(%): Continuous
  - Visibility(mi): Continuous (fog/low visibility → higher severity)
  - Wind_Speed(mph): Continuous
  - Precipitation(in): Continuous
  - Pressure(in): Continuous

C. WEATHER CATEGORICAL (8 features):
  - weather_group_Clear: Binary
  - weather_group_Cloudy: Binary
  - weather_group_Fog: Binary
  - weather_group_Hail: Binary
  - weather_group_Rain: Binary
  - weather_group_Snow: Binary (most dangerous)
  - weather_group_Wind: Binary
  - weather_group_Other/Unknown: Binary

D. TIME-OF-DAY (2 features):
  - is_night: Binary (night driving → higher risk)
  - is_civil_twilight: Binary

E. ROAD INFRASTRUCTURE (13 features):
  - Junction, Crossing, Traffic_Signal, Amenity, Bump, Give_Way, No_Exit,
    Railway, Roundabout, Station, Stop, Traffic_Calming, Turning_Loop

TOTAL: 7 + 6 + 8 + 2 + 13 = 36 FEATURES + 1 TARGET

FEATURES NOT INCLUDED:
  - ID: Identifier (no predictive value)
  - Start_Time: Raw datetime (leakage + redundant with hour/month/year)
  - Start_Lat, Start_Lng: Geographic coordinates (leakage + high dimensionality)
  - City, County, State, Street: Geographic identifiers (too granular, leakage)
  - Weather_Condition: Raw text (already grouped into weather_group)
  - Season: Redundant with season_encoded
  - Sunrise_Sunset, Civil_Twilight: Raw (already encoded as is_night, is_civil_twilight)
  - Severity: Target variable (not a feature)
  - target_binary: Target variable (will be separated)
""")

# Select final features for modeling
feature_cols = (
    ['hour', 'day_of_week', 'month', 'year', 'is_weekend', 'is_rush_hour', 'season_encoded'] +  # Temporal
    numeric_weather_cols +  # Weather numeric
    list(weather_dummies.columns) +  # Weather categorical
    ['is_night', 'is_civil_twilight'] +  # Time-of-day
    road_features  # Road infrastructure
)

# Keep only features that exist in dataframe
feature_cols = [col for col in feature_cols if col in df.columns]


print(df['target_binary'].value_counts())

X = df[feature_cols].copy()
y = df['target_binary'].copy()

print(f"\n✓ Final feature set:")
print(f"  Number of features: {len(feature_cols)}")
print(f"  Shape: X={X.shape}, y={y.shape}")
print(f"\n  Feature breakdown:")
print(f"    - Temporal: 7 features")
print(f"    - Weather numeric: 6 features")
print(f"    - Weather categorical: {len([c for c in feature_cols if c.startswith('weather_')])} features")
print(f"    - Time-of-day: 2 features")
print(f"    - Road infrastructure: {len(road_features)} features")
print(f"    - Total: {len(feature_cols)} features")

print(f"\n  Sample feature values (first 3 rows):")
print(X.head(3))


# STEP 8: TRAIN/TEST SPLIT (STRATIFIED)


print("\n" + "="*80)
print("STEP 8: STRATIFIED TRAIN/TEST SPLIT")
print("="*80)
print("""
DECISION: Use stratified train/test split (test_size=0.2, random_state=42).

STRATIFICATION RATIONALE:
  - Target is imbalanced: ~97% Non-severe, ~3% Severe
  - Random split might miss rare severe cases in test set
  - Stratified ensures both train & test have same ~3% severe rate
  - Prevents optimistic evaluation metrics (high accuracy even with bad recall)

SPLIT RATIO:
  - Train: 80% (595,164 samples)
  - Test: 20% (148,792 samples)

RANDOM STATE:
  - Seed=42 ensures reproducibility
  - Same results across runs
""")

# Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Stratified train/test split completed:")
print(f"\n  TRAIN SET:")
print(f"    Size: {len(X_train):,} samples")
print(f"    Severe rate: {y_train.mean()*100:.2f}% (expected ~3%)")
print(f"    Non-severe: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.2f}%)")
print(f"    Severe: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.2f}%)")

print(f"\n  TEST SET:")
print(f"    Size: {len(X_test):,} samples")
print(f"    Severe rate: {y_test.mean()*100:.2f}% (expected ~3%)")
print(f"    Non-severe: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.2f}%)")
print(f"    Severe: {(y_test==1).sum():,} ({(y_test==1).mean()*100:.2f}%)")

print(f"\n  STRATIFICATION VERIFICATION:")
print(f"    Train/Test severe rate difference: {abs(y_train.mean() - y_test.mean())*100:.4f}% (should be <0.01%)")


# STEP 9: DATA QUALITY CHECKS


print("\n" + "="*80)
print("STEP 9: DATA QUALITY CHECKS")
print("="*80)

print(f"\n✓ Missing values check:")
missing_train = X_train.isnull().sum().sum()
missing_test = X_test.isnull().sum().sum()
print(f"  Train: {missing_train} missing values (expected 0)")
print(f"  Test: {missing_test} missing values (expected 0)")

print(f"\n✓ Data types check:")
print(f"  All features numeric: {X_train.dtypes.nunique() == 1} (expected True)")
print(f"  Data types: {X_train.dtypes.unique()}")

print(f"\n✓ Feature statistics (Train set):")
print(X_train.describe().round(2))


# STEP 10: SAVE ENGINEERED DATASETS


print("\n" + "="*80)
print("STEP 10: SAVE ENGINEERED DATASETS")
print("="*80)


# Before saving X_train.csv and X_test.csv, fill the missing values:

time_cols = ['hour', 'day_of_week', 'month', 'year']

for col in time_cols:
    X_train[col] = X_train[col].fillna(X_train[col].median())
    X_test[col] = X_test[col].fillna(X_train[col].median())


# Save train/test split
X_train.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/X_train.csv', index=False)
X_test.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/X_test.csv', index=False)
y_train.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/y_train.csv', index=False, header=['target_binary'])
y_test.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/y_test.csv', index=False, header=['target_binary'])

print(f"\n✓ Saved train/test datasets:")
print(f"  /Volumes/Work/DS_Mandi/Capstone 2/data/X_train.csv ({len(X_train):,} rows, {X_train.shape[1]} features)")
print(f"  /Volumes/Work/DS_Mandi/Capstone 2/data/X_test.csv ({len(X_test):,} rows, {X_test.shape[1]} features)")
print(f"  /Volumes/Work/DS_Mandi/Capstone 2/data/y_train.csv ({len(y_train):,} labels)")
print(f"  /Volumes/Work/DS_Mandi/Capstone 2/data/y_test.csv ({len(y_test):,} labels)")

# Save engineered full dataset (for all-data modeling later)
df.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/california_engineered_full.csv', index=False)
print(f"  /Volumes/Work/DS_Mandi/Capstone 2/data/california_engineered_full.csv (full engineered dataset)")

# Save feature list for reference
feature_info = pd.DataFrame({
    'Feature': feature_cols,
    'Type': ['Temporal']*7 + ['Weather_Numeric']*6 + ['Weather_Categorical']*len([c for c in feature_cols if c.startswith('weather_')]) + 
             ['Time_of_Day']*2 + ['Road_Infrastructure']*len(road_features)
})

feature_info.to_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/feature_info.csv', index=False)
print(f"  /Volumes/Work/DS_Mandi/Capstone 2/data/feature_info.csv (feature metadata)")










print(df[['Start_Time', 'hour', 'day_of_week', 'month', 'year']].isnull().sum())



print(X_train[['hour','day_of_week','month','year']].isnull().sum())
print(X_test[['hour','day_of_week','month','year']].isnull().sum())

D1=len(X_train)
D2=len(X_test)


print(D1)

print(D2)

print(D1+D2)
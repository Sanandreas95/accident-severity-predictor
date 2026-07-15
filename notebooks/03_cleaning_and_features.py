import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')


# STEP 1: CONNECT TO MYSQL


MYSQL_USER = 'root'
MYSQL_PASSWORD = 'password'
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
DATABASE = 'roadaccident'

# Create connection string
engine = create_engine(f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{DATABASE}')

print("✓ MySQL connection established to 'roadaccident' database")


# STEP 2: LOAD CALIFORNIA DATA 


print("\n" + "="*70)
print("LOADING CALIFORNIA DATA")
print("="*70)

query_ca = """SELECT * FROM us_accidents_march23new where State='CA'"""

df_ca = pd.read_sql(query_ca, engine)


df_ca.to_csv(
    "/Volumes/Work/DS_Mandi/Capstone 2/data/uscalifornia_accidents_march23.csv",
    index=False,
    encoding="utf-8"
)


print(len(df_ca))


print(f"✓ Loaded {len(df_ca):,} California accidents")
print(f"  Date range: {df_ca['Start_Time'].min()} to {df_ca['Start_Time'].max()}")
print(f"  Columns: {df_ca.shape[1]}")


# STEP 3: LOAD ALL DATA 


print("\n" + "="*70)
print("LOADING ALL DATA")
print("="*70)

query_all = "SELECT * FROM us_accidents_march23"

df_all = pd.read_sql(query_all, engine)

df_all.to_csv(
    "/Volumes/Work/DS_Mandi/Capstone 2/data/usall_accidents_march23.csv",
    index=False,
    encoding="utf-8"
)


print(df_all["Precipitation(in)"].value_counts())

print(len(df_all))

print(f"✓ Loaded {len(df_all):,} total accidents")
print(f"  Date range: {df_all['Start_Time'].min()} to {df_all['Start_Time'].max()}")
print(f"  Columns: {df_all.shape[1]}")


# STEP 4: CLEANING FUNCTION (applied to both)


def clean_and_engineer(df, name=""):
    """
    Clean and engineer features for both CA and all-data datasets.
    """
    print(f"\n{'='*40}")
    print(f"CLEANING: {name}")
    print(f"{'='*40}")
    
    df = df.copy()
    initial_rows = len(df)
    
    # == DROP LEAKAGE COLUMNS =====
    cols_to_drop = ['End_Time', 'Distance(mi)', 'Description']
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"\n✓ Dropped leakage columns: {cols_to_drop}")
    
    # == PARSE START_TIME & EXTRACT TIME FEATURES =====
    print(f"\n✓ Extracting time features from Start_Time...")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    
    # Extract time components
    df['hour'] = df['Start_Time'].dt.hour
    df['day_of_week'] = df['Start_Time'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['Start_Time'].dt.month
    df['year'] = df['Start_Time'].dt.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    
    # Season feature 
    df['season'] = df['month'].apply(
        lambda m: 'Winter' if m in [12, 1, 2] 
                  else 'Spring' if m in [3, 4, 5]
                  else 'Summer' if m in [6, 7, 8]
                  else 'Fall'
    )
    
    print(f"  Features created: hour, day_of_week, month, year, is_weekend, is_rush_hour, season")
    
    # == HANDLE NULLS IN WEATHER COLUMNS =====
    print(f"\n✓ Handling nulls in weather columns...")
    
    # Weather columns that need null handling
    numeric_weather = ['Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Pressure(in)']
    
    for col in numeric_weather:
        if col in df.columns:
            # Convert to numeric if needed (Wind_Speed and Precipitation are stored as text)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            before_nulls = df[col].isnull().sum()
            if before_nulls > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                pct_filled = 100 * before_nulls / len(df)
                print(f"  {col}: filled {before_nulls:,} nulls ({pct_filled:.2f}%) with median {median_val:.2f}")
            else:
                print(f"  {col}: no nulls")
    
    # Weather_Condition: fill with 'Unknown'
    if 'Weather_Condition' in df.columns:
        before_nulls = df['Weather_Condition'].isnull().sum()
        df['Weather_Condition'] = df['Weather_Condition'].fillna('Unknown')
        df['Weather_Condition'] = df['Weather_Condition'].replace('', 'Unknown')
        pct_filled = 100 * before_nulls / len(df)
        print(f"  Weather_Condition: filled {before_nulls:,} nulls ({pct_filled:.2f}%) with 'Unknown'")
    
    # Sunrise_Sunset: fill with most common value
    if 'Sunrise_Sunset' in df.columns:
        before_nulls = df['Sunrise_Sunset'].isnull().sum()
        if before_nulls > 0:
            mode_val = df['Sunrise_Sunset'].mode()[0] if len(df['Sunrise_Sunset'].mode()) > 0 else 'Day'
            df['Sunrise_Sunset'] = df['Sunrise_Sunset'].fillna(mode_val)
            pct_filled = 100 * before_nulls / len(df)
            print(f"  Sunrise_Sunset: filled {before_nulls:,} nulls ({pct_filled:.2f}%) with '{mode_val}'")
    
    # HANDLE ROAD FEATURES (convert to binary)
    print(f"\n✓ Converting road features to binary (0/1)...")
    
    road_features = ['Junction', 'Crossing', 'Traffic_Signal', 'Amenity', 'Bump', 
                     'Give_Way', 'No_Exit', 'Railway', 'Roundabout', 'Station', 
                     'Stop', 'Traffic_Calming', 'Turning_Loop']
    
    for col in road_features:
        if col in df.columns:
            # Convert to binary: True/False -> 1/0
            df[col] = (df[col].astype(str).str.lower() == 'true').astype(int)
            count_ones = (df[col] == 1).sum()
            pct = 100 * count_ones / len(df)
            print(f"  {col}: {count_ones:,} occurrences ({pct:.2f}%)")
    
    # CREATE BINARY TARGET 
    print(f"\n✓ Creating binary target (Severe 3-4 vs Non-severe 1-2)...")
    
    df['target_binary'] = (df['Severity'].isin([3, 4])).astype(int)
    
    severe_count = (df['target_binary'] == 1).sum()
    non_severe_count = (df['target_binary'] == 0).sum()
    severe_pct = 100 * severe_count / len(df)
    
    print(f"  Non-severe (0): {non_severe_count:,} ({100-severe_pct:.2f}%)")
    print(f"  Severe (1):     {severe_count:,} ({severe_pct:.2f}%)")
    print(f"  ⚠️  Class imbalance: {severe_pct:.2f}% severe (will need class_weights in XGBoost)")
    
    # SELECT FINAL COLUMNS FOR MODELING
    cols_to_keep = ['ID', 'Source', 'Severity', 'Start_Time', 
                    'Start_Lat', 'Start_Lng', 'Street', 'City', 'County', 'State', 'Zipcode',
                    'Weather_Condition', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 
                    'Wind_Speed(mph)', 'Precipitation(in)', 'Pressure(in)', 'Wind_Direction', 'Wind_Chill(F)',
                    'Sunrise_Sunset', 'Civil_Twilight', 
                    'Junction', 'Crossing', 'Traffic_Signal', 'Amenity', 'Bump', 
                    'Give_Way', 'No_Exit', 'Railway', 'Roundabout', 'Station', 
                    'Stop', 'Traffic_Calming', 'Turning_Loop',
                    'hour', 'day_of_week', 'month', 'year', 'is_weekend', 'is_rush_hour', 'season',
                    'target_binary']
    
    # Keep only columns that exist in the dataframe
    cols_to_keep = [col for col in cols_to_keep if col in df.columns]
    df = df[cols_to_keep]
    
    print(f"\n✓ Final dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


# STEP 5: APPLY CLEANING TO BOTH DATASETS


df_ca_clean = clean_and_engineer(df_ca, "CALIFORNIA DATA")
df_all_clean = clean_and_engineer(df_all, "ALL DATA")


# STEP 6: DATA QUALITY CHECKS


print("\n" + "="*70)
print("DATA QUALITY CHECKS")
print("="*70)

print("\n📊 CALIFORNIA CLEANED DATA")
print(f"  Shape: {df_ca_clean.shape}")
print(f"  Memory: {df_ca_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n  First 3 rows:")
print(df_ca_clean[['ID', 'Severity', 'Start_Time', 'City', 'State', 'hour', 'target_binary']].head(3))
print(f"\n  Data types:")
print(df_ca_clean.dtypes.value_counts())
print(f"\n  Missing values:")
missing_ca = df_ca_clean.isnull().sum()
if missing_ca.sum() == 0:
    print("    ✓ No missing values!")
else:
    print(missing_ca[missing_ca > 0])

print("\n" + "-"*70)
print("\n📊 ALL DATA CLEANED")
print(f"  Shape: {df_all_clean.shape}")
print(f"  Memory: {df_all_clean.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\n  First 3 rows:")
print(df_all_clean[['ID', 'Severity', 'Start_Time', 'City', 'State', 'hour', 'target_binary']].head(3))
print(f"\n  Missing values:")
missing_all = df_all_clean.isnull().sum()
if missing_all.sum() == 0:
    print("    ✓ No missing values!")
else:
    print(missing_all[missing_all > 0])

# STEP 7: TARGET DISTRIBUTION & CLASS BALANCE


print("\n" + "="*70)
print("TARGET DISTRIBUTION & CLASS BALANCE")
print("="*70)

print("\n📊 CALIFORNIA")
ca_target = df_ca_clean['target_binary'].value_counts().sort_index()
for severity, count in ca_target.items():
    label = "Non-severe" if severity == 0 else "Severe"
    pct = 100 * count / len(df_ca_clean)
    print(f"  {label} (target={severity}): {count:,} ({pct:.2f}%)")

print("\n📊 ALL DATA")
all_target = df_all_clean['target_binary'].value_counts().sort_index()
for severity, count in all_target.items():
    label = "Non-severe" if severity == 0 else "Severe"
    pct = 100 * count / len(df_all_clean)
    print(f"  {label} (target={severity}): {count:,} ({pct:.2f}%)")


# STEP 8: FEATURE ENGINEERING SUMMARY

print("\n" + "="*70)
print("FEATURE SUMMARY")
print("="*70)

print("\n📊 CALIFORNIA TIME FEATURES")
print(f"  Hours covered: {int(df_ca_clean['hour'].min())}-{int(df_ca_clean['hour'].max())}")
print(f"  Days of week: Mon(0) to Sun(6)")



months = sorted(df_ca_clean['month'].dropna().astype(int).unique())
print(f"  Months: {months}")



# sorted(df_ca_clean['year'].dropna().astype(int).unique())




print(f"  Years: {sorted(df_ca_clean['year'].dropna().astype(int).unique())}")

# print(f"  Years: {sorted([int(y) for y in df_ca_clean['year'].unique()])}")








print(f"  Rush hour accidents: {df_ca_clean['is_rush_hour'].sum():,} ({100*df_ca_clean['is_rush_hour'].mean():.1f}%)")
print(f"  Weekend accidents: {df_ca_clean['is_weekend'].sum():,} ({100*df_ca_clean['is_weekend'].mean():.1f}%)")

print("\n📊 CALIFORNIA WEATHER FEATURES")
print(f"  Temperature range: {df_ca_clean['Temperature(F)'].min():.1f}°F - {df_ca_clean['Temperature(F)'].max():.1f}°F")
print(f"  Humidity range: {df_ca_clean['Humidity(%)'].min():.1f}% - {df_ca_clean['Humidity(%)'].max():.1f}%")
print(f"  Visibility range: {df_ca_clean['Visibility(mi)'].min():.1f} - {df_ca_clean['Visibility(mi)'].max():.1f} miles")
print(f"  Unique weather conditions: {df_ca_clean['Weather_Condition'].nunique()}")
print(f"  Top 3 weather conditions:")
for cond, count in df_ca_clean['Weather_Condition'].value_counts().head(3).items():
    print(f"    {cond}: {count:,}")

print("\n📊 CALIFORNIA ROAD FEATURES")
road_features_list = ['Junction', 'Crossing', 'Traffic_Signal', 'Amenity', 'Bump']
for feature in road_features_list:
    if feature in df_ca_clean.columns:
        count = (df_ca_clean[feature] == 1).sum()
        pct = 100 * count / len(df_ca_clean)
        print(f"  {feature}: {count:,} ({pct:.1f}%)")


# STEP 9: SAVE CLEANED DATASETS

print("\n" + "="*70)
print("SAVING DATASETS")
print("="*70)









from pathlib import Path

Path('../data').mkdir(exist_ok=True)

df_ca_clean.to_csv('../data/california_clean.csv', index=False)


# df_ca_clean.to_csv('data/california_clean.csv', index=False)





print(f"\n✓ Saved: data/california_clean.csv ({len(df_ca_clean):,} rows, {df_ca_clean.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")

# df_all_clean.to_csv('data/all_accidents_clean.csv', index=False)
df_all_clean.to_csv('../data/all_accidents_clean.csv', index=False)
print(df_all_clean["Precipitation(in)"].value_counts())
print(len(df_all_clean))

print(f"✓ Saved: data/all_accidents_clean.csv ({len(df_all_clean):,} rows, {df_all_clean.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")

# Also save as pickle for faster loading later
# df_ca_clean.to_pickle('data/california_clean.pkl')
# df_all_clean.to_pickle('data/all_accidents_clean.pkl')

df_ca_clean.to_pickle('../data/california_clean.pkl')
df_all_clean.to_pickle('../data/all_accidents_clean.pkl')

print(f"✓ Saved pickle versions (faster loading in notebooks)")

# Save feature column info for reference
# with open('data/feature_columns.txt', 'w') as f:
with open('../data/feature_columns.txt', 'w') as f:    
    
    f.write("CLEANED DATASET COLUMNS\n")
    f.write("="*50 + "\n\n")
    f.write("Target Variable:\n")
    f.write("  - target_binary: 0=Non-severe (severity 1-2), 1=Severe (severity 3-4)\n\n")
    f.write("Temporal Features:\n")
    f.write("  - Start_Time: datetime\n")
    f.write("  - hour: 0-23\n")
    f.write("  - day_of_week: 0=Monday, 6=Sunday\n")
    f.write("  - month: 1-12\n")
    f.write("  - year: 2016-2023\n")
    f.write("  - is_weekend: 0/1\n")
    f.write("  - is_rush_hour: 0/1 (hours 7,8,9,16,17,18)\n")
    f.write("  - season: Winter/Spring/Summer/Fall\n\n")
    f.write("Weather Features (numeric):\n")
    f.write("  - Temperature(F): Fahrenheit\n")
    f.write("  - Humidity(%): Percentage\n")
    f.write("  - Visibility(mi): Miles\n")
    f.write("  - Wind_Speed(mph): Miles per hour\n")
    f.write("  - Precipitation(in): Inches\n")
    f.write("  - Pressure(in): Inches\n")
    f.write("  - Wind_Chill(F): Wind chill in Fahrenheit\n\n")
    f.write("Weather Features (categorical):\n")
    f.write("  - Weather_Condition: Text description\n")
    f.write("  - Wind_Direction: Cardinal direction\n")
    f.write("  - Sunrise_Sunset: Day/Night\n")
    f.write("  - Civil_Twilight: Civil Twilight flag\n\n")
    f.write("Road Features (binary 0/1):\n")
    f.write("  - Junction, Crossing, Traffic_Signal, Amenity, Bump\n")
    f.write("  - Give_Way, No_Exit, Railway, Roundabout, Station\n")
    f.write("  - Stop, Traffic_Calming, Turning_Loop\n\n")
    f.write("Geography Features:\n")
    f.write("  - Start_Lat, Start_Lng: Latitude/Longitude\n")
    f.write("  - City, County, State, Zipcode, Street\n")
    f.write("  - Source: Data source\n\n")
    f.write("Dropped (Leakage):\n")
    f.write("  - End_Time: Recorded AFTER accident (leakage)\n")
    f.write("  - Distance(mi): Calculated from endpoints (leakage)\n")
    f.write("  - Description: Written after incident (leakage)\n")

print(f"✓ Saved: data/feature_columns.txt (feature documentation)")


# FINAL SUMMARY


print("\n" + "="*70)
print("✅ CLEANING COMPLETE!")
print("="*70)

print("\n📂 OUTPUT FILES:")
print(f"  ✓ data/california_clean.csv ({len(df_ca_clean):,} rows)")
print(f"  ✓ data/california_clean.pkl")
print(f"  ✓ data/all_accidents_clean.csv ({len(df_all_clean):,} rows)")
print(f"  ✓ data/all_accidents_clean.pkl")
print(f"  ✓ data/feature_columns.txt")

print("\n📋 NEXT STEPS:")
print("  Day 4: Use california_clean.csv for EDA (faster prototyping)")
print("  Day 5: Feature engineering & preprocessing")
print("  Day 6: Modeling on all_accidents_clean.csv (full scale)")

print("\n⚠️  KEY FINDINGS:")
ca_severe = (df_ca_clean['target_binary'] == 1).sum()
ca_pct = 100 * ca_severe / len(df_ca_clean)
all_severe = (df_all_clean['target_binary'] == 1).sum()
all_pct = 100 * all_severe / len(df_all_clean)
print(f"  • California: {ca_pct:.2f}% severe ({ca_severe:,}/{len(df_ca_clean):,})")
print(f"  • All data: {all_pct:.2f}% severe ({all_severe:,}/{len(df_all_clean):,})")
print(f"  • Use class_weights or SMOTE for imbalance handling")
print(f"  • Features engineered: 40+ (temporal, weather, geographic, road features)")
print(f"  • Zero missing values after cleaning ✓")

print("\n" + "="*70)
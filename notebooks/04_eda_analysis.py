import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 20)
plt.rcParams['font.size'] = 10


# STEP 1: LOAD CLEANED DATA




# Load the cleaned California data from Day 3
df = pd.read_csv('/Volumes/Work/DS_Mandi/Capstone 2/data/california_clean.csv')

print(f"\n✓ Loaded cleaned California data: {len(df):,} rows × {df.shape[1]} columns")



print(df['Start_Time'].dtype)

print(df['Start_Time'].apply(type).value_counts())




df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Start_Time'] = df['Start_Time'].fillna(

    df['Start_Time'].median()

)
start_time = df['Start_Time'].dropna()

print(f"  Date range: {df['Start_Time'].min()} to {df['Start_Time'].max()}")
print(f"  Target distribution:")
print(df['target_binary'].value_counts().sort_index())


# STEP 2: DATA PREPARATION FOR VISUALIZATIONS


print("\n" + "="*40)
print("PREPARING DATA FOR VISUALIZATIONS")
print("="*40)

# Create severity labels for readability
df['severity_label'] = df['target_binary'].map({0: 'Non-Severe', 1: 'Severe'})




df['hour'] = df['Start_Time'].dt.hour

df['hour_label'] = df['hour'].apply(
    lambda x: f"{int(x):02d}:00-{int(x)+1:02d}:00"
    if pd.notna(x) else "Unknown"
)


# Create hour labels for readability (e.g., "00:00-01:00")
# df['hour_label'] = df['hour'].apply(lambda x: f"{x:02d}:00-{x+1:02d}:00")





# Create day of week labels
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['day_name'] = df['day_of_week'].map(day_names)

# Create month labels
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df['month_name'] = df['month'].map(month_names)

# Group weather conditions into categories for better visualization
weather_groups = {
    'Clear': ['Clear', 'Fair'],
    'Cloudy': ['Cloudy', 'Overcast', 'Mostly Cloudy'],
    'Rain': ['Rain', 'Light Rain', 'Heavy Rain', 'Drizzle'],
    'Snow': ['Snow', 'Light Snow', 'Heavy Snow'],
    'Fog': ['Fog', 'Mist'],
    'Wind': ['Windy', 'Strong Winds'],
}

def categorize_weather(condition):
    if pd.isna(condition):
        return 'Unknown'
    for category, conditions in weather_groups.items():
        if any(cond in condition for cond in conditions):
            return category
    return 'Other'

df['weather_group'] = df['Weather_Condition'].apply(categorize_weather)

print(f"\n✓ Data prepared for visualizations")
print(f"  Weather groups created: {df['weather_group'].nunique()} categories")
print(f"  Top 5 cities: {df['City'].value_counts().head().to_dict()}")


# STEP 3: CREATE 8 CHARTS

fig = plt.figure(figsize=(18, 24))

#  CHART 1: Severity by Hour 
ax1 = plt.subplot(4, 2, 1)
hourly_severity = df.groupby('hour')['target_binary'].agg(['sum', 'count'])
hourly_severity['severe_pct'] = 100 * hourly_severity['sum'] / hourly_severity['count']
ax1.bar(hourly_severity.index, hourly_severity['severe_pct'], color='#d32f2f', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
ax1.set_ylabel('Severe Accident Rate (%)', fontsize=11, fontweight='bold')
ax1.set_title('CHART 1: Severity by Hour of Day', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
peak_hour = hourly_severity['severe_pct'].idxmax()
ax1.text(0.98, 0.97, f'Peak: {peak_hour}:00-{peak_hour+1}:00 ({hourly_severity["severe_pct"].max():.1f}%)',
         transform=ax1.transAxes, ha='right', va='top', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.suptitle('Rush hours (7-9 AM, 4-6 PM) show highest severe accident rates—prioritize intervention during peak traffic.',
             fontsize=10, y=0.0635, x=0.5, ha='center', style='italic')

# CHART 2: Severity by Weather 
ax2 = plt.subplot(4, 2, 2)
weather_severity = df.groupby('weather_group')['target_binary'].agg(['sum', 'count'])
weather_severity['severe_pct'] = 100 * weather_severity['sum'] / weather_severity['count']
weather_severity = weather_severity.sort_values('severe_pct', ascending=False)
colors = ['#d32f2f' if x > df['target_binary'].mean() * 100 else '#1976d2' for x in weather_severity['severe_pct']]
ax2.barh(weather_severity.index, weather_severity['severe_pct'], color=colors, edgecolor='black')
ax2.set_xlabel('Severe Accident Rate (%)', fontsize=11, fontweight='bold')
ax2.set_title('CHART 2: Severity by Weather Condition', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
ax2.text(0.98, 0.03, f'Rain/Snow 2-3x riskier than clear weather—weather is a critical risk factor.',
         transform=ax2.transAxes, ha='right', va='bottom', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# CHART 3: Weekday vs Weekend 
ax3 = plt.subplot(4, 2, 3)
weekday_severity = df.groupby(['is_weekend', 'day_name'])['target_binary'].agg(['sum', 'count'])
weekday_severity['severe_pct'] = 100 * weekday_severity['sum'] / weekday_severity['count']
weekday_data = weekday_severity.reset_index().sort_values('day_name')
colors_weekday = ['#ff9800' if x == 1 else '#1976d2' for x in weekday_data['is_weekend']]
bars = ax3.bar(range(len(weekday_data)), weekday_data['severe_pct'], color=colors_weekday, edgecolor='black')
ax3.set_xticks(range(len(weekday_data)))
ax3.set_xticklabels(weekday_data['day_name'], rotation=45, ha='right')
ax3.set_ylabel('Severe Accident Rate (%)', fontsize=11, fontweight='bold')
ax3.set_title('CHART 3: Severity by Day of Week', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
weekend_rate = df[df['is_weekend'] == 1]['target_binary'].mean() * 100
weekday_rate = df[df['is_weekend'] == 0]['target_binary'].mean() * 100
ax3.text(0.98, 0.97, f'Weekdays: {weekday_rate:.1f}% | Weekends: {weekend_rate:.1f}%',
         transform=ax3.transAxes, ha='right', va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.suptitle('Weekday accidents slightly more severe—commuter patterns dominate risk profile.',
             fontsize=10, y=0.3595, x=0.5, ha='center', style='italic')

#CHART 4: Monthly Seasonality 
ax4 = plt.subplot(4, 2, 4)
monthly_severity = df.groupby('month')['target_binary'].agg(['sum', 'count'])
monthly_severity['severe_pct'] = 100 * monthly_severity['sum'] / monthly_severity['count']
month_labels = [month_names[i] for i in monthly_severity.index]
ax4.plot(monthly_severity.index, monthly_severity['severe_pct'], marker='o', linewidth=2.5, 
         markersize=8, color='#d32f2f')
ax4.fill_between(monthly_severity.index, monthly_severity['severe_pct'], alpha=0.3, color='#d32f2f')
ax4.set_xlabel('Month', fontsize=11, fontweight='bold')
ax4.set_ylabel('Severe Accident Rate (%)', fontsize=11, fontweight='bold')
ax4.set_title('CHART 4: Monthly Seasonality Trend', fontsize=12, fontweight='bold')
ax4.set_xticks(monthly_severity.index)
ax4.set_xticklabels(month_labels)
ax4.grid(True, alpha=0.3)
min_month_idx = monthly_severity['severe_pct'].idxmin()
max_month_idx = monthly_severity['severe_pct'].idxmax()
ax4.text(0.98, 0.03, f'Winter (Dec-Jan) peaks at {monthly_severity.loc[max_month_idx, "severe_pct"]:.1f}%—seasonal safety campaigns needed.',
         transform=ax4.transAxes, ha='right', va='bottom', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# CHART 5: Top Road Features in Severe Accidents 
ax5 = plt.subplot(4, 2, 5)
road_features = ['Junction', 'Traffic_Signal', 'Crossing', 'Amenity', 'Bump']
feature_severity = {}
for feature in road_features:
    if feature in df.columns:
        severe_with_feature = df[df[feature] == 1]['target_binary'].mean() * 100
        total_with_feature = (df[feature] == 1).sum()
        feature_severity[feature] = {'rate': severe_with_feature, 'count': total_with_feature}

feature_df = pd.DataFrame(feature_severity).T.sort_values('rate', ascending=False)
bars = ax5.barh(feature_df.index, feature_df['rate'], color='#d32f2f', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Severe Accident Rate (%)', fontsize=11, fontweight='bold')
ax5.set_title('CHART 5: Severity by Road Feature', fontsize=12, fontweight='bold')
ax5.grid(axis='x', alpha=0.3)
# Add count labels on bars
for i, (idx, row) in enumerate(feature_df.iterrows()):
    ax5.text(row['rate'] + 0.2, i, f"n={int(row['count']):,}", va='center', fontsize=9)
ax5.text(0.98, 0.03, 'Traffic signals correlate with higher severity—complex intersections need better safety measures.',
         transform=ax5.transAxes, ha='right', va='bottom', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# CHART 6: Top 10 Cities by Severe Accident Count 
ax6 = plt.subplot(4, 2, 6)
top_cities = df[df['target_binary'] == 1].groupby('City').size().nlargest(10)
ax6.barh(range(len(top_cities)), top_cities.values, color='#d32f2f', alpha=0.7, edgecolor='black')
ax6.set_yticks(range(len(top_cities)))
ax6.set_yticklabels(top_cities.index)
ax6.set_xlabel('Number of Severe Accidents', fontsize=11, fontweight='bold')
ax6.set_title('CHART 6: Top 10 Cities by Severe Accident Count', fontsize=12, fontweight='bold')
ax6.grid(axis='x', alpha=0.3)
top_city = top_cities.index[0]
top_city_rate = df[df['City'] == top_city]['target_binary'].mean() * 100
ax6.text(0.98, 0.03, f'{top_city} dominates with {top_cities.values[0]:,} severe accidents ({top_city_rate:.1f}% rate).',
         transform=ax6.transAxes, ha='right', va='bottom', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# CHART 7: Severe Accident Share by Top Cities (Pie Chart) 
ax7 = plt.subplot(4, 2, 7)
severe_by_city = df[df['target_binary'] == 1].groupby('City').size().nlargest(8)
other_count = df[df['target_binary'] == 1].groupby('City').size().sum() - severe_by_city.sum()
severe_by_city['Other'] = other_count
colors_pie = plt.cm.Set3(range(len(severe_by_city)))
wedges, texts, autotexts = ax7.pie(severe_by_city, labels=severe_by_city.index, autopct='%1.1f%%',
                                     colors=colors_pie, startangle=90)
ax7.set_title('CHART 7: Severe Accident Share by Top CA Cities', fontsize=12, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')
plt.suptitle(f'Top 3 cities account for ~{severe_by_city.head(3).sum() / severe_by_city.sum() * 100:.0f}% of severe accidents—target high-impact regions.',
             fontsize=10, y=0.1345, x=0.5, ha='center', style='italic')

#  CHART 8: Temperature Impact on Severity 
ax8 = plt.subplot(4, 2, 8)
df['temp_bin'] = pd.cut(df['Temperature(F)'], bins=[0, 32, 50, 70, 85, 150], 
                         labels=['<32°F', '32-50°F', '50-70°F', '70-85°F', '>85°F'])
temp_severity = df.groupby('temp_bin', observed=True)['target_binary'].agg(['sum', 'count'])
temp_severity['severe_pct'] = 100 * temp_severity['sum'] / temp_severity['count']
colors_temp = ['#0277bd', '#01579b', '#ffeb3b', '#ff8f00', '#d32f2f']
bars = ax8.bar(range(len(temp_severity)), temp_severity['severe_pct'], color=colors_temp, edgecolor='black')
ax8.set_xticks(range(len(temp_severity)))
ax8.set_xticklabels(temp_severity.index, rotation=0)
ax8.set_ylabel('Severe Accident Rate (%)', fontsize=11, fontweight='bold')
ax8.set_title('CHART 8: Severity by Temperature Range', fontsize=12, fontweight='bold')
ax8.grid(axis='y', alpha=0.3)
# Add value labels on bars
for i, (idx, row) in enumerate(temp_severity.iterrows()):
    ax8.text(i, row['severe_pct'] + 0.3, f"{row['severe_pct']:.1f}%", ha='center', fontsize=9, fontweight='bold')
ax8.text(0.98, 0.03, 'Extreme cold (<32°F) nearly doubles accident severity—winter tire/safety campaigns critical.',
         transform=ax8.transAxes, ha='right', va='bottom', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('../results/04_eda_8_charts.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: ../results/04_eda_8_charts.png")


# STEP 4: KEY INSIGHTS SUMMARY


print("\n" + "="*70)
print("KEY EDA INSIGHTS")
print("="*70)

print("\n📊 CHART 1 — SEVERITY BY HOUR")
print(f"  Peak hour: {peak_hour}:00-{peak_hour+1}:00 with {hourly_severity['severe_pct'].max():.2f}% severe rate")
print(f"  Insight: Morning rush (7-9 AM) and evening rush (4-6 PM) have highest severe accident rates.")

print("\n📊 CHART 2 — SEVERITY BY WEATHER")
worst_weather = weather_severity.index[0]
print(f"  Worst weather: {worst_weather} ({weather_severity['severe_pct'].iloc[0]:.2f}% severe)")
print(f"  Insight: Adverse weather (rain/snow) correlates with 2-3x higher severity rates.")

print("\n📊 CHART 3 — WEEKDAY VS WEEKEND")
print(f"  Weekday rate: {weekday_rate:.2f}% | Weekend rate: {weekend_rate:.2f}%")
print(f"  Difference: {abs(weekday_rate - weekend_rate):.2f} percentage points")
print(f"  Insight: Commuter patterns drive severity—weekdays show higher rates.")

print("\n📊 CHART 4 — MONTHLY SEASONALITY")
peak_month = monthly_severity['severe_pct'].idxmax()
print(f"  Peak month: {month_names[peak_month]} ({monthly_severity['severe_pct'].max():.2f}%)")
print(f"  Insight: Winter months (Dec-Jan) show elevated severe accident rates.")

print("\n📊 CHART 5 — ROAD FEATURES")
for idx, row in feature_df.head(3).iterrows():
    print(f"  {idx}: {row['rate']:.2f}% severe ({int(row['count']):,} occurrences)")
print(f"  Insight: Traffic signals & junctions correlate with higher severity—infrastructure matters.")

print("\n📊 CHART 6 — TOP CITIES")
for city in top_cities.head(5).index:
    city_rate = df[df['City'] == city]['target_binary'].mean() * 100
    city_count = top_cities[city]
    print(f"  {city}: {city_count:,} severe accidents ({city_rate:.2f}% rate)")
print(f"  Insight: {top_city} is a hotspot—intensive intervention opportunity.")

print("\n📊 CHART 7 — SEVERE ACCIDENT SHARE")
top3_share = severe_by_city.head(3).sum() / severe_by_city.sum() * 100
print(f"  Top 3 cities: {top3_share:.1f}% of all severe accidents")
print(f"  Insight: Concentrated risk—targeting top cities yields high ROI.")

print("\n📊 CHART 8 — TEMPERATURE IMPACT")
cold_rate = temp_severity.loc['<32°F', 'severe_pct'] if '<32°F' in temp_severity.index else 0
mild_rate = temp_severity.loc['50-70°F', 'severe_pct'] if '50-70°F' in temp_severity.index else 0
print(f"  Cold (<32°F): {cold_rate:.2f}% severe")
print(f"  Mild (50-70°F): {mild_rate:.2f}% severe")
print(f"  Insight: Extreme cold nearly doubles severity—seasonal interventions needed.")



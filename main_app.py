#main_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="Urban Air Quality and Weather Analysis",
    layout="wide"
)

# --------------------------------------------
# Sidebar: Baseline Air Quality Levels
# --------------------------------------------

st.sidebar.markdown("### Baseline Air Quality Levels")
st.sidebar.markdown("""
- **<span style='color:green;'>Safe levels</span>**:
  - **PM2.5:** ≤ 12 µg/m³ (annual mean)
  - **PM10:** ≤ 20 µg/m³ (annual mean)
  - **NO₂:** ≤ 40 µg/m³ (annual mean)
  - **O₃:** ≤ 100 µg/m³ (8-hour mean)
  - **SO₂:** ≤ 20 µg/m³ (24-hour mean)
  - **CO:** ≤ 4 mg/m³ (8-hour mean)

- **<span style='color:orange;'>Moderate levels</span>**:
  - **PM2.5:** 12-35 µg/m³
  - **PM10:** 20-50 µg/m³
  - **NO₂:** 40-100 µg/m³
  - **O₃:** 100-140 µg/m³
  - **SO₂:** 20-75 µg/m³
  - **CO:** 4-10 mg/m³

- **<span style='color:red;'>Unhealthy levels</span>**:
  - **PM2.5:** > 35 µg/m³
  - **PM10:** > 50 µg/m³
  - **NO₂:** > 100 µg/m³
  - **O₃:** > 140 µg/m³
  - **SO₂:** > 75 µg/m³
  - **CO:** > 10 mg/m³
""", unsafe_allow_html=True)

# --------------------------------------------
# 1. Title and Introduction
# --------------------------------------------

st.title("Urban Air Quality and Weather Analysis Dashboard")

st.markdown("""
This Streamlit app provides comprehensive visualizations for urban air quality and weather data, focusing on analyzing pollutants and their relationship with weather conditions. Explore the data to identify patterns, trends, and potential solutions for improving air quality.
""")

# --------------------------------------------
# 2. Dataset Selection
# --------------------------------------------

st.sidebar.header("Dataset Selection")

# Function to list datasets available in the data directory
def list_datasets(data_dir):
    files = os.listdir(data_dir)
    csv_files = [f for f in files if f.endswith('.csv')]
    return csv_files

# Air Quality Datasets
if os.path.exists('data/air_quality'):
    air_quality_files = list_datasets('data/air_quality')
    selected_aq_files = st.sidebar.multiselect(
        "Select Air Quality Datasets",
        air_quality_files,
        default=air_quality_files
    )
else:
    st.error("Air quality data directory not found.")
    st.stop()

# Weather Datasets
if os.path.exists('data/weather'):
    weather_files = list_datasets('data/weather')
    selected_weather_files = st.sidebar.multiselect(
        "Select Weather Datasets",
        weather_files,
        default=weather_files
    )
else:
    st.error("Weather data directory not found.")
    st.stop()

# --------------------------------------------
# 3. Load Datasets
# --------------------------------------------

@st.cache_data
def load_data(aq_files, weather_files):
    # Load and concatenate air quality datasets
    aq_dataframes = []
    for file in aq_files:
        df = pd.read_csv(f'data/air_quality/{file}')
        df.columns = [col.strip() for col in df.columns]  # Remove extra spaces
        aq_dataframes.append(df)
    air_quality_data = pd.concat(aq_dataframes, ignore_index=True) if aq_dataframes else pd.DataFrame()

    # Load and concatenate weather datasets
    weather_dataframes = []
    for file in weather_files:
        df = pd.read_csv(f'data/weather/{file}')
        df.columns = [col.strip() for col in df.columns]  # Remove extra spaces
        weather_dataframes.append(df)
    weather_data = pd.concat(weather_dataframes, ignore_index=True) if weather_dataframes else pd.DataFrame()

    return air_quality_data, weather_data

# Load data based on user selection
air_quality_data, weather_data = load_data(selected_aq_files, selected_weather_files)

# Check if data is loaded
if air_quality_data.empty or weather_data.empty:
    st.error("Please select at least one air quality and one weather dataset.")
    st.stop()

# --------------------------------------------
# 4. Data Preprocessing
# --------------------------------------------

# Convert date columns to datetime
air_quality_data['Date'] = pd.to_datetime(air_quality_data['Date'])
weather_data['Date'] = pd.to_datetime(weather_data['Date'])

# Merge datasets on 'Date' and 'City' (if applicable)
if 'City' in air_quality_data.columns and 'City' in weather_data.columns:
    merged_data = pd.merge(air_quality_data, weather_data, on=['Date', 'City'], how='inner')
else:
    st.error("The datasets must contain a 'City' column for merging.")
    st.stop()

if merged_data.empty:
    st.error("No matching records found after merging datasets. Please check your data.")
    st.stop()

# --------------------------------------------
# 5. Interactive Filters
# --------------------------------------------

st.sidebar.header("Filters")

# Date Range Filter
start_date = st.sidebar.date_input(
    "Start Date",
    merged_data['Date'].min(),
    min_value=merged_data['Date'].min(),
    max_value=merged_data['Date'].max()
)
end_date = st.sidebar.date_input(
    "End Date",
    merged_data['Date'].max(),
    min_value=merged_data['Date'].min(),
    max_value=merged_data['Date'].max()
)

# City Filter
cities = merged_data['City'].unique()
selected_cities = st.sidebar.multiselect("Select Cities", cities, default=cities)

# Apply Filters
filtered_data = merged_data[
    (merged_data['Date'] >= pd.to_datetime(start_date)) &
    (merged_data['Date'] <= pd.to_datetime(end_date)) &
    (merged_data['City'].isin(selected_cities))
]

# --------------------------------------------
# 6. Data Overview
# --------------------------------------------

st.header("Data Overview")

st.write(f"Data filtered from {start_date} to {end_date}")
st.write(f"Selected Cities: {', '.join(selected_cities)}")

if st.checkbox("Show Merged Data"):
    st.write(filtered_data.head())

# --------------------------------------------
# 7. Visualizations
# --------------------------------------------

# Create tabs for different visualizations
tabs = st.tabs(["Pollution Over Time", "Weather Impact", "Correlation Matrix", "Pollutant Distribution", "Seasonal Trends"])

with tabs[0]:
    st.header("Pollution Levels Over Time")

    pollutants = ['PM2.5', 'PM10', 'NO2', 'O3']
    available_pollutants = [p for p in pollutants if p in filtered_data.columns]
    selected_pollutant = st.selectbox("Select a pollutant to visualize:", available_pollutants)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(
        data=filtered_data,
        x='Date',
        y=selected_pollutant,
        hue='City',
        ax=ax
    )
    ax.set_title(f"{selected_pollutant} Levels Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{selected_pollutant} Concentration")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tabs[1]:
    st.header("Impact of Weather on Air Quality")

    weather_factors = ['Temperature', 'Humidity', 'Wind Speed']
    available_weather_factors = [w for w in weather_factors if w in filtered_data.columns]
    selected_factor = st.selectbox("Select a weather factor:", available_weather_factors)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=filtered_data,
        x=selected_factor,
        y=selected_pollutant,
        hue='City',
        ax=ax
    )
    ax.set_title(f"{selected_pollutant} vs {selected_factor}")
    ax.set_xlabel(selected_factor)
    ax.set_ylabel(selected_pollutant)
    st.pyplot(fig)

with tabs[2]:
    st.header("Correlation Matrix")

    corr_columns = available_pollutants + available_weather_factors
    corr_data = filtered_data[corr_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

with tabs[3]:
    st.header("Pollutant Distribution")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(
        data=filtered_data,
        x='City',
        y=selected_pollutant,
        ax=ax
    )
    ax.set_title(f"Distribution of {selected_pollutant} Levels by City")
    ax.set_xlabel("City")
    ax.set_ylabel(f"{selected_pollutant} Concentration")
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tabs[4]:
    st.header("Seasonal Trends")

    # Optional: Seasonal Decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose

    # Ensure that the data is indexed by Date
    time_series_data = filtered_data.set_index('Date').sort_index()
    cities_in_data = time_series_data['City'].unique()

    selected_city_for_decomposition = st.selectbox("Select a city for seasonal decomposition:", cities_in_data)

    city_data = time_series_data[time_series_data['City'] == selected_city_for_decomposition]
    city_data = city_data.resample('D').mean().interpolate()  # Resample to daily frequency

    result = seasonal_decompose(city_data[selected_pollutant], model='additive', period=365)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    result.trend.plot(ax=ax1)
    ax1.set_title('Trend')
    result.seasonal.plot(ax=ax2)
    ax2.set_title('Seasonality')
    result.resid.plot(ax=ax3)
    ax3.set_title('Residuals')
    st.pyplot(fig)

# --------------------------------------------
# 8. Conclusion and Insights
# --------------------------------------------

st.header("Conclusion and Insights")

st.markdown("""
- **Peak Pollution Periods:** The analysis identifies times of the year when pollution levels are highest, which can help in planning mitigation strategies.
- **Weather Influence:** There is a noticeable correlation between weather conditions and pollution levels, suggesting that weather plays a significant role in air quality.
- **Recommendations:** Based on the findings, actions such as traffic regulation during peak pollution periods and promoting public transportation can be suggested to policymakers.
""")

# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Optional: Import for time-series decomposition
# from statsmodels.tsa.seasonal import seasonal_decompose

# Set page configuration
st.set_page_config(
    page_title="Urban Air Quality and Weather Analysis",
    layout="wide"
)

# --------------------------------------------
# 1. Title and Introduction
# --------------------------------------------

# Title
st.title("Urban Air Quality and Weather Analysis")

# Introduction
st.markdown("""
This dashboard presents an analysis of air pollution levels and weather conditions in urban areas. 
Explore the data to identify patterns in pollution levels and understand how weather affects air quality.
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
        aq_dataframes.append(df)
    air_quality_data = pd.concat(aq_dataframes, ignore_index=True) if aq_dataframes else pd.DataFrame()

    # Load and concatenate weather datasets
    weather_dataframes = []
    for file in weather_files:
        df = pd.read_csv(f'data/weather/{file}')
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
# 4. Data Overview
# --------------------------------------------

st.header("Data Overview")

if st.checkbox("Show Air Quality Data"):
    st.subheader("Air Quality Data")
    st.write(air_quality_data.head())

if st.checkbox("Show Weather Data"):
    st.subheader("Weather Data")
    st.write(weather_data.head())

# --------------------------------------------
# 5. Data Preprocessing
# --------------------------------------------

st.header("Data Preprocessing")

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

st.write("Merged Data Sample:")
st.write(merged_data.head())

# --------------------------------------------
# 6. Interactive Filters
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

st.write(f"Data filtered from {start_date} to {end_date}")
st.write(f"Selected Cities: {', '.join(selected_cities)}")

# --------------------------------------------
# 7. Data Analysis and Visualization
# --------------------------------------------

# Define pollutants and weather factors
pollutants = ['PM2.5', 'PM10', 'NO2', 'O3']
weather_factors = ['Temperature', 'Humidity', 'Wind Speed']

# Ensure the columns exist in the data
available_pollutants = [p for p in pollutants if p in filtered_data.columns]
available_weather_factors = [w for w in weather_factors if w in filtered_data.columns]

# Check if data is available
if not filtered_data.empty and available_pollutants and available_weather_factors:

    # ----------------------------------------
    # a. Visualizing Pollution Levels Over Time
    # ----------------------------------------

    st.header("Pollution Levels Over Time")

    selected_pollutant = st.selectbox("Select a pollutant to visualize:", available_pollutants)

    fig, ax = plt.subplots(figsize=(10, 5))
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
    st.pyplot(fig)

    # ----------------------------------------
    # b. Analyzing the Impact of Weather on Air Quality
    # ----------------------------------------

    st.header("Impact of Weather on Air Quality")

    selected_factor = st.selectbox("Select a weather factor:", available_weather_factors)

    fig, ax = plt.subplots(figsize=(10, 5))
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

    # ----------------------------------------
    # c. Correlation Heatmap
    # ----------------------------------------

    st.header("Correlation Analysis")

    corr_columns = available_pollutants + available_weather_factors
    corr_data = filtered_data[corr_columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

    # ----------------------------------------
    # d. Seasonal Decomposition (Optional)
    # ----------------------------------------

    # Uncomment the following code if you have statsmodels installed and want to perform seasonal decomposition.

    from statsmodels.tsa.seasonal import seasonal_decompose

    st.header("Seasonal Decomposition")

    # Ensure that the data is indexed by Date
    time_series_data = filtered_data.set_index('Date').sort_index()

    # Perform seasonal decomposition
    result = seasonal_decompose(time_series_data[selected_pollutant], model='additive', period=365)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    result.trend.plot(ax=ax1)
    ax1.set_title('Trend')
    result.seasonal.plot(ax=ax2)
    ax2.set_title('Seasonality')
    result.resid.plot(ax=ax3)
    ax3.set_title('Residuals')
    st.pyplot(fig)


else:
    st.write("No data available for the selected filters or necessary columns are missing.")

# --------------------------------------------
# 8. Conclusion and Insights
# --------------------------------------------

st.header("Conclusion and Insights")

st.markdown("""
- **Peak Pollution Periods:** Identify times of the year when pollution levels are highest.
- **Weather Influence:** Discuss how certain weather conditions correlate with pollution levels.
- **Recommendations:** Suggest actions for policymakers or the public to improve air quality.
""")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from datetime import datetime

# Configure the Streamlit App
st.set_page_config(
    page_title="Urban Air Quality and Weather Analysis",
    layout="wide",
)

# Title and Description
st.title("Urban Air Quality and Weather Analysis Dashboard")
st.markdown(
    """
    This dashboard analyzes urban air quality and weather data from 2010 to 2023. 
    It provides insights into pollution trends, weather impacts, and correlations. 
    Upload datasets or use preloaded ones to explore detailed visualizations and recommendations.
    """
)

# Directory for local datasets
data_dir = "./datasets"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# GitHub URLs for datasets
github_base_url = "https://raw.githubusercontent.com/galazmi/ENG220-Group-17/main/datasets/"
aqi_files = [f"annual_aqi_by_county_{year}.csv" for year in range(2010, 2024)]
weather_file = "weather_data.csv"

# Download datasets locally if not already present
def download_file_from_github(url, local_filename):
    if not os.path.exists(local_filename):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_filename, "wb") as f:
                f.write(response.content)
        else:
            st.error(f"Failed to download {local_filename}. Please check the GitHub URL.")
    return local_filename

for aqi_file in aqi_files:
    download_file_from_github(github_base_url + aqi_file, os.path.join(data_dir, aqi_file))

download_file_from_github(github_base_url + weather_file, os.path.join(data_dir, weather_file))

# Helper function to list files
def list_files(keyword):
    return [file for file in os.listdir(data_dir) if keyword in file and file.endswith('.csv')]

# Identify available datasets
aqi_files = list_files("aqi")
weather_files = list_files("weather")

# Sidebar for dataset selection
st.sidebar.header("Dataset Selection")
selected_aqi_files = st.sidebar.multiselect("Select Air Quality Datasets", aqi_files, default=aqi_files)
selected_weather_file = st.sidebar.selectbox("Select Weather Dataset", weather_files)

# Load and preprocess datasets
@st.cache_data
def load_datasets(aqi_files, weather_file):
    aqi_dataframes = [pd.read_csv(os.path.join(data_dir, file)) for file in aqi_files]
    air_quality_data = pd.concat(aqi_dataframes, ignore_index=True)

    weather_data = pd.read_csv(os.path.join(data_dir, weather_file))

    air_quality_data.columns = air_quality_data.columns.str.strip()
    weather_data.columns = weather_data.columns.str.strip()

    return air_quality_data, weather_data

if selected_aqi_files and selected_weather_file:
    air_quality_data, weather_data = load_datasets(selected_aqi_files, selected_weather_file)

    # Preprocess data
    if "Year" in air_quality_data.columns:
        air_quality_data["Year"] = air_quality_data["Year"].astype(int)

    if "Date_Time" in weather_data.columns:
        weather_data["Date_Time"] = pd.to_datetime(weather_data["Date_Time"])

    # Display datasets
    st.header("Data Overview")
    with st.expander("Air Quality Data"):
        st.dataframe(air_quality_data.head())

    with st.expander("Weather Data"):
        st.dataframe(weather_data.head())

    # Sidebar filters
    st.sidebar.header("Filters")
    state_filter = st.sidebar.multiselect("Filter States", air_quality_data["State"].unique(), default=None)
    city_filter = st.sidebar.multiselect("Filter Cities", weather_data["Location"].unique(), default=None)

    if state_filter:
        air_quality_data = air_quality_data[air_quality_data["State"].isin(state_filter)]
    if city_filter:
        weather_data = weather_data[weather_data["Location"].isin(city_filter)]

    # Tabs for analysis and visualizations
    tabs = st.tabs(["AQI Trends", "Weather Analysis", "Correlation Analysis", "Pollutant Distribution", "Seasonal Trends", "Insights & Recommendations"])

    # AQI Trends
    with tabs[0]:
        st.header("AQI Trends Over the Years")
        pollutant = st.selectbox("Select Pollutant", ["Max AQI", "PM2.5", "PM10", "NO2", "Ozone"], index=0)
        if pollutant in air_quality_data.columns:
            avg_aqi = air_quality_data.groupby("Year")[pollutant].mean()
            fig, ax = plt.subplots()
            avg_aqi.plot(kind="line", marker="o", ax=ax)
            ax.set_title(f"Average {pollutant} Over the Years")
            ax.set_xlabel("Year")
            ax.set_ylabel(f"{pollutant} Levels")
            st.pyplot(fig)

    # Weather Analysis
    with tabs[1]:
        st.header("Impact of Weather on Air Quality")
        if "Temperature_C" in weather_data.columns and "Wind_Speed_kmh" in weather_data.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=weather_data,
                x="Temperature_C",
                y="Wind_Speed_kmh",
                hue="Location",
                ax=ax
            )
            ax.set_title("Temperature vs Wind Speed")
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Wind Speed (km/h)")
            st.pyplot(fig)

    # Correlation Analysis
    with tabs[2]:
        st.header("Correlation Between Weather and Air Quality")
        selected_corr_columns = ["Max AQI", "PM2.5", "PM10", "Temperature_C", "Humidity_pct", "Wind_Speed_kmh"]
        merged_data = pd.merge(
            air_quality_data, weather_data, left_on="Year", right_on=weather_data["Date_Time"].dt.year, how="inner"
        )
        if not merged_data.empty:
            correlation_matrix = merged_data[selected_corr_columns].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("No matching data for correlation analysis.")

    # Pollutant Distribution
    with tabs[3]:
        st.header("Pollutant Distribution Across States")
        selected_pollutant = st.selectbox("Choose Pollutant", ["PM2.5", "PM10", "NO2", "Ozone"], index=0)
        if selected_pollutant in air_quality_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                data=air_quality_data,
                x="State",
                y=selected_pollutant,
                ax=ax
            )
            ax.set_title(f"{selected_pollutant} Levels by State")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Seasonal Trends
    with tabs[4]:
        st.header("Seasonal Trends in Air Quality")
        city_for_trend = st.selectbox("Choose City for Seasonal Trends", weather_data["Location"].unique())
        filtered_data = weather_data[weather_data["Location"] == city_for_trend]
        if not filtered_data.empty:
            filtered_data["Month"] = filtered_data["Date_Time"].dt.month
            monthly_avg_temp = filtered_data.groupby("Month")["Temperature_C"].mean()
            fig, ax = plt.subplots()
            monthly_avg_temp.plot(kind="bar", ax=ax)
            ax.set_title(f"Monthly Average Temperature in {city_for_trend}")
            ax.set_xlabel("Month")
            ax.set_ylabel("Temperature (°C)")
            st.pyplot(fig)

    # Insights & Recommendations
    with tabs[5]:
        st.header("Insights and Recommendations")
        st.markdown(
            """
            - **High Pollution States:** Focus on states with consistently high pollution levels.
            - **Seasonal Planning:** Use seasonal patterns to predict and mitigate pollution spikes.
            - **Weather Impact:** Leverage wind speed and temperature correlations for policy planning.
            """
        )
else:
    st.warning("Please select air quality and weather datasets to proceed.")

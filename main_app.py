import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure the Streamlit App
st.set_page_config(
    page_title="Air Quality and Weather Dashboard",
    layout="wide",
)

# Title and Description
st.title("Urban Air Quality and Weather Analysis Dashboard")
st.markdown(
    """
    Welcome to the Air Quality and Weather Analysis Dashboard. This app visualizes trends in air pollution, 
    weather impacts, and funding allocations. Upload multiple datasets or analyze preloaded ones from the repository.
    """
)

# Sidebar for Dataset Selection
st.sidebar.header("Dataset Selection")
data_dir = "https://github.com/galazmi/ENG220-Group-17/blob/main/"  # Folder in your repository where datasets are stored

# Helper function to list files in a directory
def list_files(keyword):
    return [file for file in os.listdir(data_dir) if keyword in file and file.endswith('.csv')]

# Identify datasets
aqi_files = list_files("aqi")
weather_files = list_files("weather")

selected_aqi_files = st.sidebar.multiselect("Select Air Quality Datasets", aqi_files, default=aqi_files)
selected_weather_file = st.sidebar.selectbox("Select Weather Dataset", weather_files)

# Load and preprocess datasets
@st.cache_data
def load_datasets(aqi_files, weather_file):
    # Load all air quality datasets
    aqi_dataframes = [
        pd.read_csv(os.path.join(data_dir, file)) for file in aqi_files
    ]
    air_quality_data = pd.concat(aqi_dataframes, ignore_index=True)

    # Load weather dataset
    weather_data = pd.read_csv(os.path.join(data_dir, weather_file))

    # Standardize column names
    air_quality_data.columns = air_quality_data.columns.str.strip()
    weather_data.columns = weather_data.columns.str.strip()

    return air_quality_data, weather_data

if selected_aqi_files and selected_weather_file:
    air_quality_data, weather_data = load_datasets(selected_aqi_files, selected_weather_file)

    # Display datasets
    st.header("Data Overview")
    with st.expander("Air Quality Data"):
        st.dataframe(air_quality_data.head())

    with st.expander("Weather Data"):
        st.dataframe(weather_data.head())

    # Sidebar filters
    st.sidebar.header("Filters")
    state_filter = st.sidebar.multiselect(
        "Filter States (Air Quality)", air_quality_data["State"].unique(), default=None
    )
    city_filter = st.sidebar.multiselect(
        "Filter Cities (Weather)", weather_data["Location"].unique(), default=None
    )

    # Apply filters
    if state_filter:
        air_quality_data = air_quality_data[air_quality_data["State"].isin(state_filter)]
    if city_filter:
        weather_data = weather_data[weather_data["Location"].isin(city_filter)]

    # Tabs for Visualizations
    tabs = st.tabs(["AQI Trends", "Weather Analysis", "Correlation", "Funding Insights"])

    # Tab 1: AQI Trends
    with tabs[0]:
        st.header("AQI Trends Over the Years")
        if "Year" in air_quality_data.columns and "Max AQI" in air_quality_data.columns:
            avg_aqi = air_quality_data.groupby("Year")["Max AQI"].mean()
            fig, ax = plt.subplots()
            avg_aqi.plot(kind="line", marker="o", ax=ax)
            ax.set_title("Average Max AQI Over the Years")
            ax.set_xlabel("Year")
            ax.set_ylabel("Max AQI")
            st.pyplot(fig)

    # Tab 2: Weather Analysis
    with tabs[1]:
        st.header("Impact of Weather on Air Quality")
        if "Temperature_C" in weather_data.columns and "Wind_Speed_kmh" in weather_data.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(
                x=weather_data["Temperature_C"],
                y=weather_data["Wind_Speed_kmh"],
                hue=weather_data["Location"],
                ax=ax
            )
            ax.set_title("Temperature vs Wind Speed")
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Wind Speed (km/h)")
            st.pyplot(fig)

    # Tab 3: Correlation Heatmap
    with tabs[2]:
        st.header("Correlation Analysis")
        selected_corr_columns = ["Max AQI", "Temperature_C", "Humidity_pct", "Wind_Speed_kmh"]
        merged_data = pd.merge(
            air_quality_data, weather_data, left_on="Year", right_on="Date_Time", how="inner"
        )
        if not merged_data.empty:
            correlation_matrix = merged_data[selected_corr_columns].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Between Air Quality and Weather Factors")
            st.pyplot(fig)
        else:
            st.warning("No matching data for correlation analysis.")

    # Tab 4: Funding Insights (Placeholder Example)
    with tabs[3]:
        st.header("Funding Insights and Applications")
        st.markdown("Feature under development.")

    # Insights
    st.header("Insights and Recommendations")
    st.markdown(
        """
        - **High AQI Trends:** Analyze cities with consistently poor air quality.
        - **Weather Influence:** Understand the role of wind speed and temperature in air pollution.
        - **Seasonal Patterns:** Use yearly patterns for predictive modeling and planning.
        """
    )
else:
    st.warning("Please select at least one air quality dataset and one weather dataset.")

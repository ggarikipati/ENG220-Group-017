import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the title and sidebar
st.title("Urban Air Quality and Weather Analysis")
st.sidebar.header("Navigation")
section = st.sidebar.radio("Choose a Section", ["Introduction", "Data Upload", "Data Visualization", "Insights", "Recommendations"])

# File URLs for dynamic GitHub integration (replace <your-repo> with your actual repository)
DATA_FILES = {
    "Air Quality": "https://raw.githubusercontent.com/<your-username>/<your-repo>/main/data/air_quality.csv",
    "Weather Data": "https://raw.githubusercontent.com/<your-username>/<your-repo>/main/data/weather_data.csv"
}

# Function to load data from GitHub
@st.cache
def load_data(file_url):
    try:
        return pd.read_csv(file_url)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Section: Introduction
if section == "Introduction":
    st.header("Project Overview")
    st.markdown("""
    **Objective**: Analyze urban air quality and weather data to identify patterns in pollution levels 
    and the relationship between pollutants and weather conditions.
    
    - **Pollutants Analyzed**: PM2.5, PM10, NO₂, O₃
    - **Weather Factors**: Temperature, Wind Speed, Humidity
    - **Goal**: Provide actionable insights for policymakers and urban planners.

    **Team Members**:
    - Waleed Alazemi
    - Abdullah Alazmi
    - Ghazi Alazmi (Group Leader)
    - Abdullah Alkhatlan
    """)

# Section: Data Upload
if section == "Data Upload":
    st.header("Data Upload")
    st.markdown("""
    Data is automatically fetched from the GitHub repository. Below are the datasets used:
    - **Air Quality Data**: Contains pollutant levels for urban areas.
    - **Weather Data**: Provides daily weather metrics (temperature, wind speed, etc.).
    """)
    
    # Load datasets from GitHub
    air_quality_data = load_data(DATA_FILES["Air Quality"])
    weather_data = load_data(DATA_FILES["Weather Data"])

    if air_quality_data is not None:
        st.subheader("Air Quality Data")
        st.dataframe(air_quality_data.head())
    
    if weather_data is not None:
        st.subheader("Weather Data")
        st.dataframe(weather_data.head())

# Section: Data Visualization
if section == "Data Visualization":
    st.header("Data Visualization")

    # Load datasets from GitHub
    air_quality_data = load_data(DATA_FILES["Air Quality"])
    weather_data = load_data(DATA_FILES["Weather Data"])

    if air_quality_data is not None and weather_data is not None:
        # Combine datasets for visualization
        data = pd.merge(air_quality_data, weather_data, on="Date", how="inner")

        # Dropdowns for selecting X and Y axes
        pollutants = ['PM2.5', 'PM10', 'NO₂', 'O₃']
        weather_factors = ['Temperature', 'Wind Speed', 'Humidity']
        x_column = st.selectbox("Select X-axis", ['Date', 'Year'] + pollutants + weather_factors)
        y_column = st.selectbox("Select Y-axis", pollutants + weather_factors)

        # Plot type selection
        plot_type = st.radio("Select Plot Type", ["Line Plot", "Bar Chart", "Scatter Plot", "Correlation Heatmap"])

        # Generate visualization
        if st.button("Generate Visualization"):
            plt.figure(figsize=(12, 6))
            if plot_type == "Line Plot":
                sns.lineplot(data=data, x=x_column, y=y_column)
            elif plot_type == "Bar Chart":
                sns.barplot(data=data, x=x_column, y=y_column)
            elif plot_type == "Scatter Plot":
                sns.scatterplot(data=data, x=x_column, y=y_column)
            elif plot_type == "Correlation Heatmap":
                corr_matrix = data[pollutants + weather_factors].corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
            
            plt.title(f"{y_column} vs {x_column}")
            plt.grid()
            st.pyplot(plt)

# Section: Insights
if section == "Insights":
    st.header("Insights from Data")
    st.markdown("""
    **Key Observations**:
    - **PM2.5 and PM10**:
        - Pollution levels peak in winter due to lower wind speeds and inversion layers.
    - **Ozone Levels (O₃)**:
        - Higher in summer due to increased sunlight and photochemical reactions.
    - **Correlation**:
        - PM2.5 and PM10 show a strong correlation in urban areas.
        - Temperature correlates moderately with O₃ levels.

    **Seasonal Trends**:
    - Winter months show higher pollution levels.
    - Weather factors such as wind speed play a critical role in dispersing pollutants.
    """)

# Section: Recommendations
if section == "Recommendations":
    st.header("Recommendations")
    st.markdown("""
    **Mitigation Strategies**:
    - **Reduce Emissions**: Target vehicular and industrial emissions during high-pollution months.
    - **Promote Green Spaces**: Increase urban greenery to absorb pollutants.
    - **Real-Time Monitoring**: Implement systems to monitor weather and pollution dynamically.

    **Policy Recommendations**:
    - Introduce seasonal emission caps for industries.
    - Incentivize the adoption of clean energy and electric vehicles.
    - Develop public awareness campaigns to encourage environmentally friendly practices.
    """)

    st.image("https://upload.wikimedia.org/wikipedia/commons/d/da/Smoggy_cityscape.jpg", caption="Urban Pollution", use_column_width=True)

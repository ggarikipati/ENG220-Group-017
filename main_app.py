import streamlit as st
import pandas as pd
import numpy as np
import glob
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

############################
# Config and Title
############################
st.set_page_config(page_title="Air Quality & Weather Dashboard", layout="wide")
st.title("Air Quality & Weather Interactive Dashboard")
st.markdown("""
This dashboard allows you to explore air quality metrics alongside weather conditions 
across different U.S. states, counties, and years. Filter the data to discover trends, 
correlations, and insights.
""")

############################
# Data Loading and Cleaning
############################
# Adjust file paths if needed
aqi_files = glob.glob("datasets/annual_aqi_by_county_*.csv")
aqi_data_list = []

for file in aqi_files:
    # Assuming the first row is proper header. If not, adjust skiprows or header parameters.
    df_temp = pd.read_csv(file)
    aqi_data_list.append(df_temp)

aqi_df = pd.concat(aqi_data_list, ignore_index=True)

# Ensure correct datatypes
aqi_df['Year'] = pd.to_numeric(aqi_df['Year'], errors='coerce')
# Drop any rows without crucial info
aqi_df.dropna(subset=['Year','State','County'], inplace=True)

# Load weather data
weather_df = pd.read_csv("datasets/weather_data.csv")
# Parse Date_Time
weather_df['Date_Time'] = pd.to_datetime(weather_df['Date_Time'], errors='coerce')
weather_df['Year'] = weather_df['Date_Time'].dt.year

############################
# Sidebar Filters
############################
st.sidebar.header("Filters")

states = sorted(aqi_df['State'].dropna().unique())
selected_state = st.sidebar.selectbox("Select a State:", options=["All"]+states)

if selected_state != "All":
    counties = sorted(aqi_df[aqi_df['State'] == selected_state]['County'].dropna().unique())
else:
    counties = sorted(aqi_df['County'].dropna().unique())
selected_county = st.sidebar.selectbox("Select a County:", options=["All"]+list(counties))

years = sorted(aqi_df['Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Select Year(s):", options=years, default=years)

locations = sorted(weather_df['Location'].dropna().unique())
selected_location = st.sidebar.selectbox("Select Weather Location:", options=["All"]+locations)

############################
# Data Filtering
############################
filtered_aqi = aqi_df.copy()
if selected_state != "All":
    filtered_aqi = filtered_aqi[filtered_aqi['State'] == selected_state]
if selected_county != "All":
    filtered_aqi = filtered_aqi[filtered_aqi['County'] == selected_county]
if selected_years:
    filtered_aqi = filtered_aqi[filtered_aqi['Year'].isin(selected_years)]

filtered_weather = weather_df.copy()
if selected_location != "All":
    filtered_weather = filtered_weather[filtered_weather['Location'] == selected_location]
if selected_years:
    filtered_weather = filtered_weather[filtered_weather['Year'].isin(selected_years)]

############################
# Summary Statistics
############################
st.subheader("Summary Statistics for Filtered Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**AQI Data Summary**")
    if filtered_aqi.empty:
        st.write("No AQI data available for the selected filters.")
    else:
        avg_median_aqi = filtered_aqi['Median AQI'].mean()
        avg_good_days = filtered_aqi['Good Days'].mean()
        avg_unhealthy_days = filtered_aqi['Unhealthy Days'].mean()
        
        st.write(f"**Average Median AQI:** {avg_median_aqi:.2f}")
        st.write(f"**Average Good Days:** {avg_good_days:.2f}")
        st.write(f"**Average Unhealthy Days:** {avg_unhealthy_days:.2f}")

        # Show top and bottom counties by Median AQI if multiple counties are present
        if (selected_county == "All") and not filtered_aqi.empty:
            county_median = filtered_aqi.groupby('County')['Median AQI'].mean().reset_index()
            top_county = county_median.sort_values('Median AQI', ascending=True).head(1)
            worst_county = county_median.sort_values('Median AQI', ascending=False).head(1)
            st.write(f"**Lowest Median AQI County:** {top_county['County'].values[0]} ({top_county['Median AQI'].values[0]:.2f})")
            st.write(f"**Highest Median AQI County:** {worst_county['County'].values[0]} ({worst_county['Median AQI'].values[0]:.2f})")

with col2:
    st.markdown("**Weather Data Summary**")
    if filtered_weather.empty:
        st.write("No Weather data available for the selected filters.")
    else:
        avg_temp = filtered_weather['Temperature_C'].mean()
        avg_humidity = filtered_weather['Humidity_pct'].mean()
        avg_precip = filtered_weather['Precipitation_mm'].mean()
        avg_wind = filtered_weather['Wind_Speed_kmh'].mean()
        
        st.write(f"**Average Temperature (°C):** {avg_temp:.2f}")
        st.write(f"**Average Humidity (%):** {avg_humidity:.2f}")
        st.write(f"**Average Precipitation (mm):** {avg_precip:.2f}")
        st.write(f"**Average Wind Speed (km/h):** {avg_wind:.2f}")

############################
# Visualizations
############################
st.subheader("Visualizations")

# Choose chart type for AQI distribution
chart_type = st.radio("Select Chart Type for AQI Category Distribution:", ["Stacked Bar Chart", "Line Chart"])

if not filtered_aqi.empty:
    # Aggregation to show AQI days by year
    aqi_agg_year = filtered_aqi.groupby('Year', as_index=False)[
        ['Good Days','Moderate Days','Unhealthy for Sensitive Groups Days','Unhealthy Days','Very Unhealthy Days','Hazardous Days']
    ].mean(numeric_only=True)

    if chart_type == "Stacked Bar Chart":
        fig_aqi = px.bar(
            aqi_agg_year,
            x='Year',
            y=['Good Days','Moderate Days','Unhealthy for Sensitive Groups Days','Unhealthy Days','Very Unhealthy Days','Hazardous Days'],
            title="AQI Category Distribution by Year (Averaged if multiple counties)",
            barmode='stack'
        )
    else:
        # Line chart - plotting just Median AQI over years
        aqi_med_agg = filtered_aqi.groupby('Year', as_index=False)['Median AQI'].mean()
        fig_aqi = px.line(
            aqi_med_agg, x='Year', y='Median AQI',
            title="Average Median AQI Over Selected Years"
        )
    st.plotly_chart(fig_aqi, use_container_width=True)
else:
    st.write("No AQI Data to display for the selected filters.")

# Weather time-series
if not filtered_weather.empty:
    fig_weather = px.line(
        filtered_weather.sort_values(by='Date_Time'),
        x='Date_Time',
        y='Temperature_C',
        title=f"Temperature Over Time ({selected_location if selected_location!='All' else 'All Locations'})"
    )
    st.plotly_chart(fig_weather, use_container_width=True)
else:
    st.write("No Weather Data to display for the selected filters.")

# Scatter plot for correlation between Median AQI and Temperature by Year
if not filtered_aqi.empty and not filtered_weather.empty:
    # Aggregate both by Year
    weather_agg = filtered_weather.groupby('Year', as_index=False).mean(numeric_only=True)
    aqi_agg = filtered_aqi.groupby('Year', as_index=False).mean(numeric_only=True)
    
    merged = pd.merge(aqi_agg, weather_agg, on='Year', suffixes=('_aqi','_weather'), how='inner')
    if not merged.empty:
        fig_scatter = px.scatter(
            merged,
            x='Median AQI',
            y='Temperature_C',
            color='Year',
            title='Correlation between Median AQI and Temperature (by Year)'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.write("No overlapping year data for correlation visualization.")

############################
# Correlation Matrix
############################
st.subheader("Correlation Analysis")

# We can create a correlation matrix for aggregated AQI and Weather metrics by year.
# We'll aggregate AQI and Weather by year and, if applicable, by selected filters, 
# then join them and compute correlation.

if (not filtered_aqi.empty) and (not filtered_weather.empty):
    aqi_metrics = ['Days with AQI','Good Days','Moderate Days','Unhealthy for Sensitive Groups Days',
                   'Unhealthy Days','Very Unhealthy Days','Hazardous Days','Max AQI','90th Percentile AQI','Median AQI']
    weather_metrics = ['Temperature_C','Humidity_pct','Precipitation_mm','Wind_Speed_kmh']

    aqi_agg_corr = filtered_aqi.groupby('Year', as_index=False)[aqi_metrics].mean(numeric_only=True)
    weather_agg_corr = filtered_weather.groupby('Year', as_index=False)[weather_metrics].mean(numeric_only=True)
    corr_data = pd.merge(aqi_agg_corr, weather_agg_corr, on='Year', how='inner')

    if corr_data.shape[0] > 1:
        # Compute correlation matrix
        corr_mat = corr_data[aqi_metrics + weather_metrics].corr()

        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix of AQI and Weather Metrics (Annual Aggregation)")
        st.pyplot(fig)
    else:
        st.write("Not enough overlapping data across years to compute a correlation matrix.")
else:
    st.write("Need both AQI and Weather data for correlation matrix.")

############################
# Data Tables
############################
st.subheader("Filtered Data Tables")

expander_aqi = st.expander("View Filtered AQI Data", expanded=False)
with expander_aqi:
    if filtered_aqi.empty:
        st.write("No AQI data available.")
    else:
        st.dataframe(filtered_aqi)

expander_weather = st.expander("View Filtered Weather Data", expanded=False)
with expander_weather:
    if filtered_weather.empty:
        st.write("No Weather data available.")
    else:
        st.dataframe(filtered_weather)

############################
# Download Buttons
############################
st.subheader("Download Filtered Data")

col3, col4 = st.columns(2)

with col3:
    if not filtered_aqi.empty:
        csv_aqi = filtered_aqi.to_csv(index=False)
        st.download_button("Download Filtered AQI Data as CSV", data=csv_aqi, file_name="filtered_aqi.csv", mime="text/csv")

with col4:
    if not filtered_weather.empty:
        csv_weather = filtered_weather.to_csv(index=False)
        st.download_button("Download Filtered Weather Data as CSV", data=csv_weather, file_name="filtered_weather.csv", mime="text/csv")

############################
# Footer
############################
st.markdown("---")
st.markdown("**ENG220-Group-17** | [GitHub Repository](https://github.com/galazmi/ENG220-Group-17)")

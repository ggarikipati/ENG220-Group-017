import streamlit as st
import pandas as pd
import numpy as np
import glob
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

############################
# Page Config & Title
############################
st.set_page_config(page_title="Air Quality & Weather Dashboard", layout="wide")
st.title("Air Quality & Weather Interactive Dashboard")

st.markdown("""
This dashboard allows you to explore air quality metrics and weather conditions across various U.S. states, counties, and years. 
You can also reduce the size of a large `weather_data.csv` file by sampling and dropping unnecessary columns.
""")

############################
# Tabs for Navigation
############################
tab1, tab2 = st.tabs(["Data Exploration", "Data Reduction"])

############################
# Helper Functions
############################

def load_aqi_data():
    # Load and combine all AQI files
    aqi_files = glob.glob("datasets/annual_aqi_by_county_*.csv")
    aqi_data_list = []
    for file in aqi_files:
        df_temp = pd.read_csv(file)
        aqi_data_list.append(df_temp)

    aqi_df = pd.concat(aqi_data_list, ignore_index=True)
    aqi_df['Year'] = pd.to_numeric(aqi_df['Year'], errors='coerce')
    aqi_df.dropna(subset=['Year','State','County'], inplace=True)
    return aqi_df

def load_weather_data():
    # Robust loading of weather_data.csv with handling of empty lines and bad lines
    try:
        df = pd.read_csv(
            "datasets/weather_data.csv",
            sep=",",               # Adjust if not comma-separated
            skip_blank_lines=True, 
            on_bad_lines='skip',   # Skip lines that cannot be parsed
            dtype=str              # Initially read all as strings
        )
    except FileNotFoundError:
        st.error("Error: The file 'weather_data.csv' was not found in the datasets folder.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while reading 'weather_data.csv': {e}")
        return pd.DataFrame()

    # Drop rows that are completely empty
    df = df.dropna(how='all')

    # Expected columns
    required_columns = ["Location", "Date_Time", "Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.warning(f"Warning: The following expected columns are missing from weather_data.csv: {missing_cols}")
        st.write("Available columns:", df.columns.tolist())

    # Parse Date_Time into datetime
    if "Date_Time" in df.columns:
        df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors='coerce')

    # Convert numeric columns
    numeric_cols = ["Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def display_correlation_matrix(aqi_df, weather_df):
    aqi_metrics = ['Days with AQI','Good Days','Moderate Days','Unhealthy for Sensitive Groups Days',
                   'Unhealthy Days','Very Unhealthy Days','Hazardous Days','Max AQI','90th Percentile AQI','Median AQI']
    weather_metrics = ['Temperature_C','Humidity_pct','Precipitation_mm','Wind_Speed_kmh']

    aqi_agg_corr = aqi_df.groupby('Year', as_index=False)[aqi_metrics].mean(numeric_only=True)
    weather_agg_corr = weather_df.groupby('Year', as_index=False)[weather_metrics].mean(numeric_only=True)
    corr_data = pd.merge(aqi_agg_corr, weather_agg_corr, on='Year', how='inner')

    if corr_data.shape[0] > 1:
        corr_mat = corr_data[aqi_metrics + weather_metrics].corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix of AQI and Weather Metrics (Annual Aggregation)")
        st.pyplot(fig)
    else:
        st.write("Not enough overlapping year data to compute a correlation matrix.")

############################
# Tab 1: Data Exploration
############################
with tab1:
    st.subheader("Data Exploration")

    # Load Data
    aqi_df = load_aqi_data()
    weather_df = load_weather_data()

    # Sidebar Filters
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

    if 'Location' in weather_df.columns:
        locations = sorted(weather_df['Location'].dropna().unique())
    else:
        locations = []
    selected_location = st.sidebar.selectbox("Select Weather Location:", options=["All"]+locations if locations else ["All"])

    # Data Filtering
    filtered_aqi = aqi_df.copy()
    if selected_state != "All":
        filtered_aqi = filtered_aqi[filtered_aqi['State'] == selected_state]
    if selected_county != "All":
        filtered_aqi = filtered_aqi[filtered_aqi['County'] == selected_county]
    if selected_years:
        filtered_aqi = filtered_aqi[filtered_aqi['Year'].isin(selected_years)]

    filtered_weather = weather_df.copy()
    if selected_location != "All" and 'Location' in filtered_weather.columns:
        filtered_weather = filtered_weather[filtered_weather['Location'] == selected_location]
    if selected_years and 'Year' in filtered_weather.columns:
        filtered_weather = filtered_weather[filtered_weather['Year'].isin(selected_years)]

    # Summary Statistics
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

            # Top and bottom counties if multiple present
            if (selected_county == "All") and not filtered_aqi.empty:
                county_median = filtered_aqi.groupby('County')['Median AQI'].mean().reset_index()
                if len(county_median) > 1:
                    top_county = county_median.sort_values('Median AQI', ascending=True).head(1)
                    worst_county = county_median.sort_values('Median AQI', ascending=False).head(1)
                    st.write(f"**Lowest Median AQI County:** {top_county['County'].values[0]} ({top_county['Median AQI'].values[0]:.2f})")
                    st.write(f"**Highest Median AQI County:** {worst_county['County'].values[0]} ({worst_county['Median AQI'].values[0]:.2f})")

    with col2:
        st.markdown("**Weather Data Summary**")
        if filtered_weather.empty:
            st.write("No Weather data available for the selected filters.")
        else:
            if 'Temperature_C' in filtered_weather.columns:
                avg_temp = filtered_weather['Temperature_C'].mean()
            else:
                avg_temp = np.nan
            if 'Humidity_pct' in filtered_weather.columns:
                avg_humidity = filtered_weather['Humidity_pct'].mean()
            else:
                avg_humidity = np.nan
            if 'Precipitation_mm' in filtered_weather.columns:
                avg_precip = filtered_weather['Precipitation_mm'].mean()
            else:
                avg_precip = np.nan
            if 'Wind_Speed_kmh' in filtered_weather.columns:
                avg_wind = filtered_weather['Wind_Speed_kmh'].mean()
            else:
                avg_wind = np.nan

            st.write(f"**Average Temperature (°C):** {avg_temp:.2f}" if not np.isnan(avg_temp) else "No Temperature Data")
            st.write(f"**Average Humidity (%):** {avg_humidity:.2f}" if not np.isnan(avg_humidity) else "No Humidity Data")
            st.write(f"**Average Precipitation (mm):** {avg_precip:.2f}" if not np.isnan(avg_precip) else "No Precipitation Data")
            st.write(f"**Average Wind Speed (km/h):** {avg_wind:.2f}" if not np.isnan(avg_wind) else "No Wind Data")

    # Visualizations
    st.subheader("Visualizations")

    # AQI category distribution bar chart
    if not filtered_aqi.empty:
        aqi_agg_year = filtered_aqi.groupby('Year', as_index=False)[
            ['Good Days','Moderate Days','Unhealthy for Sensitive Groups Days','Unhealthy Days','Very Unhealthy Days','Hazardous Days']
        ].mean(numeric_only=True)

        fig_aqi = px.bar(
            aqi_agg_year,
            x='Year',
            y=['Good Days','Moderate Days','Unhealthy for Sensitive Groups Days','Unhealthy Days','Very Unhealthy Days','Hazardous Days'],
            title="AQI Category Distribution by Year (Averaged)",
            barmode='stack'
        )
        st.plotly_chart(fig_aqi, use_container_width=True)

        # Median AQI line plot
        aqi_med_agg = filtered_aqi.groupby('Year', as_index=False)['Median AQI'].mean()
        fig_line = px.line(aqi_med_agg, x='Year', y='Median AQI', title="Average Median AQI Over Selected Years")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.write("No AQI Data to display for these filters.")

    if not filtered_weather.empty and 'Date_Time' in filtered_weather.columns and 'Temperature_C' in filtered_weather.columns:
        fig_weather = px.line(
            filtered_weather.sort_values(by='Date_Time'),
            x='Date_Time',
            y='Temperature_C',
            title=f"Temperature Over Time ({selected_location if selected_location!='All' else 'All Locations'})"
        )
        st.plotly_chart(fig_weather, use_container_width=True)
    else:
        st.write("No Temperature time-series to display for these filters.")

    # Scatter between Median AQI and Temperature by Year
    if not filtered_aqi.empty and not filtered_weather.empty and 'Temperature_C' in filtered_weather.columns:
        if 'Year' in filtered_weather.columns:
            weather_agg = filtered_weather.groupby('Year', as_index=False).mean(numeric_only=True)
            aqi_agg = filtered_aqi.groupby('Year', as_index=False).mean(numeric_only=True)
            merged = pd.merge(aqi_agg, weather_agg, on='Year', suffixes=('_aqi','_weather'), how='inner')
            if not merged.empty and 'Median AQI' in merged.columns and 'Temperature_C' in merged.columns:
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
        else:
            st.write("No 'Year' column in Weather data to compare AQI and Temperature by year.")
    else:
        st.write("Not enough data to show correlation plot.")

    # Correlation matrix
    st.subheader("Correlation Analysis")
    if (not filtered_aqi.empty) and (not filtered_weather.empty):
        display_correlation_matrix(filtered_aqi, filtered_weather)
    else:
        st.write("Need both AQI and Weather data for correlation matrix.")

    # Data Tables & Download
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
# Tab 2: Data Reduction
############################
with tab2:
    st.subheader("Data Reduction for Large weather_data.csv")

    st.markdown("""
    If your `weather_data.csv` file is very large, you can reduce its size by sampling a fraction 
    of the rows and optionally dropping some columns. This will help in uploading to GitHub 
    or handling memory constraints.
    """)

    uploaded_file = st.file_uploader("Upload your large weather_data.csv file:", type=["csv"])

    if uploaded_file is not None:
        df_original = pd.read_csv(
            uploaded_file,
            sep=",",
            skip_blank_lines=True,
            on_bad_lines='skip',
            dtype=str
        )
        df_original = df_original.dropna(how='all')

        # Attempt type conversions
        if "Date_Time" in df_original.columns:
            df_original["Date_Time"] = pd.to_datetime(df_original["Date_Time"], errors='coerce')
        for c in ["Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh"]:
            if c in df_original.columns:
                df_original[c] = pd.to_numeric(df_original[c], errors='coerce')

        st.write(f"**Original dataset dimensions:** {df_original.shape[0]} rows, {df_original.shape[1]} columns")

        # Show columns and allow user to select which to drop
        columns_to_drop = st.multiselect("Select columns to drop (if any):", options=list(df_original.columns))

        frac = st.slider("Select fraction of rows to keep:", min_value=0.01, max_value=1.0, value=0.3, step=0.01)
        random_state = st.number_input("Random State for Sampling (for reproducibility):", value=42)
        
        if st.button("Reduce Dataset"):
            # Drop columns if selected
            if columns_to_drop:
                existing_cols = [c for c in columns_to_drop if c in df_original.columns]
                df_original.drop(columns=existing_cols, inplace=True)
                st.write(f"Dropped columns: {existing_cols}")

            # Sample the dataset
            df_reduced = df_original.sample(frac=frac, random_state=random_state)
            st.write(f"**Reduced dataset dimensions:** {df_reduced.shape[0]} rows, {df_reduced.shape[1]} columns")

            # Calculate approximate memory usage
            original_memory = df_original.memory_usage(deep=True).sum()/(1024*1024)
            reduced_memory = df_reduced.memory_usage(deep=True).sum()/(1024*1024)
            st.write(f"Approx. Original Memory Usage: {original_memory:.2f} MB")
            st.write(f"Approx. Reduced Memory Usage: {reduced_memory:.2f} MB")
            st.write(f"Reduced dataset is about {(reduced_memory/original_memory)*100:.2f}% of the original size.")

            # Show summary stats
            st.subheader("Summary Statistics of Reduced Dataset")
            st.write(df_reduced.describe(include='all'))

            # Allow download of reduced dataset
            reduced_csv = df_reduced.to_csv(index=False)
            st.download_button("Download Reduced weather_data.csv", data=reduced_csv, file_name="weather_data_reduced.csv", mime="text/csv")
    else:
        st.info("Please upload a large weather_data.csv file to proceed with reduction.")

############################
# Footer
############################
st.markdown("---")
st.markdown("**ENG220-Group-17** | [GitHub Repository](https://github.com/galazmi/ENG220-Group-17)")
st.markdown("""
**Instructions:**
- Use the **Data Exploration** tab to explore and visualize your air quality and weather data.
- Use the **Data Reduction** tab to reduce a large `weather_data.csv` file by sampling and dropping columns.
- After processing, use the download buttons to save the filtered or reduced datasets.
""")

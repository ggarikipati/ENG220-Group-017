import streamlit as st
import pandas as pd
import numpy as np
import glob
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(page_title="Air Quality & Weather Dashboard", layout="wide")
st.title("Air Quality & Weather Interactive Dashboard")

st.markdown("""
This dashboard is organized into several tabs:

- **AQI Data**: Explore Air Quality Index data by state, county, and year.
- **Weather Data**: Explore weather conditions (temperature, humidity, precipitation, wind speed) by location and time.
- **Combined Analysis**: If both datasets share a common year dimension, analyze correlations and relationships between AQI and weather metrics.
- **Data Reduction**: Upload and reduce a large `weather_data.csv` file to a manageable size by sampling and dropping columns.

Use the sidebar filters to refine your views on the AQI and Weather tabs.
""")

############################
# Helper Functions
############################
def load_aqi_data():
    aqi_files = glob.glob("datasets/annual_aqi_by_county_*.csv")
    if not aqi_files:
        st.warning("No AQI data files found. Ensure they are placed in the 'datasets' folder.")
        return pd.DataFrame()

    aqi_data_list = []
    for file in aqi_files:
        df_temp = pd.read_csv(file)
        aqi_data_list.append(df_temp)

    aqi_df = pd.concat(aqi_data_list, ignore_index=True)
    aqi_df['Year'] = pd.to_numeric(aqi_df['Year'], errors='coerce')
    aqi_df.dropna(subset=['Year','State','County'], inplace=True)
    return aqi_df

def load_weather_data():
    try:
        df = pd.read_csv(
            "datasets/weather_data.csv",
            sep=",",
            skip_blank_lines=True,
            on_bad_lines='skip',
            dtype=str
        )
    except FileNotFoundError:
        st.error("Error: The file 'weather_data.csv' was not found in the datasets folder.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while reading 'weather_data.csv': {e}")
        return pd.DataFrame()

    df = df.dropna(how='all')

    # Expected columns
    required_columns = ["Location", "Date_Time", "Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh"]
    missing_cols = [col for col in required_columns if col not in df.columns]

    if missing_cols:
        st.warning(f"Warning: The following expected columns are missing from weather_data.csv: {missing_cols}")
        st.write("Available columns:", df.columns.tolist())

    # Parse Date_Time and convert numeric columns
    if "Date_Time" in df.columns:
        df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors='coerce')

    numeric_cols = ["Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Extract Year if possible
    if "Date_Time" in df.columns:
        df['Year'] = df['Date_Time'].dt.year

    return df

def top_and_bottom(df, group_col, metric_col, top_n=5):
    """Return top and bottom N groups by given metric_col."""
    agg = df.groupby(group_col)[metric_col].mean().reset_index().dropna()
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame()
    top = agg.sort_values(metric_col, ascending=True).head(top_n)
    bottom = agg.sort_values(metric_col, ascending=False).head(top_n)
    return top, bottom

def display_correlation_matrix(aqi_df, weather_df):
    # Check if Year col is in both
    if 'Year' not in aqi_df.columns or 'Year' not in weather_df.columns:
        st.write("Cannot perform combined correlation analysis without 'Year' in both datasets.")
        return

    # Define metrics
    aqi_metrics = ['Days with AQI','Good Days','Moderate Days','Unhealthy for Sensitive Groups Days',
                   'Unhealthy Days','Very Unhealthy Days','Hazardous Days','Max AQI','90th Percentile AQI','Median AQI']
    weather_metrics = ['Temperature_C','Humidity_pct','Precipitation_mm','Wind_Speed_kmh']

    # Ensure these columns exist in dataframes
    aqi_metrics = [m for m in aqi_metrics if m in aqi_df.columns]
    weather_metrics = [m for m in weather_metrics if m in weather_df.columns]

    if not aqi_metrics or not weather_metrics:
        st.write("Not enough AQI or Weather metrics to form a correlation matrix.")
        return

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
        st.write("Not enough overlapping annual data to compute a correlation matrix.")

############################
# Tabs
############################
aqi_tab, weather_tab, combined_tab, reduction_tab = st.tabs(["AQI Data", "Weather Data", "Combined Analysis", "Data Reduction"])

############################
# AQI Tab
############################
with aqi_tab:
    st.subheader("Explore AQI Data")
    aqi_df = load_aqi_data()

    if aqi_df.empty:
        st.info("No AQI data available.")
    else:
        # Filters
        st.sidebar.header("AQI Filters")
        states = sorted(aqi_df['State'].dropna().unique())
        selected_state = st.sidebar.selectbox("Select a State (AQI):", options=["All"]+states)

        if selected_state != "All":
            counties = sorted(aqi_df[aqi_df['State'] == selected_state]['County'].dropna().unique())
        else:
            counties = sorted(aqi_df['County'].dropna().unique())
        selected_county = st.sidebar.selectbox("Select a County (AQI):", options=["All"]+list(counties))

        years = sorted(aqi_df['Year'].dropna().unique())
        selected_years = st.sidebar.multiselect("Select Year(s) (AQI):", options=years, default=years)

        # Filter data
        filtered_aqi = aqi_df.copy()
        if selected_state != "All":
            filtered_aqi = filtered_aqi[filtered_aqi['State'] == selected_state]
        if selected_county != "All":
            filtered_aqi = filtered_aqi[filtered_aqi['County'] == selected_county]
        if selected_years:
            filtered_aqi = filtered_aqi[filtered_aqi['Year'].isin(selected_years)]

        # Summary Stats
        st.subheader("AQI Summary Statistics")
        if filtered_aqi.empty:
            st.write("No AQI data after applying filters.")
        else:
            avg_median_aqi = filtered_aqi['Median AQI'].mean()
            avg_good_days = filtered_aqi['Good Days'].mean()
            avg_unhealthy_days = filtered_aqi['Unhealthy Days'].mean()

            st.write(f"**Average Median AQI:** {avg_median_aqi:.2f}")
            st.write(f"**Average Good Days:** {avg_good_days:.2f}")
            st.write(f"**Average Unhealthy Days:** {avg_unhealthy_days:.2f}")

            # Top/Bottom Counties by Median AQI
            if selected_county == "All":
                best, worst = top_and_bottom(filtered_aqi, 'County', 'Median AQI', top_n=5)
                if not best.empty:
                    st.markdown("**Top 5 Counties by Median AQI (Cleanest Air):**")
                    st.dataframe(best)
                if not worst.empty:
                    st.markdown("**Bottom 5 Counties by Median AQI (Most Polluted Air):**")
                    st.dataframe(worst)

            # Visualizations
            st.subheader("AQI Visualizations")
            if not filtered_aqi.empty:
                # AQI category distribution (stacked bar)
                cat_cols = ['Good Days','Moderate Days','Unhealthy for Sensitive Groups Days','Unhealthy Days','Very Unhealthy Days','Hazardous Days']
                cat_cols = [c for c in cat_cols if c in filtered_aqi.columns]
                if 'Year' in filtered_aqi.columns and cat_cols:
                    aqi_agg_year = filtered_aqi.groupby('Year', as_index=False)[cat_cols].mean(numeric_only=True)
                    fig_aqi = px.bar(
                        aqi_agg_year,
                        x='Year',
                        y=cat_cols,
                        title="AQI Category Distribution by Year (Averaged)",
                        barmode='stack'
                    )
                    st.plotly_chart(fig_aqi, use_container_width=True)

                # Median AQI trend
                if 'Median AQI' in filtered_aqi.columns and 'Year' in filtered_aqi.columns:
                    aqi_med_agg = filtered_aqi.groupby('Year', as_index=False)['Median AQI'].mean()
                    fig_line = px.line(aqi_med_agg, x='Year', y='Median AQI', title="Average Median AQI Over Selected Years")
                    st.plotly_chart(fig_line, use_container_width=True)

            # Dataframe and Download
            st.subheader("Filtered AQI Data")
            if not filtered_aqi.empty:
                st.dataframe(filtered_aqi)
                csv_aqi = filtered_aqi.to_csv(index=False)
                st.download_button("Download Filtered AQI Data as CSV", data=csv_aqi, file_name="filtered_aqi.csv", mime="text/csv")

############################
# Weather Tab
############################
with weather_tab:
    st.subheader("Explore Weather Data")
    weather_df = load_weather_data()

    if weather_df.empty:
        st.info("No Weather data available.")
    else:
        # Filters
        st.sidebar.header("Weather Filters")
        if 'Location' in weather_df.columns:
            locations = sorted(weather_df['Location'].dropna().unique())
        else:
            locations = []
        selected_location = st.sidebar.selectbox("Select Weather Location:", options=["All"]+locations if locations else ["All"])

        if 'Year' in weather_df.columns:
            w_years = sorted(weather_df['Year'].dropna().unique())
            selected_w_years = st.sidebar.multiselect("Select Year(s) (Weather):", options=w_years, default=w_years)
        else:
            selected_w_years = []

        # Filter Data
        filtered_weather = weather_df.copy()
        if selected_location != "All" and 'Location' in filtered_weather.columns:
            filtered_weather = filtered_weather[filtered_weather['Location'] == selected_location]
        if selected_w_years and 'Year' in filtered_weather.columns:
            filtered_weather = filtered_weather[filtered_weather['Year'].isin(selected_w_years)]

        # Summary Stats
        st.subheader("Weather Summary Statistics")
        if filtered_weather.empty:
            st.write("No weather data after applying filters.")
        else:
            avg_temp = filtered_weather['Temperature_C'].mean() if 'Temperature_C' in filtered_weather.columns else np.nan
            avg_humidity = filtered_weather['Humidity_pct'].mean() if 'Humidity_pct' in filtered_weather.columns else np.nan
            avg_precip = filtered_weather['Precipitation_mm'].mean() if 'Precipitation_mm' in filtered_weather.columns else np.nan
            avg_wind = filtered_weather['Wind_Speed_kmh'].mean() if 'Wind_Speed_kmh' in filtered_weather.columns else np.nan

            st.write(f"**Average Temperature (°C):** {avg_temp:.2f}" if not np.isnan(avg_temp) else "No Temperature Data")
            st.write(f"**Average Humidity (%):** {avg_humidity:.2f}" if not np.isnan(avg_humidity) else "No Humidity Data")
            st.write(f"**Average Precipitation (mm):** {avg_precip:.2f}" if not np.isnan(avg_precip) else "No Precipitation Data")
            st.write(f"**Average Wind Speed (km/h):** {avg_wind:.2f}" if not np.isnan(avg_wind) else "No Wind Data")

            # Top/Bottom Locations by Average Temperature (if multiple locations)
            if selected_location == "All" and 'Location' in filtered_weather.columns and 'Temperature_C' in filtered_weather.columns:
                best_temp, worst_temp = top_and_bottom(filtered_weather, 'Location', 'Temperature_C', top_n=5)
                if not best_temp.empty:
                    st.markdown("**Top 5 Locations by Average Temperature:**")
                    st.dataframe(best_temp)
                if not worst_temp.empty:
                    st.markdown("**Bottom 5 Locations by Average Temperature:**")
                    st.dataframe(worst_temp)

            # Visualizations
            st.subheader("Weather Visualizations")
            if not filtered_weather.empty and 'Date_Time' in filtered_weather.columns and 'Temperature_C' in filtered_weather.columns:
                fig_weather = px.line(
                    filtered_weather.sort_values(by='Date_Time'),
                    x='Date_Time',
                    y='Temperature_C',
                    title="Temperature Over Time"
                )
                st.plotly_chart(fig_weather, use_container_width=True)

            # Histogram and box plot for Temperature distribution
            if 'Temperature_C' in filtered_weather.columns and not filtered_weather.empty:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Temperature Distribution (Histogram)**")
                    fig_hist = px.histogram(filtered_weather, x='Temperature_C', nbins=30, title="Temperature Distribution")
                    st.plotly_chart(fig_hist, use_container_width=True)

                with col2:
                    st.markdown("**Temperature Distribution (Box Plot)**")
                    fig_box = px.box(filtered_weather, y='Temperature_C', title="Temperature Box Plot")
                    st.plotly_chart(fig_box, use_container_width=True)

            # Dataframe and Download
            st.subheader("Filtered Weather Data")
            st.dataframe(filtered_weather)
            csv_weather = filtered_weather.to_csv(index=False)
            st.download_button("Download Filtered Weather Data as CSV", data=csv_weather, file_name="filtered_weather.csv", mime="text/csv")

############################
# Combined Analysis Tab
############################
with combined_tab:
    st.subheader("Combined Analysis of AQI and Weather Data")
    # Load both data
    aqi_df = load_aqi_data()
    weather_df = load_weather_data()

    if aqi_df.empty or weather_df.empty:
        st.info("Need both AQI and Weather data available for combined analysis.")
    else:
        # Check if both have Year column
        if 'Year' not in aqi_df.columns or 'Year' not in weather_df.columns:
            st.write("Cannot perform combined analysis without 'Year' in both datasets.")
        else:
            st.markdown("**Correlation Analysis**")
            display_correlation_matrix(aqi_df, weather_df)

            # Scatter plot of Median AQI vs Temperature
            # Aggregate both by Year
            if 'Median AQI' in aqi_df.columns and 'Temperature_C' in weather_df.columns:
                weather_agg = weather_df.groupby('Year', as_index=False).mean(numeric_only=True)
                aqi_agg = aqi_df.groupby('Year', as_index=False).mean(numeric_only=True)
                merged = pd.merge(aqi_agg, weather_agg, on='Year', suffixes=('_aqi','_weather'), how='inner')

                if not merged.empty and 'Median AQI' in merged.columns and 'Temperature_C' in merged.columns:
                    st.markdown("**Relationship between Median AQI and Temperature**")
                    fig_scatter = px.scatter(
                        merged,
                        x='Median AQI',
                        y='Temperature_C',
                        color='Year',
                        title='Median AQI vs Temperature by Year'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.write("Not enough overlapping metrics to plot Median AQI vs Temperature.")

############################
# Data Reduction Tab
############################
with reduction_tab:
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

            # Summary stats of reduced dataset
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
- **AQI Data Tab**: Adjust filters to explore AQI metrics by state, county, and year.  
- **Weather Data Tab**: Adjust filters to explore weather metrics by location and year.  
- **Combined Analysis Tab**: View correlations and relationships if both datasets share a common 'Year' dimension.  
- **Data Reduction Tab**: Upload a large file, sample rows, drop columns, and download a reduced dataset.

Enjoy exploring your data!
""")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Group-017 | AQI & Weather", layout="wide")
st.title("Group-017: Air Quality & Weather Dashboard")

st.markdown("""
This project explores the relationship between **Air Quality Index (AQI)** and **Weather Patterns** across the United States, with a focus on New Mexico.  
It provides multiple views to analyze AQI data by state, county, and year, and correlates it with weather parameters like temperature, humidity, and wind speed.  
The dashboard enables insights into how environmental conditions vary by region and over time, including a data reduction tool for handling large weather datasets.
""")

# ========== Utility ==========
def load_csv(filename):
    try:
        data_dir = os.path.dirname(__file__)
        filepath = os.path.join(data_dir, "datasets", filename)
        return pd.read_csv(filepath)
    except Exception as e:
        st.warning(f"⚠️ Could not load {filename}: {e}")
        return pd.DataFrame()

# ========== Tabs ==========
tab1, tab2, tab3, tab4 = st.tabs(["AQI Data", "Weather Data", "Combined Analysis", "Data Reduction"])

# ========== Tab 1: AQI Data ==========
with tab1:
    st.subheader("Explore AQI Data")

    dataset_dir = os.path.join(os.path.dirname(__file__), "datasets")
    aqi_files = [f for f in os.listdir(dataset_dir) if f.startswith("annual_aqi_by_county_")]
    aqi_df = pd.concat([load_csv(f) for f in aqi_files], ignore_index=True) if aqi_files else pd.DataFrame()

    if not aqi_df.empty:
        aqi_df["Year"] = pd.to_numeric(aqi_df["Year"], errors="coerce")
        aqi_df.dropna(subset=["Year", "State", "County"], inplace=True)

        st.markdown("#### Filter Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            state = st.selectbox("Select State", ["All"] + sorted(aqi_df["State"].dropna().unique()))
        with col2:
            county = st.selectbox("Select County", ["All"] + sorted(aqi_df["County"].dropna().unique()))
        with col3:
            valid_years = sorted(aqi_df["Year"].dropna().unique().tolist())
            years = st.multiselect("Select Year(s)", valid_years, default=valid_years)

        filtered = aqi_df.copy()
        if state != "All":
            filtered = filtered[filtered["State"] == state]
        if county != "All":
            filtered = filtered[filtered["County"] == county]
        if years:
            filtered = filtered[filtered["Year"].isin(years)]

        st.dataframe(filtered)

        st.markdown("#### AQI Visualization")
        if "Median AQI" in filtered.columns and filtered["Median AQI"].dropna().shape[0] > 0:
            try:
                fig = px.line(
                    filtered.dropna(subset=["Median AQI"]),
                    x="Year",
                    y="Median AQI",
                    color="County",
                    title="Median AQI Over Years"
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error plotting AQI line chart: {e}")
        else:
            st.warning("No valid data for 'Median AQI'.")

# ========== Tab 2: Weather Data ==========
with tab2:
    st.subheader("Explore Weather Data")
    weather_df = load_csv("weather_data.csv")

    if not weather_df.empty:
        try:
            weather_df["Date_Time"] = pd.to_datetime(weather_df["Date_Time"], errors="coerce")
            weather_df["Year"] = weather_df["Date_Time"].dt.year
            for col in ["Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh"]:
                if col in weather_df.columns:
                    weather_df[col] = pd.to_numeric(weather_df[col], errors="coerce")

            st.markdown("#### Filter Options")
            col1, col2 = st.columns(2)
            with col1:
                location = st.selectbox("Select Location", ["All"] + sorted(weather_df["Location"].dropna().unique()))
            with col2:
                valid_w_years = sorted(weather_df["Year"].dropna().unique().tolist())
                w_years = st.multiselect("Select Year(s)", valid_w_years, default=valid_w_years)

            filtered = weather_df.copy()
            if location != "All":
                filtered = filtered[filtered["Location"] == location]
            if w_years:
                filtered = filtered[filtered["Year"].isin(w_years)]

            st.dataframe(filtered)

            if not filtered.empty and "Temperature_C" in filtered.columns:
                st.markdown("#### Temperature Over Time")
                try:
                    fig = px.line(
                        filtered.sort_values("Date_Time"),
                        x="Date_Time",
                        y="Temperature_C",
                        title="Temperature Over Time"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating temperature plot: {e}")
        except Exception as e:
            st.error(f"Error processing weather data: {e}")
    else:
        st.warning("No weather data available.")

# ========== Tab 3: Combined Analysis ==========
with tab3:
    st.subheader("Correlation Analysis Between AQI and Weather")

    if 'aqi_df' in locals() and not aqi_df.empty and 'weather_df' in locals() and not weather_df.empty:
        try:
            aqi_avg = aqi_df.groupby("Year").mean(numeric_only=True).reset_index()
            weather_avg = weather_df.groupby("Year").mean(numeric_only=True).reset_index()
            combined = pd.merge(aqi_avg, weather_avg, on="Year", suffixes=("_aqi", "_weather"))

            if not combined.empty:
                st.markdown("#### Correlation Heatmap")
                corr = combined.corr()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No matching years between AQI and Weather data for correlation.")
        except Exception as e:
            st.error(f"Error generating combined analysis: {e}")
    else:
        st.warning("Please ensure both AQI and Weather data are loaded and valid.")

# ========== Tab 4: Data Reduction ==========
with tab4:
    st.subheader("Weather Data Reduction Utility")

    uploaded = st.file_uploader("Upload large weather_data.csv file", type=["csv"])
    if uploaded is not None:
        try:
            df_large = pd.read_csv(uploaded)
            st.write(f"Original size: {df_large.shape}")

            columns_to_drop = st.multiselect("Drop Columns", options=df_large.columns.tolist())
            frac = st.slider("Fraction to keep", 0.01, 1.0, 0.3, step=0.01)

            if st.button("Reduce and Download"):
                if columns_to_drop:
                    df_large.drop(columns=columns_to_drop, inplace=True)
                df_reduced = df_large.sample(frac=frac, random_state=42)
                st.write(f"Reduced size: {df_reduced.shape}")
                csv = df_reduced.to_csv(index=False)
                st.download_button("Download Reduced CSV", data=csv, file_name="weather_reduced.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error during data reduction: {e}")

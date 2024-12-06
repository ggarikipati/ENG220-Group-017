Below is a fully revised and direct `README.md` tailored for your **Group 17** project, featuring the correct team members and focusing on the Air Quality & Weather dashboard you’ve developed. Following the README is a separate video script. This version does not reference previous groups or samples.

---

### README.md (Code Block)

```markdown
# ENG220-Group-17 Air Quality & Weather Analysis Project

## Authors:
- Waleed Alazemi
- Abdullah Alazmi
- **Ghazi Alazmi (Group Leader)**
- Abdullah Alkhatlan

## Project Overview

This project provides an integrated dashboard that merges Air Quality Index (AQI) data with corresponding weather conditions across various U.S. states, counties, and years. Through this interactive Streamlit application, stakeholders can explore how environmental factors (temperature, humidity, precipitation, wind speed) might relate to air quality trends. The goal is to inform researchers, policymakers, and the public about possible correlations and patterns that could influence environmental management and decision-making.

## Key Objectives
- **Understand Air Quality Patterns**: Examine how AQI metrics (e.g., Good Days, Moderate Days, Unhealthy Days) vary by region and over time.
- **Correlate Weather & AQI**: Investigate if and how weather conditions align with changes in air quality indicators.
- **Facilitate Data-Driven Insights**: Provide an accessible, interactive interface for deeper exploration, encouraging users to form hypotheses and identify trends.

## Features

1. **Multiple Tabs for Focused Exploration**:
   - **AQI Data Tab**: Filter by state, county, and year. Visualize AQI categories using stacked bar charts, track median AQI trends, and identify top/bottom counties by air quality.
   - **Weather Data Tab**: Filter by location and year. Examine temperature trends over time, view distributions of weather parameters, and understand their variability through histograms and box plots.
   - **Combined Analysis Tab**: If both datasets share a `Year` column, generate correlation matrices and scatter plots to see potential relationships between AQI and weather metrics.
   - **Data Reduction Tab**: Upload a large `weather_data.csv`, sample a fraction of rows, drop unnecessary columns, and download a reduced dataset for easier handling.

2. **Interactive Filters & Customization**:
   - Use the sidebar to adjust geographic, temporal, and category filters.
   - Instantly refresh visualizations and summary statistics based on user selections.
   - Hover over charts to reveal detailed data points.

3. **Data Export & Offline Analysis**:
   - Download filtered AQI or weather data as CSV files.
   - Simplify large datasets for more efficient external analysis or storage.

## Data Sources

- **AQI Data (2010-2023)**: Annual AQI metrics by county, indicating the number of Good, Moderate, and Unhealthy days, as well as other pollutant-related statistics.
- **Weather Data (Various Locations & Years)**: Temperature, humidity, precipitation, and wind speed readings, enabling comparisons and temporal assessments.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/galazmi/ENG220-Group-17.git
   cd ENG220-Group-17
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App**:
   ```bash
   streamlit run main_app.py
   ```
   Open the provided URL (usually `http://localhost:8501`) in your browser.

## Usage

- **AQI Data Tab**: Select a state, county, and one or multiple years. Observe AQI summaries, charts, and data tables. Download filtered results as needed.
- **Weather Data Tab**: Choose a location and year(s) to examine weather parameters. Adjust filters to see how conditions fluctuate, and download filtered data.
- **Combined Analysis Tab**: If both datasets align by year, explore correlation matrices and scatter plots to identify potential relationships between AQI and weather variables.
- **Data Reduction Tab**: Upload a large `weather_data.csv`. Specify a sampling fraction and columns to drop, then download a smaller, more manageable dataset.

## Acknowledgments & Contact

**Team ENG220-Group-17**  
- Waleed Alazemi  
- Abdullah Alazmi  
- Ghazi Alazmi (Group Leader)  
- Abdullah Alkhatlan

For questions or feedback, please reach out to any team member.

---

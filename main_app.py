import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description of the dashboard
st.title("Advanced Air Quality Permit Financing Dashboard")
st.markdown("""
Analyze air quality permits, pollution trends, and economic impact data using advanced visualizations and interactive features.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV Dataset", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV file
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Sidebar filters for advanced interactivity
    st.sidebar.header("Filter Data")
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    category_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Numeric column filters
    if numeric_cols:
        st.sidebar.subheader("Filter by Numeric Columns")
        for col in numeric_cols:
            min_val = float(data[col].min())
            max_val = float(data[col].max())
            step = (max_val - min_val) / 100
            selected_range = st.sidebar.slider(f"Range for {col}", min_val, max_val, (min_val, max_val), step=step)
            data = data[(data[col] >= selected_range[0]) & (data[col] <= selected_range[1])]

    # Category column filters
    if category_cols:
        st.sidebar.subheader("Filter by Categorical Columns")
        for col in category_cols:
            selected_values = st.sidebar.multiselect(f"Values for {col}", data[col].unique(), default=data[col].unique())
            data = data[data[col].isin(selected_values)]

    # Display filtered data
    st.write("### Filtered Dataset")
    st.dataframe(data)

    # Correlation heatmap
    if st.checkbox("Show Correlation Heatmap (Numeric Columns)"):
        st.write("### Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Dropdowns for graph selection
    st.write("### Visualization Options")
    x_col = st.selectbox("Select X-axis Column", data.columns)
    y_col = st.selectbox("Select Y-axis Column", data.columns)
    graph_type = st.selectbox("Select the Type of Graph", ["Line", "Bar", "Scatter", "Box", "Histogram"])

    # Plotting
    if st.button("Generate Plot"):
        fig, ax = plt.subplots()

        if graph_type == "Line":
            ax.plot(data[x_col], data[y_col], marker='o', label=f"{y_col} vs {x_col}")
            ax.set_title("Line Plot")
        elif graph_type == "Bar":
            ax.bar(data[x_col], data[y_col], label=f"{y_col} vs {x_col}")
            ax.set_title("Bar Chart")
        elif graph_type == "Scatter":
            ax.scatter(data[x_col], data[y_col], label=f"{y_col} vs {x_col}")
            ax.set_title("Scatter Plot")
        elif graph_type == "Box":
            sns.boxplot(data=data, x=x_col, y=y_col, ax=ax)
            ax.set_title("Box Plot")
        elif graph_type == "Histogram":
            sns.histplot(data[x_col], bins=30, kde=True, ax=ax)
            ax.set_title("Histogram")

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        st.pyplot(fig)

    # Statistical summary
    st.write("### Summary Statistics")
    st.write(data.describe())

    # Recommendations section
    st.write("### Insights and Recommendations")
    st.markdown("""
    - **Correlation Analysis**: Use the heatmap to identify relationships between numeric variables.
    - **Trends**: Visualize trends using line and scatter plots to evaluate permit cost effectiveness.
    - **Outliers**: Use box plots to detect outliers in permit costs or pollution levels.
    """)
else:
    st.info("Please upload a dataset to proceed.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import os
import traceback

# Define the build directory
BUILD_DIR = '/home/ubuntu/workdir/build'

# Ensure the build directory exists
os.makedirs(BUILD_DIR, exist_ok=True)

# Read the CSV files
consumption_df = pd.read_csv('/home/ubuntu/workdir/coffee_consumption_data.csv')
production_df = pd.read_csv('/home/ubuntu/workdir/FAO_Coffee_Data/Production_Crops_Livestock_E_All_Data.csv', encoding='latin-1')
climate_df = pd.read_csv('/home/ubuntu/workdir/climate_data.csv')  # Assuming this file exists

# Filter and process production data
coffee_production = production_df[production_df['Item'].str.contains('coffee', case=False, na=False) &
                                  (production_df['Element'] == 'Production')]
coffee_production = coffee_production.melt(id_vars=['Area', 'Item', 'Element'],
                                           value_vars=[col for col in coffee_production.columns if col.startswith('Y')],
                                           var_name='Year', value_name='Production')
coffee_production['Year'] = pd.to_numeric(coffee_production['Year'].str.extract('(\d+)', expand=False), errors='coerce')
coffee_production['Production'] = pd.to_numeric(coffee_production['Production'], errors='coerce')
coffee_production = coffee_production.dropna(subset=['Year', 'Production'])

# Create animated map of coffee consumption
def create_animated_consumption_map():
    fig = px.choropleth(consumption_df,
                        locations="Country",
                        locationmode="country names",
                        color="Total Coffee Consumption 2020-21 (1000s of 60-lb bags)",
                        hover_name="Country",
                        color_continuous_scale=px.colors.sequential.Viridis)

    fig.update_layout(title_text='Global Coffee Consumption (2020-21)')
    fig.write_html(os.path.join(BUILD_DIR, "animated_coffee_consumption_map.html"))

# Create 2D plots of production, consumption, and temperature
def create_2d_plots():
    try:
        # Merge data and validate
        merged_df = pd.merge(coffee_production, consumption_df, left_on='Area', right_on='Country', how='inner')
        merged_df = pd.merge(merged_df, climate_df, left_on=['Area', 'Year'], right_on=['Country', 'Year'], how='inner')

        # Check for missing values
        if merged_df.isnull().any().any():
            print("Warning: Missing values detected in the merged dataframe.")
            print(f"Columns with missing values: {merged_df.columns[merged_df.isnull().any()].tolist()}")
            merged_df = merged_df.dropna()
            print(f"Rows with missing values removed. Remaining rows: {len(merged_df)}")

        # Ensure correct data types and remove invalid data
        for col in ['Production', 'Total Coffee Consumption 2020-21 (1000s of 60-lb bags)', 'Temperature']:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
            merged_df = merged_df[merged_df[col].notnull() & (merged_df[col] > 0)]  # Remove rows with null or non-positive values
            print(f"Column '{col}' - min: {merged_df[col].min()}, max: {merged_df[col].max()}")

        if len(merged_df) == 0:
            raise ValueError("No valid data remaining after cleaning.")

        # Create 2D scatter plots
        plots = [
            ('Production', 'Total Coffee Consumption 2020-21 (1000s of 60-lb bags)', 'production_vs_consumption'),
            ('Production', 'Temperature', 'production_vs_temperature'),
            ('Total Coffee Consumption 2020-21 (1000s of 60-lb bags)', 'Temperature', 'consumption_vs_temperature')
        ]

        for x_col, y_col, filename in plots:
            fig = go.Figure(data=go.Scatter(
                x=merged_df[x_col],
                y=merged_df[y_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=merged_df['Year'],
                    colorscale='Viridis',
                    colorbar=dict(title='Year'),
                    opacity=0.8
                ),
                text=merged_df.apply(lambda row: f"Country: {row['Country_y']}<br>Year: {row['Year']}<br>{x_col}: {row[x_col]:.2f}<br>{y_col}: {row[y_col]:.2f}", axis=1),
                hoverinfo='text'
            ))

            fig.update_layout(
                title=f'{x_col} vs {y_col}',
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=600,
                width=800
            )

            fig.write_html(os.path.join(BUILD_DIR, f"{filename}.html"))
            print(f"2D plot '{filename}' created successfully.")

    except Exception as e:
        print(f"Error creating 2D plots: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")

# Trend analysis
def analyze_trends():
    global_production = coffee_production.groupby('Year')['Production'].sum().reset_index()
    trend, _ = np.polyfit(global_production['Year'], global_production['Production'], 1)
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Year', y='Production', data=global_production)
    plt.title(f'Global Coffee Production Trend (Slope: {trend:.2f})')
    plt.savefig(os.path.join(BUILD_DIR, 'production_trend.png'))
    plt.close()

# Correlation analysis
def analyze_correlations():
    merged_df = pd.merge(coffee_production, climate_df, left_on=['Area', 'Year'], right_on=['Country', 'Year'], how='inner')
    corr_temp = stats.pearsonr(merged_df['Production'], merged_df['Temperature'])
    corr_precip = stats.pearsonr(merged_df['Production'], merged_df['Precipitation'])

    print(f"Correlation between Production and Temperature: {corr_temp[0]:.2f} (p-value: {corr_temp[1]:.4f})")
    print(f"Correlation between Production and Precipitation: {corr_precip[0]:.2f} (p-value: {corr_precip[1]:.4f})")

# Clustering analysis
def perform_clustering():
    merged_df = pd.merge(coffee_production, climate_df, left_on=['Area', 'Year'], right_on=['Country', 'Year'], how='inner')
    X = merged_df[['Production', 'Temperature', 'Precipitation']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    merged_df['Cluster'] = kmeans.fit_predict(X)

    fig = px.scatter_3d(merged_df, x='Production', y='Temperature', z='Precipitation', color='Cluster', hover_name='Country')
    fig.update_layout(title='Coffee Production Clusters based on Climate Factors')
    fig.write_html(os.path.join(BUILD_DIR, "production_clusters.html"))

# Create visualizations and perform analyses
create_animated_consumption_map()
create_2d_plots()
analyze_trends()
analyze_correlations()
perform_clustering()

# Calculate and print some basic statistics
total_consumption = consumption_df['Total Coffee Consumption 2020-21 (1000s of 60-lb bags)'].sum()
average_consumption = consumption_df['Total Coffee Consumption 2020-21 (1000s of 60-lb bags)'].mean()
median_consumption = consumption_df['Total Coffee Consumption 2020-21 (1000s of 60-lb bags)'].median()

print(f"Total global coffee consumption: {total_consumption:.2f} thousand 60-lb bags")
print(f"Average consumption per country: {average_consumption:.2f} thousand 60-lb bags")
print(f"Median consumption: {median_consumption:.2f} thousand 60-lb bags")

print("Analysis complete. Check the generated HTML files and PNG images for visualizations in the build directory.")

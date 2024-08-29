# Global Coffee Consumption Trends and Climate Impact Analysis

## Project Overview
This project aims to analyze global coffee consumption trends and investigate the potential impact of climate change on coffee production. By combining data on coffee consumption, production, and climate variables, we seek to uncover insights into the relationships between these factors and their implications for the global coffee industry.

## Data Sources
- Global Coffee Consumption Data: Annual consumption data for various countries
- Coffee Production Data: Historical production data from major coffee-producing regions
- Climate Data: Temperature and precipitation data for coffee-producing countries

## Methodology
1. Data Collection and Preprocessing
2. Exploratory Data Analysis
3. Trend Analysis
4. Correlation Analysis
5. Clustering Analysis
6. Visualization Creation

## Analysis Performed
- Time series analysis of global coffee production trends
- Correlation analysis between climate variables and coffee production
- Clustering analysis to identify patterns in coffee production and climate factors
- Visualization of coffee consumption and production data

## Key Findings
1. Global coffee consumption continues to rise, with the United States and Brazil leading in consumption.
2. There is a weak positive correlation between coffee production and both temperature and precipitation.
3. Coffee-producing countries can be clustered based on their production levels and climate conditions.
4. Climate change may have varying impacts on different coffee-producing regions.

## Visualizations
- Interactive global coffee consumption map: [View Map](docs/animated_coffee_consumption_map.html)
- Coffee production trend analysis: [View Trend](docs/production_trend.png)
- Production vs. Consumption scatter plot: [View Plot](docs/production_vs_consumption.html)
- Production vs. Temperature scatter plot: [View Plot](docs/production_vs_temperature.html)
- Consumption vs. Temperature scatter plot: [View Plot](docs/consumption_vs_temperature.html)
- Coffee production clusters based on climate factors: [View Clusters](docs/production_clusters.html)

## Repository Structure
- `analyze_and_visualize.py`: Main script for data analysis and visualization
- `coffee_consumption_data.csv`: Global coffee consumption dataset
- `climate_data.csv`: Climate data for coffee-producing countries
- `docs/`: Directory containing generated visualizations and HTML files

## How to Run
1. Ensure you have Python 3.7+ installed
2. Install required packages: `pip install pandas matplotlib seaborn plotly scikit-learn`
3. Run the analysis script: `python analyze_and_visualize.py`
4. View the generated visualizations in the `docs/` directory

## Future Work
- Incorporate more detailed climate data and projections
- Analyze the impact of coffee production methods on quality and yield
- Investigate the economic implications of changing coffee production patterns

## Contributors
- [iftekharanwar]

## License
This project is open source and available under the [MIT License](LICENSE).

# Description
This dashboard renders line and histogram charts for daily peak UV Idex (Ultraviolet Index) for 5 sites across New Zealand between 2010 and 2017. It has aspects of data cleansing, API data retrieval, EDA, prediction, and 

# Data Description
This dashboard uses a CSV of daily UVI, collected and collated by the Ministry for the Environment.
Sourced here
https://data.mfe.govt.nz/table/89468-daily-peak-uv-index-value-19812017/

As the full dataset is quite long, only a subset of the data, from approximately 2010 to 2017 is used in this codebase.

# Components

- Line chart
This graph displays daily peak UV Index at 5 locations across New Zealand, by year, with the ability to filter location.

-Histogram
This graph displays the frequency of daily UVI values, coded by location. If the line chart is filtered to look at a particular time period within a year, histogram data will be updated to reflect this.

-UVI trend
The overall UVI trend (using each daily datapoint across all years) for each location can be optionally calculated. This trend analysis uses simple Linear Regression, and reports on whether the overall trend is increasing or decreasing, given the dataset.

# Installation


1) First generate a docker build via

```
docker build -t python-dashboard .
```

2) Then run the dashboard via cli command; or using Docker GUI

```
docker run -p 8050:8050 python-dashboard   
```
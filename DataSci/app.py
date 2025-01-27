import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import numpy as np
# import dash_bootstrap_components as dbc
import wikipediaapi
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time  # Used to simulate a delay for demonstration


def load_wikipedia_information():
    """Return description section of specified length, from wikipedia via API"""
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='DataSci test (kaitlinn1567@gmail.com)', language='en')
    page_py = wiki_wiki.page('Ultraviolet index')
    section_history = page_py.section_by_title('Description')
    summary_text = ("%s - %s" %
                    (section_history.title, section_history.text[16:442]))
    return summary_text


def process_load_data():
    """Load CSV and preprocess data"""
    # Read csv file
    # df = pd.read_csv(
    #    'C:\\Users\\User\\Downloads\\mfe-daily-peak-uv-index-value-19812017-CSV\\daily-peak-uv-index-value-19812018.csv')
    df = pd.read_csv('daily-peak-uv-index-value-19812018.csv')
    # Remove any NA/incomplete rows
    df = df[['Date', 'Location', 'Daily_peak_UVI']].dropna()
    # Create 'Year' from date to use in dropdown
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    df['Year'] = df['Date'].dt.year

    return df

# TODO not fully implemented yet


def model_design(df):

    X = df[['Date', 'Location']]
    X = pd.get_dummies(df, columns=['Date', 'Location'])

    # Splitting the data into features (X) and target (y)
    y = df['Daily_peak_UVI']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Training a simple linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predicting on the test set
    y_pred = model.predict(X_test)

    # Calculate model performance
    mae = mean_absolute_error(y_test, y_pred)
    return X, y

# TODO not fully implemented yet


def predict_future_years():
    model = LinearRegression()
    future_years = pd.DataFrame({
        'Year': [2024, 2025, 2026],
        'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    })

    future_predictions = model.predict(future_years)
    return future_predictions


def predict_slope(df):
    time.sleep(2)
    """Using Linear Regression, fit a model for each location, and calculate slope and coefficients"""
    # Dict to hold model for each location
    models = {}

    # Split DataFrame based on 'Location' column
    split_df = {location: df[df['Location'] == location]
                for location in df['Location'].unique()}

    # List to hold string text of LM slope
    result = []

    # Fit a Linear Regression model for each location in loop
    for location, location_df in split_df.items():
        print(location)
        # Define features and target
        #location_df.drop(columns=['Date'], inplace=True)

        X = pd.get_dummies(location_df, columns=['Date','Location'])
        y = location_df['Daily_peak_UVI']

        # Initialize and fit the model
        model = LinearRegression()
        model.fit(X, y)

        models[location] = model
        print(f"Model for {location}:")
        print(f"Slope: {model.coef_[0]}")
        print(f"Intercept: {model.intercept_}")
        print(location_df.columns)

        if model.coef_[0] > 0:
            result.append(
                f"Between {df['Year'].min()} and {df['Year'].max()}, the overall UV Index trend is increasing at {location}")
        else:
            result.append(
                f"Between {df['Year'].min()} and {df['Year'].max()}, the overall UV Index trend is decreasing at {location}")
    print(result)
    return result


# Run internal methods to generate data
df = process_load_data()
summary_text = load_wikipedia_information()

# Generate initial representation of charts
fig = px.line(df[df['Year'] == 2016], x='Date', y='Daily_peak_UVI',
              color='Location', title='Daily Peak UVI over time (2013)')
fig2 = px.histogram(df[df['Year'] == 2016],
                    x='Daily_peak_UVI', color="Location")

# Initialize app
app = dash.Dash(__name__)
# Required to use gunix
server = app.server

# Define app layout
app.layout = html.Div([
    html.H1("New Zealand Daily Peak UVI"),
    dcc.Dropdown(
        id='year-dropdown',
        options=[
            {'label': str(year), 'value': year} for year in df['Year'].unique()
        ],
        value=2010  # Default selected year
    ),
    html.H3('UVI summary description (sourced from Wikipedia)'),
    html.Pre(summary_text, style={
             'white-space': 'pre-wrap', 'word-wrap': 'break-word'}),
    dcc.Graph(id='uvi-line-chart', figure=fig),

    dcc.Loading(
        id="loading-btn",
        type="default",  # Spinner type (can be "circle", "dot", or "default")
        children=html.Button('Show overall UVI trend for all years and locations',
                             id='generate-btn', n_clicks=0)),
    html.Pre(id='output-text', children="Initial value",
             style={'white-space': 'pre-wrap'}),
    dcc.Graph(id='uvi-hist-chart', figure=fig2)
])


@app.callback(
    dash.dependencies.Output('uvi-line-chart', 'figure'),
    [dash.dependencies.Input('year-dropdown', 'value')]
)
def update_graph(selected_year):
    # Filter the data based on the selected year
    # Filter data to selected year only
    filtered_df = df[df['Year'] == selected_year]
    fig = px.line(filtered_df, x='Date', y='Daily_peak_UVI', color='Location',
                  title=f'Daily Peak UVI over time ({selected_year})')

    return fig


# Second 'Input' parameter puts line chart as input to histogram, for responsive chart updates
@app.callback(
    Output('uvi-hist-chart', 'figure'),
    [Input('year-dropdown', 'value'),
     Input('uvi-line-chart', 'relayoutData')]
)
def update_graph(selected_year, relayoutData):
    # Filter data to selected year only
    filtered_df = df[df['Year'] == selected_year]
    # if line chart x axis has been modified, and selected year in dropdown is the same as graph data, then update histogram
    if relayoutData and 'xaxis.range[0]' in relayoutData and selected_year == pd.to_datetime(relayoutData['xaxis.range[0]']).year:
        xaxis_range = [relayoutData['xaxis.range[0]'],
                       relayoutData['xaxis.range[1]']]
        start_date, end_date = pd.to_datetime(
            xaxis_range[0]), pd.to_datetime(xaxis_range[1])
        filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & (
            filtered_df['Date'] <= end_date)]
    # else different year selected in dropdown, so reset data
    else:
        filtered_df = df[df['Year'] == selected_year]
    fig = px.histogram(filtered_df, x='Daily_peak_UVI', color="Location",
                       title=f'Distribution of Daily Peak UV Index ({selected_year})')
    return fig


@app.callback(
    Output('output-text', 'children'),
    Input('generate-btn', 'n_clicks')
)
def update_text(n_clicks):
    # Call method only when button clicked at least once
    if n_clicks > 0:
        result = predict_slope(df)
        output_string = "\n".join(result)
        return output_string
    else:
        default_text = "Please press button for output."
        return default_text


# Run app in debug mode
if __name__ == '__main__':
    app.run_server(debug=True)

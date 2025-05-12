# dashboard_app/layout.py
from dash import dcc, html
import dash_bootstrap_components as dbc
import data_loader  # Changed from relative import
import config       # Changed from relative import
import os

def create_main_layout():
    # Fetch experiments inside the layout function
    available_experiments = data_loader.get_experiment_folders(config.DATA_REPO_PATH)
    # print(f"Debug: Available experiments inside create_main_layout: {available_experiments}") # Optionally remove this debug print too

    # # --- TEMPORARY DEBUG: Use hardcoded options --- # << REMOVED
    # hardcoded_options = [
    #     {'label': 'Test Experiment 1', 'value': 'path/to/exp1'},
    #     {'label': 'Test Experiment 2', 'value': 'path/to/exp2'}
    # ]
    # # --- END TEMPORARY DEBUG --- # << REMOVED

    return dbc.Container([
        dbc.Row(dbc.Col(html.H1("Hovercraft Sensor Data Dashboard"), width=12, className="mb-3")),

        dbc.Row([
            dbc.Col([
                # # --- ADDING OBVIOUS TEXT BLOCK FOR DEBUG --- # << REMOVED
                # html.H1("--- IS THIS VISIBLE??? ---", style={'color': 'red', 'fontSize': '40px'}),
                # # --- END DEBUG TEXT BLOCK --- # << REMOVED
                html.H4("Select Experiment"),
                dcc.Dropdown(
                    id='experiment-dropdown',
                    # Restore dynamic options
                    options=[{'label': name, 'value': path} for name, path in available_experiments.items()],
                    # Restore dynamic default value
                    value=list(available_experiments.values())[0] if available_experiments else None,
                    clearable=False
                )
            ], width=12, className="mb-4")
        ]),

        # GPS Section (will be populated by callbacks)
        dbc.Row([
            dbc.Col([
                html.H3("GPS Data"),
                dcc.Loading(dcc.Graph(id='gps-map-plot')) # Add dcc.Loading
            ], width=6),
            dbc.Col([
                html.H4("GPS Speed"),
                dcc.Loading(dcc.Graph(id='gps-speed-plot')),
                html.H4("GPS Altitude"),
                dcc.Loading(dcc.Graph(id='gps-altitude-plot'))
            ], width=6)
        ], className="mb-4"),

        # IMU Section (controls will be updated by callbacks)
        dbc.Row([
            dbc.Col([
                html.H3("IMU Data"),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(id='imu-sensor-dropdown', clearable=False), width=4), # Options populated by callback
                    dbc.Col(dcc.Dropdown(
                        id='imu-measurement-dropdown',
                        options=[{'label': m.capitalize(), 'value': m} for m in config.IMU_MEASUREMENT_TYPES],
                        value=config.IMU_MEASUREMENT_TYPES[0],
                        clearable=False
                    ), width=4),
                ]),
                dcc.Loading(dcc.Graph(id='imu-data-plot')),
                html.Div(id='sensor-orientation-info', className="mt-3")
            ], width=12)
        ]),
        # Store for intermediate data if needed, e.g., currently selected experiment's sensor list
        dcc.Store(id='current-experiment-path-store'),
        dcc.Store(id='current-orientations-store'),
        dcc.Store(id='current-imu-sensors-list-store')

    ], fluid=True) 
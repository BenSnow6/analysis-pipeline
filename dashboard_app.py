# dashboard_app.py
import dash
from dash import dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time # For benchmarking if needed

# Import your data loading utilities
try:
    from data_utils import (
        load_experiment_data,
        list_experiment_types,
        list_experiments,
        ExperimentData, # For type hinting
        SENSOR_INFO,
        # IMU_DATA_TYPES, # Not directly needed here, taken from exp_data
        IMU_DF_COLUMN_MAP
    )
except ImportError:
    print("Error: data_utils.py not found. Please ensure it's in the same directory or Python path.")
    exit()

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# --- Global store for the currently loaded ExperimentData object ---
# This is a simplification for single-user scenarios.
# For multi-user, use server-side caching (e.g., Flask-Caching).
current_experiment_data_store: ExperimentData | None = None
current_experiment_key: str | None = None


# --- App Layout (mostly unchanged, minor ID change for clarity if needed) ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Hovercraft Experiment Data Dashboard", className="text-center my-4"))),
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Experiment Type:"),
            dcc.Dropdown(id='dropdown-exp-type', options=[])
        ], width=6),
        dbc.Col([
            dbc.Label("Select Experiment Run:"),
            dcc.Dropdown(id='dropdown-exp-name', options=[])
        ], width=6)
    ], className="mb-3"),
    dbc.Row(
        dbc.Col(
            dbc.Button("Load Experiment Data", id="button-load-experiment", color="primary", className="w-100"),
            width=12
        ), className="mb-4"
    ),
    dcc.Store(id='store-trigger-imu-update'), # Triggers IMU plots when experiment or sensors change
    dcc.Store(id='store-trigger-gps-update'), # Triggers GPS plots when experiment changes
    html.Div(id='div-visualization-area', children=[
        dbc.Spinner(html.Div(id='loading-feedback'), color="primary"),
        dbc.Row([
            dbc.Col(html.H3("GPS Data", className="mt-4"), width=12),
            dbc.Col(dcc.Graph(id='graph-gps-path'), width=12),
            dbc.Col(dcc.Graph(id='graph-gps-speed'), width=6),
            dbc.Col(dcc.Graph(id='graph-gps-altitude'), width=6)
        ], id='row-gps-viz', style={'display': 'none'}),
        dbc.Row([
             dbc.Col(html.H3("IMU Data", className="mt-4"), width=12),
             dbc.Col([
                 dbc.Label("Select Sensors to Display:"),
                 dcc.Checklist(id='checklist-imu-sensors', options=[], value=[], inline=True, labelStyle={'margin-right': '20px'})
             ], width=12, className="mb-2")
        ], id='row-imu-controls', style={'display': 'none'}),
        html.Div(id='div-imu-graphs')
    ])
], fluid=True)

# --- Helper Functions (create_empty_figure is same, get_timestamp_column modified) ---
def create_empty_figure(message="No data to display"):
    fig = go.Figure()
    fig.update_layout(
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{'text': message, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 16}}]
    )
    return fig

def get_imu_time_sync_column(imu_type: str):
    """Gets the specific 'time_from_sync_IMUTYPE' column name."""
    # e.g. for 'accel', IMU_DF_COLUMN_MAP['accel']['time_from_sync_raw'] is 'time_from_sync_accel'
    if imu_type in IMU_DF_COLUMN_MAP and 'time_from_sync_raw' in IMU_DF_COLUMN_MAP[imu_type]:
        return IMU_DF_COLUMN_MAP[imu_type]['time_from_sync_raw']
    return None # Should not happen if IMU_DF_COLUMN_MAP is correct

def get_gps_time_column(df: pd.DataFrame):
    """Finds the best available time column for GPS data, exclusively time_from_sync."""
    # Prefer parsed datetime column time_from_sync_dt
    dt_col = 'time_from_sync_dt'
    if dt_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[dt_col]):
        if not df[dt_col].isnull().all():
            return dt_col
            
    # Fallback to raw numeric time_from_sync column
    raw_col = 'time_from_sync'
    if raw_col in df.columns:
        # Could add a check here to ensure it's numeric if strictness is needed
        return raw_col
        
    return None # Return None if neither time_from_sync_dt nor time_from_sync is found

# --- Callbacks ---
@app.callback(
    Output('dropdown-exp-type', 'options'),
    Input('dropdown-exp-type', 'id')
)
def populate_exp_type_dropdown(_):
    types = list_experiment_types()
    return [{'label': t, 'value': t} for t in types]

@app.callback(
    Output('dropdown-exp-name', 'options'),
    Output('dropdown-exp-name', 'value'),
    Input('dropdown-exp-type', 'value')
)
def populate_exp_name_dropdown(selected_exp_type):
    if not selected_exp_type:
        return [], None
    names = list_experiments(selected_exp_type)
    return [{'label': n, 'value': n} for n in names], None

@app.callback(
    [Output('store-trigger-gps-update', 'data'), # Use a simple trigger like a timestamp
     Output('store-trigger-imu-update', 'data'),
     Output('checklist-imu-sensors', 'options'),
     Output('checklist-imu-sensors', 'value'),
     Output('row-gps-viz', 'style'),
     Output('row-imu-controls', 'style'),
     Output('div-imu-graphs', 'children', allow_duplicate=True),
     Output('loading-feedback', 'children')],
    Input('button-load-experiment', 'n_clicks'),
    [State('dropdown-exp-type', 'value'),
     State('dropdown-exp-name', 'value')],
    prevent_initial_call=True
)
def handle_load_experiment_button(n_clicks, exp_type, exp_name):
    global current_experiment_data_store, current_experiment_key
    if not n_clicks or not exp_type or not exp_name:
        return no_update, no_update, [], [], {'display': 'none'}, {'display': 'none'}, [], "Select experiment type and name."

    feedback_msg = f"Loading data for {exp_type}/{exp_name}..."
    print(feedback_msg) # Server-side log

    exp_key = f"{exp_type}_{exp_name}"
    # Load data only if it's a new experiment
    # if exp_key != current_experiment_key: # This check is implicitly handled by button click
    current_experiment_data_store = load_experiment_data(exp_type, exp_name)
    current_experiment_key = exp_key
    # else:
    #     print(f"Data for {exp_key} already loaded, reusing.")
    #     # This else branch is less likely useful with button click triggering

    if not current_experiment_data_store:
        current_experiment_key = None
        return (None, None, [], [], {'display': 'none'}, {'display': 'none'}, [],
                f"Failed to load data for {exp_type}/{exp_name}.")

    sensor_options = [{'label': s, 'value': s} for s in current_experiment_data_store.available_sensors]
    default_sensor_value = [current_experiment_data_store.available_sensors[0]] if current_experiment_data_store.available_sensors else []
    display_style = {'display': 'flex'}
    trigger_data = time.time() # Unique value to trigger updates

    return (trigger_data, trigger_data, sensor_options, default_sensor_value,
            display_style, display_style, [], f"Loaded: {exp_type}/{exp_name}.")

@app.callback(
    [Output('graph-gps-path', 'figure'),
     Output('graph-gps-speed', 'figure'),
     Output('graph-gps-altitude', 'figure')],
    Input('store-trigger-gps-update', 'data') # Triggered when experiment loaded
)
def update_gps_graphs(trigger_data):
    global current_experiment_data_store
    if not trigger_data or not current_experiment_data_store:
        msg = "Select and load an experiment to view GPS data."
        return create_empty_figure(msg), create_empty_figure(msg), create_empty_figure(msg)

    exp_data = current_experiment_data_store
    if exp_data.gps_data is None or exp_data.gps_data.empty:
        msg = "No GPS data available for this experiment."
        return create_empty_figure(msg), create_empty_figure(msg), create_empty_figure(msg)
    
    gps_df = exp_data.gps_data.copy() # Work with a copy

    # GPS Path
    lat_col = next((col for col in gps_df.columns if 'lat' in col.lower()), None)
    lon_col = next((col for col in gps_df.columns if 'lon' in col.lower() or 'lng' in col.lower()), None)
    fig_path = create_empty_figure("Lat/Lon columns not found.")
    if lat_col and lon_col:
        try:
            gps_df[lat_col] = pd.to_numeric(gps_df[lat_col], errors='coerce')
            gps_df[lon_col] = pd.to_numeric(gps_df[lon_col], errors='coerce')
            gps_df_cleaned = gps_df.dropna(subset=[lat_col, lon_col])
            if not gps_df_cleaned.empty:
                 fig_path = px.scatter_mapbox(gps_df_cleaned, lat=lat_col, lon=lon_col, zoom=12, height=400, title=f"GPS Path: {exp_data.experiment_name}")
                 fig_path.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":30,"l":0,"b":0})
            else: fig_path = create_empty_figure("GPS Lat/Lon data empty/invalid.")
        except Exception as e: fig_path = create_empty_figure(f"Error plotting GPS map: {e}")

    # GPS Speed
    speed_col = next((col for col in gps_df.columns if 'speed' in col.lower()), None)
    time_col_gps = get_gps_time_column(gps_df)
    fig_speed = create_empty_figure("Speed or Time data not found.")
    if speed_col and time_col_gps:
        try:
            gps_df[speed_col] = pd.to_numeric(gps_df[speed_col], errors='coerce')
            gps_df_cleaned = gps_df.dropna(subset=[time_col_gps, speed_col])
            if not gps_df_cleaned.empty:
                fig_speed = px.line(gps_df_cleaned, x=time_col_gps, y=speed_col, title="GPS Speed")
            else: fig_speed = create_empty_figure("GPS Speed/Time data empty/invalid.")
        except Exception as e: fig_speed = create_empty_figure(f"Error plotting GPS speed: {e}")
    
    # GPS Altitude
    alt_col = next((col for col in gps_df.columns if 'alt' in col.lower()), None)
    fig_alt = create_empty_figure("Altitude data not found.")
    if alt_col and time_col_gps:
        try:
            gps_df[alt_col] = pd.to_numeric(gps_df[alt_col], errors='coerce')
            gps_df_cleaned = gps_df.dropna(subset=[time_col_gps, alt_col])
            if not gps_df_cleaned.empty:
                fig_alt = px.line(gps_df_cleaned, x=time_col_gps, y=alt_col, title="GPS Altitude")
            else: fig_alt = create_empty_figure("GPS Altitude/Time data empty/invalid.")
        except Exception as e: fig_alt = create_empty_figure(f"Error plotting GPS alt: {e}")
        
    return fig_path, fig_speed, fig_alt

@app.callback(
    Output('div-imu-graphs', 'children', allow_duplicate=True),
    [Input('store-trigger-imu-update', 'data'), # Triggered when experiment changes
     Input('checklist-imu-sensors', 'value')],  # Triggered when sensor selection changes
    prevent_initial_call=True
)
def update_imu_graphs(trigger_data, selected_sensors):
    global current_experiment_data_store
    if not trigger_data or not current_experiment_data_store or not selected_sensors:
        return [dbc.Alert("Load experiment and select sensors for IMU data.", color="info", className="mt-3")]

    exp_data = current_experiment_data_store
    imu_graphs_layout = []
    MAX_POINTS_TO_PLOT = 200 # Define max points for downsampling

    for sensor_name in selected_sensors:
        if sensor_name not in exp_data.available_sensors:
            continue

        sensor_title = html.H4(f"Sensor: {sensor_name}", className="mt-3 mb-2")
        imu_graphs_layout.append(sensor_title)
        row_content = []

        for imu_type in exp_data.available_imu_types.get(sensor_name, []):
            df_original = exp_data.get_imu_data(sensor_name, imu_type)
            if df_original is None or df_original.empty:
                row_content.append(dbc.Col(dcc.Graph(figure=create_empty_figure(f"No {imu_type} data for {sensor_name}")), width=12, md=6, lg=4))
                continue
            
            df = df_original.copy() # Work with a copy

            sensor_specific_info = SENSOR_INFO.get(imu_type, {})
            plot_title = f"{imu_type.capitalize()} - {sensor_name}"
            y_label = sensor_specific_info.get('ylabel', imu_type.capitalize())
            axes_to_plot = sensor_specific_info.get('axes', [])
            
            time_col_imu = get_imu_time_sync_column(imu_type)
            
            if not time_col_imu or time_col_imu not in df.columns:
                row_content.append(dbc.Col(dcc.Graph(figure=create_empty_figure(f"Time_from_sync col not found for {imu_type} - {sensor_name}")), width=12, md=6, lg=4))
                continue

            df[time_col_imu] = pd.to_numeric(df[time_col_imu], errors='coerce')

            fig_imu = go.Figure()
            valid_axes_plotted = 0
            for axis in axes_to_plot:
                if axis in df.columns:
                    try:
                        df[axis] = pd.to_numeric(df[axis], errors='coerce')
                        df_cleaned_for_axis = df.dropna(subset=[time_col_imu, axis])
                        
                        if not df_cleaned_for_axis.empty:
                            df_to_plot = df_cleaned_for_axis
                            if len(df_cleaned_for_axis) > MAX_POINTS_TO_PLOT:
                                step = len(df_cleaned_for_axis) // MAX_POINTS_TO_PLOT
                                df_to_plot = df_cleaned_for_axis.iloc[::step, :]
                            
                            fig_imu.add_trace(go.Scatter(x=df_to_plot[time_col_imu], y=df_to_plot[axis], mode='lines', name=axis))
                            valid_axes_plotted +=1
                    except Exception as e_plot:
                        print(f"Error plotting axis {axis} for {imu_type} of {sensor_name}: {e_plot}")
            
            if valid_axes_plotted > 0:
                fig_imu.update_layout(title=plot_title, xaxis_title="Time from Sync (s)", yaxis_title=y_label, height=300, margin={"t":50,"l":10,"r":10,"b":10}, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            else:
                fig_imu = create_empty_figure(f"No valid data for {imu_type} - {sensor_name}")

            row_content.append(dbc.Col(dcc.Graph(figure=fig_imu, config={'displayModeBar': True}), width=12, md=6, lg=4))
        
        if row_content: imu_graphs_layout.append(dbc.Row(row_content))
        else: imu_graphs_layout.append(dbc.Alert(f"No plottable IMU data for sensor {sensor_name}.", color="warning"))

    if not imu_graphs_layout:
         return [dbc.Alert("No IMU data to display.", color="info", className="mt-3")]
    return imu_graphs_layout

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8051)
# dashboard_app/callbacks.py
from dash import Input, Output, State, html, no_update
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import data_loader
import config

def register_callbacks(app):
    # Update stores and IMU sensor dropdown when experiment changes
    @app.callback(
        [Output('current-experiment-path-store', 'data'),
         Output('current-orientations-store', 'data'),
         Output('current-imu-sensors-list-store', 'data'),
         Output('imu-sensor-dropdown', 'options'),
         Output('imu-sensor-dropdown', 'value')],
        [Input('experiment-dropdown', 'value')]
    )
    def update_experiment_dependent_stores(selected_experiment_path):
        if not selected_experiment_path:
            return None, {}, [], [], None

        orientations = data_loader.load_sensor_orientations(selected_experiment_path)
        imu_sensors_for_exp = data_loader.get_imu_sensors_for_experiment(selected_experiment_path, orientations)

        imu_sensor_options = [{'label': s.upper(), 'value': s} for s in imu_sensors_for_exp]
        default_imu_sensor = imu_sensors_for_exp[0] if imu_sensors_for_exp else None

        return selected_experiment_path, orientations, imu_sensors_for_exp, imu_sensor_options, default_imu_sensor

    # Update GPS Plots
    @app.callback(
        [Output('gps-map-plot', 'figure'),
         Output('gps-speed-plot', 'figure'),
         Output('gps-altitude-plot', 'figure')],
        [Input('current-experiment-path-store', 'data')]
    )
    def update_gps_plots(experiment_path):
        map_fig = go.Figure(go.Scattermapbox(lat=[], lon=[]))
        map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":30,"l":0,"b":0}, height=400)
        speed_fig = go.Figure(go.Scatter(x=[], y=[]))
        speed_fig.update_layout(xaxis_title="Time (s)", yaxis_title="Speed (KPH)", margin={"r":10,"t":10,"l":40,"b":30}, height=200)
        alt_fig = go.Figure(go.Scatter(x=[], y=[]))
        alt_fig.update_layout(xaxis_title="Time (s)", yaxis_title="Altitude (m)", margin={"r":10,"t":10,"l":40,"b":30}, height=200)


        if not experiment_path:
            return map_fig, speed_fig, alt_fig

        gps_df = data_loader.load_gps_data(experiment_path)
        if gps_df.empty:
            return map_fig, speed_fig, alt_fig

        if "Lat" in gps_df.columns and "Lng" in gps_df.columns:
            map_fig = px.scatter_mapbox(gps_df, lat="Lat", lon="Lng",
                                        hover_name="Time" if "Time" in gps_df.columns else None,
                                        hover_data={"SpeedKPH": True, "Alt": True} if all(c in gps_df.columns for c in ["SpeedKPH", "Alt"]) else None,
                                        color_discrete_sequence=["blue"], zoom=14)
            map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":30,"l":0,"b":0}, height=400)


        if 'SpeedKPH' in gps_df.columns and 'time_from_sync' in gps_df.columns:
            speed_fig = px.line(gps_df, x='time_from_sync', y='SpeedKPH', title=None)
            speed_fig.update_layout(xaxis_title="Time (s)", yaxis_title="Speed (KPH)", margin={"r":10,"t":10,"l":40,"b":30}, height=200)
            speed_fig.update_traces(line=dict(color='red'))

        if 'Alt' in gps_df.columns and 'time_from_sync' in gps_df.columns:
            alt_fig = px.line(gps_df, x='time_from_sync', y='Alt', title=None)
            alt_fig.update_layout(xaxis_title="Time (s)", yaxis_title="Altitude (m)", margin={"r":10,"t":10,"l":40,"b":30}, height=200)
            alt_fig.update_traces(line=dict(color='green'))

        return map_fig, speed_fig, alt_fig

    # Update IMU Plot and Orientation Info
    @app.callback(
        [Output('imu-data-plot', 'figure'),
         Output('sensor-orientation-info', 'children')],
        [Input('imu-sensor-dropdown', 'value'),
         Input('imu-measurement-dropdown', 'value'),
         Input('current-experiment-path-store', 'data'),
         Input('current-orientations-store', 'data')]
    )
    def update_imu_plot(selected_sensor, selected_measurement, experiment_path, orientations):
        fig_title = "IMU Data"
        if selected_sensor and selected_measurement:
            fig_title = f"{(selected_sensor or 'N/A').upper()} - {(selected_measurement or 'N/A').capitalize()}"

        imu_fig = go.Figure()
        imu_fig.update_layout(
            title=fig_title,
            xaxis_title="Time (s)", yaxis_title="Value",
            margin={"r":10,"t":40,"l":40,"b":30}, height=400
        )
        orientation_info_children = [html.P("Select an experiment and sensor.")]

        if not experiment_path or not selected_sensor or not selected_measurement:
            return imu_fig, html.Div(orientation_info_children)

        imu_df = data_loader.load_imu_data(experiment_path, selected_sensor, selected_measurement)

        if not imu_df.empty and 'time_from_sync' in imu_df.columns:
            cols_to_plot = [col for col in ['x', 'y', 'z'] if col in imu_df.columns]
            if not cols_to_plot: # Fallback if x,y,z are not present
                cols_to_plot = [col for col in imu_df.select_dtypes(include='number').columns if col not in ['time_from_sync', 't']]

            for col in cols_to_plot:
                imu_fig.add_trace(go.Scatter(x=imu_df['time_from_sync'], y=imu_df[col], mode='lines', name=col))
        else:
            print(f"IMU DataFrame empty or missing 'time_from_sync' for {selected_sensor} - {selected_measurement} in {experiment_path}")


        orientation = orientations.get(selected_sensor.lower())
        if orientation:
            orientation_info_children = [
                html.H5(f"Orientation: {selected_sensor.upper()} ({orientation.get('location', 'N/A')})"),
                html.Ul([
                    html.Li(f"X-axis: {orientation.get('x_direction', 'N/A')}"),
                    html.Li(f"Y-axis: {orientation.get('y_direction', 'N/A')}"),
                    html.Li(f"Z-axis: {orientation.get('z_direction', 'N/A')}")
                ])
            ]
        else:
            orientation_info_children = [html.P(f"Orientation data not found for {selected_sensor}.")]

        return imu_fig, html.Div(orientation_info_children) 
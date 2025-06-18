# dashboard_app/app.py
from dash import Dash
import dash_bootstrap_components as dbc
import layout # Use direct import
import callbacks # Use direct import

# Initialize the Dash app
# If using assets folder, Dash automatically picks up files in it.
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Hovercraft Sensor Dashboard"

# Set the layout
app.layout = layout.create_main_layout() # Use module name

# Register callbacks
callbacks.register_callbacks(app) # Use module name

# Run the server
if __name__ == '__main__':
    # Add a check for DATA_REPO_PATH from config
    import os
    from config import DATA_REPO_PATH
    if not os.path.isdir(DATA_REPO_PATH):
        print(f"**********************************************************************************")
        print(f"ERROR: The DATA_REPO_PATH specified in config.py ('{DATA_REPO_PATH}') was not found.")
        print(f"Please ensure this path is correct and points to your main data experiments folder.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"**********************************************************************************")
    else:
        print(f"Looking for experiments in: {os.path.abspath(DATA_REPO_PATH)}")
        app.run(debug=True) 
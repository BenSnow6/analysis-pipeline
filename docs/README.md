# Hovercraft Data Analysis Pipeline

A data analysis pipeline for processing, analyzing, and visualizing data collected from hovercraft experiments.

## Overview

This repository contains the data and tools for analyzing hovercraft performance based on GPS and IMU sensor readings collected during various test maneuvers. Key components include:

- Raw experimental data stored in `02_Evaluation_Experiments/`.
- A Dash web application for interactive visualization (`hovercraft_data_analysis/dashboard_app/`).
- Supporting Python modules and potentially Jupyter notebooks for specific analyses.

## Data Structure (`02_Evaluation_Experiments/`)

Experimental data is organized within the `02_Evaluation_Experiments/` directory using a nested structure:

```
Category/
└── TimeSlot/
    └── ExperimentRun/
        ├── GPS/
        │   └── *.csv (GPS data)
        └── IMU/
            ├── Sensor_*/
            │   └── *.csv (IMU data for specific sensor)
            └── ...
```

- **Category:** Broad type of experiment (e.g., `1a_1_Minimum_Radius_Turn`).
- **TimeSlot:** Time of day the experiment was run (e.g., `afternoon`, `morning`).
- **ExperimentRun:** Specific instance of the experiment (e.g., `007_Fast_stbd_turn_1`).
- **GPS/IMU:** Subdirectories containing the respective sensor data files in CSV format.

A `sensor_orientations.json` file (if present in the root) defines the orientation offsets for specific sensors used in the analysis.

## Visualization Dashboard (`hovercraft_data_analysis/dashboard_app/`)

A Dash application provides an interactive way to explore the collected data.

**Features:**

- Select experiments based on their folder path.
- View synchronized plots of GPS track and IMU sensor readings (accelerometer, gyroscope, magnetometer, orientation).
- Select specific IMU sensors for plotting.

**Running the App:**

1.  Ensure you have the necessary Python packages installed (primarily `dash`, `plotly`, `pandas`). You might need to create a `requirements.txt` based on imports in the `dashboard_app` files.
2.  Navigate to the repository root in your terminal.
3.  Run the application using:
    ```bash
    python hovercraft_data_analysis/dashboard_app/app.py
    ```
4.  Open your web browser and go to the address provided (usually `http://127.0.0.1:8050/`).

**Application Structure:**

- `app.py`: Main application entry point, defines the Dash app instance.
- `config.py`: Configuration settings (e.g., path to data).
- `data_loader.py`: Functions for finding and loading experiment data.
- `layout.py`: Defines the structure and components of the web interface.
- `callbacks.py`: Contains the Dash callbacks that handle user interactions and update plots.

## Other Components

- `src/`: May contain legacy or supplementary Python source code (check relevance).
- `notebooks/`: May contain legacy or supplementary Jupyter notebooks (check relevance).
- `Experimental setup/`: Contains information about the experimental hardware and setup.
- `experiment_details_real_world_data.ipynb`: Notebook potentially detailing the experiments.

## Getting Started

1.  Clone the repository.
2.  Set up a Python environment and install necessary dependencies (see Visualization Dashboard section).
3.  Run the Dash application to visualize the data.
4.  Explore the data structure and notebooks/source code for deeper analysis if needed.

## Requirements

- Python 3.x
- Jupyter Notebook/Lab
- Data analysis libraries (NumPy, Pandas, Matplotlib, etc.)
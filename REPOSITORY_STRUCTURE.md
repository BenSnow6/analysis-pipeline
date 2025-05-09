# Repository Structure: 01_analysis_pipeline

## File Statistics
Total files: 1472
Files by extension:
  (no extension): 755
  .csv: 673
  .sample: 14
  .ipynb: 9
  .html: 8
  .py: 4
  .pyc: 4
  .json: 2
  .md: 1
  .mp4: 1
  .log: 1

## Experiment Statistics
Experiment types: 7
Test cases: 108
Data files: 669
GPS files: 29
IMU files: 636
Sensor types: Sensor_4, Sensor_3, Sensor_wnb, Sensor_wb, Sensor_5

## Source Files
  - __pycache__
  - classes.py
  - data_processing.py
  - plotting.py

## Notebooks
  - 1a_1.ipynb
  - Afternoon_gps.ipynb

## Directory Structure
```
├── 02_Evaluation_Experiments/
│   ├── 1a_1_Minimum_Radius_Turn/
│       └── 7 test cases with sensor and GPS data
│   ├── 1a_2_Rate_of_Turn_vs_Nosewheel_Steering_Angle/
│       └── 4 test cases with sensor and GPS data
│   ├── 1b_1_Ground_Acceleration_Time_and_Distance/
│       └── 6 test cases with sensor and GPS data
│   ├── 1b_4_Normal_Take_off/
│       └── 3 test cases with sensor and GPS data
│   ├── 1c_1_Normal_Climb_All_Engines_Operating/
│       └── 2 test cases with sensor and GPS data
│   ├── 1d_1_Level_Flight_Acceleration/
│       └── 4 test cases with sensor and GPS data
│   └── 1d_2_Level_Flight_Deceleration/
        └── 1 test cases with sensor and GPS data
├── GPS testing/
│   ├── 0_degrees_out.csv
│   ├── 180_degrees_return.csv
│   ├── 90_degree_left_return.csv
│   ├── 90_degrees_left_out.csv
│   ├── gps_testing_home.ipynb
│   └── interactive_bearing_cog_map.html
├── notebooks/
│   ├── 1a_1.ipynb
│   └── Afternoon_gps.ipynb
├── src/
│   ├── __pycache__
│   ├── classes.py
│   ├── data_processing.py
│   └── plotting.py
├── .gitignore
├── README.md
├── folders.ipynb
└── repo_structure.py
```
# Hovercraft Data Analysis Pipeline - Codebase Analysis Report

*Generated on 12th June, 2025*

## Executive Summary

This repository contains a comprehensive data analysis pipeline for processing and analyzing experimental data collected from hovercraft performance trials. The system is designed to handle multi-sensor data fusion, visualization, and analysis of vehicle dynamics during various test maneuvers.

## Project Overview

### Purpose
The analysis pipeline is part of an Engineering Doctorate (EngD) project focused on hovercraft performance evaluation. It processes synchronized GPS and Inertial Measurement Unit (IMU) data collected during controlled experiments to analyze vehicle dynamics and performance characteristics.

### Domain Context
- **Vehicle Type**: Hovercraft
- **Research Focus**: Performance evaluation, dynamics analysis, maneuvering characteristics
- **Data Sources**: GPS positioning, multi-sensor IMU arrays, environmental conditions
- **Analysis Goals**: Understanding turning dynamics, acceleration profiles, flight characteristics

## Architecture Overview

### Data Flow Architecture
```
Raw Experimental Data → Data Processing → Analysis & Visualization → Interactive Dashboard
```

1. **Data Collection**: Synchronized GPS and IMU sensors during hovercraft operations
2. **Data Storage**: Hierarchical CSV-based storage with structured experiment organization
3. **Data Processing**: Python-based pipeline with lazy loading and multi-sensor fusion
4. **Visualization**: Interactive Dash web application for real-time data exploration
5. **Analysis**: Jupyter notebooks for specialized analysis workflows

### Key Technologies
- **Python**: Core processing language
- **Pandas**: Data manipulation and analysis
- **Dash/Plotly**: Interactive web visualization
- **NumPy**: Numerical computations
- **Jupyter**: Interactive analysis notebooks

## Data Structure Analysis

### Experiment Organization
The data follows a systematic hierarchical structure:

```
02_Evaluation_Experiments/
├── {Experiment_Category}/
│   ├── {Time_Period}/
│   │   └── {Experiment_Run}/
│   │       ├── GPS/
│   │       │   └── GPS_{experiment_name}.csv
│   │       └── IMU/
│   │           ├── Sensor_3/
│   │           ├── Sensor_4/
│   │           ├── Sensor_5/
│   │           ├── Sensor_wb/
│   │           └── Sensor_wnb/
│   │               ├── accel_{experiment_name}.csv
│   │               ├── angle_{experiment_name}.csv
│   │               ├── gyro_{experiment_name}.csv
│   │               ├── mag_{experiment_name}.csv
│   │               └── quat_{experiment_name}.csv (optional)
```

### Experiment Categories Identified
1. **1a_1_Minimum_Radius_Turn**: Turning radius analysis
2. **1a_2_Rate_of_Turn_vs_Nosewheel_Steering_Angle**: Steering response studies
3. **1b_1_Ground_Acceleration_Time_and_Distance**: Linear acceleration profiles
4. **1b_4_Normal_Take_off**: Launch sequence analysis
5. **1c_1_Normal_Climb_All_Engines_Operating**: Flight dynamics
6. **1d_1_Level_Flight_Acceleration**: Cruise performance
7. **1d_2_Level_Flight_Deceleration**: Deceleration characteristics

### Sensor Configuration
The system employs a distributed sensor network:

- **GPS**: High-precision positioning (~1Hz sampling)
- **IMU Sensors** (5 units):
  - **Sensor_3**: Starboard side wall (200Hz)
  - **Sensor_4**: Above hull (200Hz) 
  - **Sensor_5**: Steering wheel location (200Hz)
  - **Sensor_wb**: Center console (100Hz)
  - **Sensor_wnb**: Port side wall (7.5Hz)

Each IMU provides:
- **Accelerometer data** (m/s²): 3-axis linear acceleration
- **Gyroscope data** (°/s): 3-axis angular velocity
- **Magnetometer data** (µT): 3-axis magnetic field
- **Angle data** (°): Roll, pitch, yaw orientations
- **Quaternion data**: Orientation quaternions (some sensors)

### Data Synchronization
All sensors include a `time_from_sync` field enabling precise temporal alignment across different sampling rates and sensor modalities.

## Code Architecture

### Core Components

#### 1. Data Processing Layer (`data_utils.py`)
- **ExperimentData Class**: Central data container with lazy loading
- **Multi-format Support**: Handles various file structures and naming conventions
- **Sensor Abstraction**: Unified interface for different sensor types
- **Lazy Loading**: Memory-efficient data access for large datasets
- **Parallel Processing**: Multi-threaded experiment loading

Key Features:
- Dynamic path discovery for flexible deployment
- Robust error handling and data validation
- Configurable sensor mappings and orientations
- Standardized column naming across sensor types

#### 2. Visualization Dashboard (`dashboard_app.py`)
Interactive web application built with Dash:
- **Real-time Data Exploration**: Select experiments and visualize immediately
- **Multi-sensor Plotting**: Synchronized plots across GPS and IMU data
- **GPS Mapping**: Interactive maps showing vehicle trajectory
- **Time-series Analysis**: Synchronized time-domain plots
- **Sensor Selection**: User-configurable sensor subset visualization

#### 3. Legacy Processing (`src/classes.py`)
Original GPS and IMU processing classes providing:
- Basic data loading and processing
- Sensor orientation transformations
- Derived parameter calculations

#### 4. Configuration Management
- **Sensor Orientations** (`config/sensor_orientations.json`): Physical sensor mounting definitions
- **Sampling Frequencies**: Documented per experiment and sensor
- **Experimental Setup**: Hardware configuration documentation

### Data Processing Pipeline

#### Loading Process
1. **Experiment Discovery**: Automatic detection of available experiments
2. **Structure Detection**: Identification of file organization patterns
3. **Lazy Loading Setup**: File path indexing without immediate data loading
4. **On-demand Processing**: Data loaded only when accessed
5. **Caching**: Loaded data retained for subsequent access

#### Processing Features
- **Automatic Type Conversion**: String data converted to appropriate numeric types
- **Missing Data Handling**: Robust NaN detection and management
- **Time Synchronization**: Common time base across all sensors
- **Coordinate Transformations**: Sensor-to-body frame conversions
- **Data Validation**: Comprehensive checks for data integrity

## Analytical Capabilities

### Current Analysis Types

#### 1. Trajectory Analysis
- GPS path visualization with interactive mapping
- Speed profiles over time
- Altitude variations during maneuvers

#### 2. Vehicle Dynamics
- Multi-axis acceleration analysis
- Angular velocity during turns
- Attitude (roll/pitch/yaw) evolution
- Magnetic field variations (heading reference)

#### 3. Performance Metrics
- Turn radius calculations
- Acceleration/deceleration profiles
- Frequency response analysis (via sampling rate data)
- Sensor correlation studies

#### 4. Experimental Comparison
- Multi-experiment overlay capabilities
- Statistical analysis across test runs
- Performance envelope characterization

### Advanced Features
- **Kalman Filtering**: Evidence of filtering experiments for state estimation
- **Animation Support**: Video generation of trajectories (craft_animation.mp4)
- **Weather Integration**: Environmental condition correlation
- **Frequency Analysis**: Sampling rate optimization studies

## Key Insights from Codebase Analysis

### Strengths
1. **Comprehensive Data Model**: Well-structured representation of complex multi-sensor data
2. **Scalable Architecture**: Lazy loading enables handling of large datasets
3. **Interactive Visualization**: User-friendly dashboard for data exploration
4. **Flexible Configuration**: Adaptable to different sensor setups and experiments
5. **Robust Error Handling**: Graceful degradation with missing or corrupted data
6. **Professional Documentation**: Clear README and structured code organization

### Data Quality Considerations
1. **Variable Sampling Rates**: Different sensors operate at different frequencies
2. **Sensor Reliability**: Some sensors (sensor_wnb) show lower sampling rates
3. **Time Synchronization**: Critical dependency on accurate time alignment
4. **Missing Data**: Some experiments may have incomplete sensor coverage

### Performance Characteristics
- **Memory Efficiency**: Lazy loading prevents memory exhaustion
- **Processing Speed**: Parallel loading for multiple experiments
- **Scalability**: Can handle hundreds of experiment files
- **Interactive Response**: Dashboard optimized for real-time interaction

## Development Observations

### Code Quality
- **Modular Design**: Clear separation of concerns
- **Type Hints**: Modern Python practices with type annotations
- **Error Handling**: Comprehensive exception management
- **Logging**: Integrated logging for debugging and monitoring
- **Testing Infrastructure**: Evidence of validation and testing approaches

### Technical Debt Areas
1. **Dual Processing Systems**: Both legacy (src/classes.py) and modern (data_utils.py) approaches
2. **Configuration Scatter**: Sensor configurations in multiple locations
3. **Documentation Sync**: Some inconsistencies between code and documentation
4. **Testing Coverage**: Limited evidence of comprehensive test suites

## Recommendations for Future Development

### Immediate Improvements
1. **Consolidate Processing**: Migrate fully to the modern data_utils.py approach
2. **Centralize Configuration**: Single source of truth for sensor configurations  
3. **Enhanced Testing**: Comprehensive unit and integration tests
4. **Documentation Update**: Sync all documentation with current implementation

### Advanced Features
1. **Real-time Processing**: Live data ingestion capabilities
2. **Machine Learning Integration**: Automated pattern recognition in vehicle dynamics
3. **Export Capabilities**: Standard format exports for external analysis tools
4. **Collaborative Features**: Multi-user dashboard access and sharing

### Research Extensions
1. **Predictive Modeling**: Vehicle behavior prediction based on historical data
2. **Optimization Studies**: Parameter optimization for performance enhancement
3. **Comparative Analysis**: Benchmarking against theoretical models
4. **Environmental Correlation**: Weather/sea state impact analysis

## Conclusion

This hovercraft data analysis pipeline represents a sophisticated and well-architected solution for multi-sensor vehicle dynamics analysis. The system successfully balances usability, performance, and analytical capability while maintaining flexibility for research applications.

The codebase demonstrates mature software engineering practices with clear abstraction layers, robust error handling, and user-friendly interfaces. The interactive dashboard provides immediate value for researchers and engineers, while the underlying data processing pipeline supports advanced analytical workflows.

The project is well-positioned for continued development and expansion, with a solid foundation that can accommodate additional sensors, new analysis techniques, and enhanced visualization capabilities.

---

*This analysis was conducted through comprehensive code review, data structure examination, and architectural assessment of the hovercraft analysis pipeline repository.*
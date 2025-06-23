# Hovercraft Experiment Catalog

## Overview
This document catalogs all experiments conducted with the Griffon 2000TD hovercraft for simulator validation purposes.

## Experiment Categories

### 1a_1: Minimum Radius Turn
Tests to determine the minimum turning radius capabilities of the hovercraft.

#### Morning Session
- **015_Skirt_shift_turns**: Turns using skirt shift control mechanism

#### Afternoon Session  
- **007_Fast_stbd_turn_1**: Fast starboard turn (first attempt)
- **009_Fast_port_turn_1**: Fast port turn (first attempt)
- **011_Static_stbd_1**: Static starboard turn (first attempt)
- **012_Static_port_1**: Static port turn (first attempt)
- **013_Static_port_2**: Static port turn (second attempt)
- **014_Static_stbd_2**: Static starboard turn (second attempt)

### 1a_2: Rate of Turn vs Nosewheel Steering Angle
Tests to characterize the relationship between steering input and turn rate.

#### Afternoon Session
- **021_Quarter_turn_port**: Quarter (90°) port turn
- **022_Quarter_turn_stbd**: Quarter (90°) starboard turn
- **023_Eigth_turn_port**: Eighth (45°) port turn
- **024_Eigth_turn_stbd**: Eighth (45°) starboard turn

### 1b_1: Ground Acceleration Time and Distance
Tests to measure acceleration performance and distance requirements.

#### Morning Session
- **007_Downwind_max_speed_1**: Maximum speed run with wind (first attempt)
- **008_Into_wind_max_speed**: Maximum speed run against wind
- **009_Downwind_max_speed_2**: Maximum speed run with wind (second attempt)
- **010_Downwind_max_speed_3**: Maximum speed run with wind (third attempt)

#### Afternoon Session
- **016_Straight_cruise_1**: Straight cruise speed test (first run)
- **018_Straight_cruise_2**: Straight cruise speed test (second run)
- **020_Straight_cruise_3**: Straight cruise speed test (third run)

### 1b_4: Normal Take-off
Tests to characterize normal take-off procedures and performance.

#### Morning Session
- **006_Departure**: Initial departure sequence
- **013_Yaw_speed_3**: Yaw speed during take-off (third test)

#### Afternoon Session
- **026_Engine_rpm_sweep**: Engine RPM sweep during take-off

### 1c_1: Normal Climb All Engines Operating
Tests to measure climb performance with all engines operational.

#### Morning Session
- **014_Floating_on_sea_and_takeoff**: From floating on water to take-off
- **016_Plough_in**: Plough-in recovery test

### 1d_1: Level Flight Acceleration
Tests to measure acceleration in level flight conditions.

#### Morning Session
- **007_Downwind_max_speed_1**: (Also used for level flight acceleration)
- **008_Into_wind_max_speed**: (Also used for level flight acceleration)
- **009_Downwind_max_speed_2**: (Also used for level flight acceleration)
- **010_Downwind_max_speed_3**: (Also used for level flight acceleration)

### 1d_2: Level Flight Deceleration
Tests to measure deceleration in level flight conditions.

#### Morning Session
- **013_Yaw_speed_3**: (Also used for deceleration testing)

## Data Structure

Each experiment contains:
- **GPS Data**: Position, speed, bearing, altitude
- **IMU Data**: Multiple sensors (3, 4, 5, wb, wnb) each with:
  - Accelerometer data (x, y, z axes)
  - Gyroscope data (x, y, z axes)
  - Magnetometer data (x, y, z axes)
  - Angle data (roll, pitch, yaw)
  - Quaternion data (where available)

## Sensor Configuration

### IMU Sensors
- **Sensor_3**: Primary hull sensor
- **Sensor_4**: Secondary hull sensor
- **Sensor_5**: Tertiary hull sensor
- **Sensor_wb**: With bag sensor
- **Sensor_wnb**: Without bag sensor

### Coordinate System
- X-axis: Forward
- Y-axis: Port (left)
- Z-axis: Up

## Time Synchronization
All data files include `time_from_sync` column for temporal alignment across sensors.
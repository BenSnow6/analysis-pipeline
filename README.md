# Analysis Pipeline

A data analysis pipeline for processing and analyzing research data collected from various experiments.

## Overview

This repository contains tools and scripts for data collection, processing, and analysis of experimental data, including:

- GPS data processing
- IMU/sensor data analysis
- Data visualization tools
- Jupyter notebooks for analysis workflows

## Project Structure

- `src/`: Python source code including data processing utilities and classes
  - `classes.py`: Core data structures and classes
  - `data_processing.py`: Data processing functionality
  - `plotting.py`: Visualization utilities

- `notebooks/`: Jupyter notebooks for analysis
  - Analysis notebooks for various experiment types

- `02_Evaluation_Experiments/`: Experimental data organized by test type
  - Minimum radius turn tests
  - Acceleration/deceleration tests
  - Climb tests
  - Level flight tests

- `GPS testing/`: GPS calibration and verification data

## Getting Started

To use this repository:

1. Clone the repository
2. Navigate to the notebooks directory to explore analysis examples
3. Use the src modules for your own data processing scripts

## Requirements

- Python 3.x
- Jupyter Notebook/Lab
- Data analysis libraries (NumPy, Pandas, Matplotlib, etc.)
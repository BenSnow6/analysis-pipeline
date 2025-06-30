Hovercraft Analysis Pipeline Documentation
==========================================

Welcome to the Hovercraft Analysis Pipeline documentation. This system provides comprehensive tools for processing and analyzing sensor data from hovercraft experiments.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   architecture
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   configuration_guide
   Experimental setup/experiment_list
   Experimental setup/plotting_requirements

Quick Links
-----------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Overview
--------

The Hovercraft Analysis Pipeline is designed to:

* Process multi-sensor data from hovercraft experiments
* Align and synchronize data from different sensors
* Analyze orientation, timestamps, and RPM
* Provide interactive visualizations through a web dashboard
* Generate comprehensive reports

Key Features
------------

**Data Processing**
   - Automatic sensor data alignment
   - Time synchronization across multiple sensors
   - Support for IMU, GPS, and other sensor types

**Analysis Modules**
   - Orientation analysis for sensor mounting validation
   - Timestamp analysis for data quality assessment
   - RPM estimation from accelerometer/gyroscope data

**Visualization**
   - Interactive web dashboard
   - Real-time data exploration
   - Export capabilities for further analysis

**Developer-Friendly**
   - Clean Python package structure
   - Comprehensive test suite
   - Type hints and documentation
   - CI/CD pipeline

Getting Help
------------

If you encounter any issues:

1. Check the :doc:`getting_started` guide
2. Review the :doc:`architecture` documentation
3. Look at the API reference for detailed function documentation
4. Check the test files for usage examples
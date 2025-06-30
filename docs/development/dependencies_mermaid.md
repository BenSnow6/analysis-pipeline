# Dependency Diagram

```mermaid
graph TD
    subgraph root["root/"]
        analyze_imports_py["analyze_imports.py"]
        basic_timestamp_analysis_py["basic_timestamp_analysis.py"]
        dashboard_app_py["dashboard_app.py"]
        data_sync_py["data_sync.py"]
        data_utils_py["data_utils.py"]
        export_static_experiments_to_csv_py["export_static_experiments_to_csv.py"]
        frame_definitions_py["frame_definitions.py"]
        preprocess_data_py["preprocess_data.py"]
        repo_tree_py["repo_tree.py"]
        run_timestamp_analysis_standalone_py["run_timestamp_analysis_standalone.py"]
        test_timestamp_analysis_py["test_timestamp_analysis.py"]
    end
    subgraph code["code/"]
        code___init___py["__init__.py"]
        code_rpm_estimation_check_wp2_results_py["check_wp2_results.py"]
        code_rpm_estimation_cli_py["cli.py"]
        code_rpm_estimation_create_unit_test_summary_py["create_unit_test_summary.py"]
        code_rpm_estimation_fusion_py["fusion.py"]
        code_rpm_estimation_generate_wp3_test_plots_py["generate_wp3_test_plots.py"]
        code_rpm_estimation_io_py["io.py"]
        code_rpm_estimation_logging_config_py["logging_config.py"]
        code_rpm_estimation_preprocess_py["preprocess.py"]
        code_rpm_estimation_quality_py["quality.py"]
        code_rpm_estimation_run_wp2_tests_py["run_wp2_tests.py"]
        code_rpm_estimation_run_wp3_simple_py["run_wp3_simple.py"]
        code_rpm_estimation_schema_py["schema.py"]
        code_rpm_estimation_spectral_py["spectral.py"]
        code_rpm_estimation_test_wp2_processing_py["test_wp2_processing.py"]
        code_rpm_estimation_test_wp3_run_py["test_wp3_run.py"]
        code_rpm_estimation_test_wp4_integration_py["test_wp4_integration.py"]
        code_rpm_estimation_tracking_py["tracking.py"]
        code_rpm_estimation_validate_wp2_py["validate_wp2.py"]
        code_rpm_estimation_visualize_unit_tests_py["visualize_unit_tests.py"]
        code_rpm_estimation_wp2_process_py["wp2_process.py"]
        code_rpm_estimation_wp3_process_py["wp3_process.py"]
        code_rpm_estimation_wp4_process_py["wp4_process.py"]
        code_rpm_estimation___init___py["__init__.py"]
        code_rpm_estimation_results_wp1_check_parquet_py["check_parquet.py"]
        code_rpm_estimation_results_wp1_run_wp1_py["run_wp1.py"]
        code_rpm_estimation_tests_test_cli_py["test_cli.py"]
        code_rpm_estimation_tests_test_config_py["test_config.py"]
        code_rpm_estimation_tests_test_dataclass_py["test_dataclass.py"]
        code_rpm_estimation_tests_test_fusion_py["test_fusion.py"]
        code_rpm_estimation_tests_test_imports_py["test_imports.py"]
        code_rpm_estimation_tests_test_io_py["test_io.py"]
        code_rpm_estimation_tests_test_preprocessing_py["test_preprocessing.py"]
        code_rpm_estimation_tests_test_quality_py["test_quality.py"]
        code_rpm_estimation_tests_test_schema_py["test_schema.py"]
        code_rpm_estimation_tests_test_spectral_py["test_spectral.py"]
        code_rpm_estimation_tests_test_stft_py["test_stft.py"]
        code_rpm_estimation_tests___init___py["__init__.py"]
    end
    subgraph hovercraft_data_analysis["hovercraft_data_analysis/"]
        hovercraft_data_analysis_process_all_experiments_py["process_all_experiments.py"]
        hovercraft_data_analysis_run_static_orientation_analysis_py["run_static_orientation_analysis.py"]
        hovercraft_data_analysis_run_week1_complete_py["run_week1_complete.py"]
        hovercraft_data_analysis_alignment_analysis_align_py["align.py"]
        hovercraft_data_analysis_alignment_analysis_align_additional_all_py["align_additional_all.py"]
        hovercraft_data_analysis_alignment_analysis_align_additional_data_py["align_additional_data.py"]
        hovercraft_data_analysis_alignment_analysis_analyze_alignment_simple_py["analyze_alignment_simple.py"]
        hovercraft_data_analysis_alignment_analysis_convert_hdf5_compat_py["convert_hdf5_compat.py"]
        hovercraft_data_analysis_alignment_analysis_export_all_to_csv_py["export_all_to_csv.py"]
        hovercraft_data_analysis_alignment_analysis_export_static_to_csv_py["export_static_to_csv.py"]
        hovercraft_data_analysis_alignment_analysis_export_to_csv_py["export_to_csv.py"]
        hovercraft_data_analysis_alignment_analysis_plot_alignment_simple_py["plot_alignment_simple.py"]
        hovercraft_data_analysis_alignment_analysis_run_alignment_py["run_alignment.py"]
        hovercraft_data_analysis_alignment_analysis_test_align_py["test_align.py"]
        hovercraft_data_analysis_alignment_analysis___init___py["__init__.py"]
        hovercraft_data_analysis_dashboard_app_app_py["app.py"]
        hovercraft_data_analysis_dashboard_app_callbacks_py["callbacks.py"]
        hovercraft_data_analysis_dashboard_app_config_py["config.py"]
        hovercraft_data_analysis_dashboard_app_data_loader_py["data_loader.py"]
        hovercraft_data_analysis_dashboard_app_layout_py["layout.py"]
        hovercraft_data_analysis_orientation_analysis_add_gyro_to_csv_py["add_gyro_to_csv.py"]
        hovercraft_data_analysis_orientation_analysis_add_gyro_to_static_experiments_py["add_gyro_to_static_experiments.py"]
        hovercraft_data_analysis_orientation_analysis_analyze_gravity_py["analyze_gravity.py"]
        hovercraft_data_analysis_orientation_analysis_analyze_static_gyro_py["analyze_static_gyro.py"]
        hovercraft_data_analysis_orientation_analysis_analyze_static_gyro_simple_py["analyze_static_gyro_simple.py"]
        hovercraft_data_analysis_orientation_analysis_bias_estimator_py["bias_estimator.py"]
        hovercraft_data_analysis_orientation_analysis_check_gyro_units_py["check_gyro_units.py"]
        hovercraft_data_analysis_orientation_analysis_check_rotation_matrix_py["check_rotation_matrix.py"]
        hovercraft_data_analysis_orientation_analysis_check_sensor3_orientation_py["check_sensor3_orientation.py"]
        hovercraft_data_analysis_orientation_analysis_check_sensor5_orientation_py["check_sensor5_orientation.py"]
        hovercraft_data_analysis_orientation_analysis_debug_orientation_py["debug_orientation.py"]
        hovercraft_data_analysis_orientation_analysis_debug_rotation_validation_py["debug_rotation_validation.py"]
        hovercraft_data_analysis_orientation_analysis_deduce_sensor_orientation_py["deduce_sensor_orientation.py"]
        hovercraft_data_analysis_orientation_analysis_dynamic_validator_py["dynamic_validator.py"]
        hovercraft_data_analysis_orientation_analysis_orientation_check_py["orientation_check.py"]
        hovercraft_data_analysis_orientation_analysis_plot_orientation_py["plot_orientation.py"]
        hovercraft_data_analysis_orientation_analysis_rotation_validator_py["rotation_validator.py"]
        hovercraft_data_analysis_orientation_analysis_run_orientation_py["run_orientation.py"]
        hovercraft_data_analysis_orientation_analysis_static_detector_py["static_detector.py"]
        hovercraft_data_analysis_orientation_analysis_test_fixes_py["test_fixes.py"]
        hovercraft_data_analysis_orientation_analysis_test_orientation_py["test_orientation.py"]
        hovercraft_data_analysis_orientation_analysis_test_orientation_simple_py["test_orientation_simple.py"]
        hovercraft_data_analysis_orientation_analysis_test_unit_conversion_py["test_unit_conversion.py"]
        hovercraft_data_analysis_orientation_analysis___init___py["__init__.py"]
        hovercraft_data_analysis_timestamp_analysis_data_loader_py["data_loader.py"]
        hovercraft_data_analysis_timestamp_analysis_main_py["main.py"]
        hovercraft_data_analysis_timestamp_analysis_report_generator_py["report_generator.py"]
        hovercraft_data_analysis_timestamp_analysis_timestamp_analyzer_py["timestamp_analyzer.py"]
        hovercraft_data_analysis_timestamp_analysis_visualizer_py["visualizer.py"]
        hovercraft_data_analysis_timestamp_analysis___init___py["__init__.py"]
        hovercraft_data_analysis_timestamp_analysis___main___py["__main__.py"]
    end
    subgraph src["src/"]
        src_classes_py["classes.py"]
        src_data_processing_py["data_processing.py"]
        src_plotting_py["plotting.py"]
    end
    subgraph thesis_analysis["thesis_analysis/"]
        thesis_analysis_plotting_experiment_plots_py["experiment_plots.py"]
        thesis_analysis_plotting___init___py["__init__.py"]
        thesis_analysis_scripts_analyze_experiments_py["analyze_experiments.py"]
        thesis_analysis_scripts_simulator_validation_py["simulator_validation.py"]
    end
    %% Import relationships
    dashboard_app_py --> data_utils_py
    data_sync_py --> data_utils_py
    preprocess_data_py --> data_utils_py
    run_timestamp_analysis_standalone_py --> hovercraft_data_analysis_timestamp_analysis_main_py
    code_rpm_estimation_cli_py --> code_rpm_estimation_io_py
    code_rpm_estimation_cli_py --> code_rpm_estimation_logging_config_py
    code_rpm_estimation_cli_py --> code_rpm_estimation_preprocess_py
    code_rpm_estimation_cli_py --> code_rpm_estimation_wp2_process_py
    code_rpm_estimation_cli_py --> code_rpm_estimation_wp3_process_py
    code_rpm_estimation_cli_py --> code_rpm_estimation_wp4_process_py
    code_rpm_estimation_fusion_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_generate_wp3_test_plots_py --> code_rpm_estimation_spectral_py
    code_rpm_estimation_generate_wp3_test_plots_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_io_py --> code_rpm_estimation_logging_config_py
    code_rpm_estimation_preprocess_py --> code_rpm_estimation_io_py
    code_rpm_estimation_preprocess_py --> code_rpm_estimation_logging_config_py
    code_rpm_estimation_preprocess_py --> code_rpm_estimation_quality_py
    code_rpm_estimation_preprocess_py --> code_rpm_estimation_schema_py
    code_rpm_estimation_quality_py --> code_rpm_estimation_logging_config_py
    code_rpm_estimation_run_wp2_tests_py --> code_rpm_estimation_spectral_py
    code_rpm_estimation_run_wp2_tests_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_run_wp2_tests_py --> code_rpm_estimation_wp2_process_py
    code_rpm_estimation_schema_py --> code_rpm_estimation_logging_config_py
    code_rpm_estimation_spectral_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_test_wp4_integration_py --> code_rpm_estimation_fusion_py
    code_rpm_estimation_test_wp4_integration_py --> code_rpm_estimation_io_py
    code_rpm_estimation_test_wp4_integration_py --> code_rpm_estimation_logging_config_py
    code_rpm_estimation_test_wp4_integration_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_test_wp4_integration_py --> code_rpm_estimation_wp4_process_py
    code_rpm_estimation_validate_wp2_py --> code_rpm_estimation_spectral_py
    code_rpm_estimation_visualize_unit_tests_py --> code_rpm_estimation_spectral_py
    code_rpm_estimation_wp2_process_py --> code_rpm_estimation_spectral_py
    code_rpm_estimation_wp2_process_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_wp3_process_py --> code_rpm_estimation_io_py
    code_rpm_estimation_wp3_process_py --> code_rpm_estimation_logging_config_py
    code_rpm_estimation_wp3_process_py --> code_rpm_estimation_quality_py
    code_rpm_estimation_wp3_process_py --> code_rpm_estimation_spectral_py
    code_rpm_estimation_wp3_process_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_wp4_process_py --> code_rpm_estimation_fusion_py
    code_rpm_estimation_wp4_process_py --> code_rpm_estimation_io_py
    code_rpm_estimation_wp4_process_py --> code_rpm_estimation_logging_config_py
    code_rpm_estimation_wp4_process_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation___init___py --> code_rpm_estimation_cli_py
    code_rpm_estimation___init___py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_results_wp1_run_wp1_py --> code_rpm_estimation_cli_py
    code_rpm_estimation_tests_test_cli_py --> code_rpm_estimation_cli_py
    code_rpm_estimation_tests_test_config_py --> code_rpm_estimation_io_py
    code_rpm_estimation_tests_test_dataclass_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_tests_test_fusion_py --> code_rpm_estimation_fusion_py
    code_rpm_estimation_tests_test_fusion_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_tests_test_imports_py --> code_rpm_estimation_cli_py
    code_rpm_estimation_tests_test_imports_py --> code_rpm_estimation_fusion_py
    code_rpm_estimation_tests_test_imports_py --> code_rpm_estimation_io_py
    code_rpm_estimation_tests_test_imports_py --> code_rpm_estimation_preprocess_py
    code_rpm_estimation_tests_test_imports_py --> code_rpm_estimation_spectral_py
    code_rpm_estimation_tests_test_imports_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_tests_test_io_py --> code_rpm_estimation_io_py
    code_rpm_estimation_tests_test_preprocessing_py --> code_rpm_estimation_preprocess_py
    code_rpm_estimation_tests_test_quality_py --> code_rpm_estimation_quality_py
    code_rpm_estimation_tests_test_quality_py --> code_rpm_estimation_quality_py
    code_rpm_estimation_tests_test_schema_py --> code_rpm_estimation_schema_py
    code_rpm_estimation_tests_test_schema_py --> code_rpm_estimation_schema_py
    code_rpm_estimation_tests_test_spectral_py --> code_rpm_estimation_spectral_py
    code_rpm_estimation_tests_test_spectral_py --> code_rpm_estimation_tracking_py
    code_rpm_estimation_tests_test_stft_py --> code_rpm_estimation_quality_py
    code_rpm_estimation_tests_test_stft_py --> code_rpm_estimation_schema_py
    code_rpm_estimation_tests_test_stft_py --> code_rpm_estimation_spectral_py
    code_rpm_estimation_tests_test_stft_py --> code_rpm_estimation_tracking_py
    hovercraft_data_analysis_alignment_analysis_align_additional_all_py --> hovercraft_data_analysis_alignment_analysis_align_additional_data_py
    hovercraft_data_analysis_alignment_analysis_run_alignment_py --> hovercraft_data_analysis_alignment_analysis_align_py
    hovercraft_data_analysis_alignment_analysis_test_align_py --> hovercraft_data_analysis_alignment_analysis_align_py
    hovercraft_data_analysis_alignment_analysis___init___py --> hovercraft_data_analysis_alignment_analysis_align_py
    hovercraft_data_analysis_dashboard_app_app_py --> hovercraft_data_analysis_dashboard_app_callbacks_py
    hovercraft_data_analysis_dashboard_app_app_py --> hovercraft_data_analysis_dashboard_app_config_py
    hovercraft_data_analysis_dashboard_app_app_py --> hovercraft_data_analysis_dashboard_app_layout_py
    hovercraft_data_analysis_dashboard_app_callbacks_py --> hovercraft_data_analysis_dashboard_app_config_py
    hovercraft_data_analysis_dashboard_app_callbacks_py --> hovercraft_data_analysis_dashboard_app_data_loader_py
    hovercraft_data_analysis_dashboard_app_data_loader_py --> hovercraft_data_analysis_dashboard_app_config_py
    hovercraft_data_analysis_dashboard_app_layout_py --> hovercraft_data_analysis_dashboard_app_config_py
    hovercraft_data_analysis_dashboard_app_layout_py --> hovercraft_data_analysis_dashboard_app_data_loader_py
    hovercraft_data_analysis_orientation_analysis_bias_estimator_py --> frame_definitions_py
    hovercraft_data_analysis_orientation_analysis_bias_estimator_py --> hovercraft_data_analysis_orientation_analysis_static_detector_py
    hovercraft_data_analysis_orientation_analysis_check_sensor3_orientation_py --> frame_definitions_py
    hovercraft_data_analysis_orientation_analysis_check_sensor3_orientation_py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis_check_sensor5_orientation_py --> frame_definitions_py
    hovercraft_data_analysis_orientation_analysis_check_sensor5_orientation_py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis_debug_orientation_py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis_debug_orientation_py --> hovercraft_data_analysis_orientation_analysis_static_detector_py
    hovercraft_data_analysis_orientation_analysis_debug_rotation_validation_py --> frame_definitions_py
    hovercraft_data_analysis_orientation_analysis_debug_rotation_validation_py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis_deduce_sensor_orientation_py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis_dynamic_validator_py --> frame_definitions_py
    hovercraft_data_analysis_orientation_analysis_orientation_check_py --> hovercraft_data_analysis_orientation_analysis_bias_estimator_py
    hovercraft_data_analysis_orientation_analysis_orientation_check_py --> hovercraft_data_analysis_orientation_analysis_dynamic_validator_py
    hovercraft_data_analysis_orientation_analysis_orientation_check_py --> frame_definitions_py
    hovercraft_data_analysis_orientation_analysis_orientation_check_py --> hovercraft_data_analysis_orientation_analysis_rotation_validator_py
    hovercraft_data_analysis_orientation_analysis_orientation_check_py --> hovercraft_data_analysis_orientation_analysis_static_detector_py
    hovercraft_data_analysis_orientation_analysis_rotation_validator_py --> frame_definitions_py
    hovercraft_data_analysis_orientation_analysis_rotation_validator_py --> hovercraft_data_analysis_orientation_analysis_static_detector_py
    hovercraft_data_analysis_orientation_analysis_run_orientation_py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis_run_orientation_py --> hovercraft_data_analysis_orientation_analysis_plot_orientation_py
    hovercraft_data_analysis_orientation_analysis_test_fixes_py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis_test_orientation_py --> hovercraft_data_analysis_orientation_analysis_bias_estimator_py
    hovercraft_data_analysis_orientation_analysis_test_orientation_py --> hovercraft_data_analysis_orientation_analysis_dynamic_validator_py
    hovercraft_data_analysis_orientation_analysis_test_orientation_py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis_test_orientation_py --> hovercraft_data_analysis_orientation_analysis_rotation_validator_py
    hovercraft_data_analysis_orientation_analysis_test_orientation_py --> hovercraft_data_analysis_orientation_analysis_static_detector_py
    hovercraft_data_analysis_orientation_analysis_test_unit_conversion_py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis___init___py --> hovercraft_data_analysis_orientation_analysis_bias_estimator_py
    hovercraft_data_analysis_orientation_analysis___init___py --> hovercraft_data_analysis_orientation_analysis_dynamic_validator_py
    hovercraft_data_analysis_orientation_analysis___init___py --> hovercraft_data_analysis_orientation_analysis_orientation_check_py
    hovercraft_data_analysis_orientation_analysis___init___py --> hovercraft_data_analysis_orientation_analysis_plot_orientation_py
    hovercraft_data_analysis_orientation_analysis___init___py --> hovercraft_data_analysis_orientation_analysis_rotation_validator_py
    hovercraft_data_analysis_orientation_analysis___init___py --> hovercraft_data_analysis_orientation_analysis_static_detector_py
    hovercraft_data_analysis_timestamp_analysis_report_generator_py --> code_rpm_estimation_io_py
    hovercraft_data_analysis_timestamp_analysis_report_generator_py --> hovercraft_data_analysis_timestamp_analysis_timestamp_analyzer_py
    hovercraft_data_analysis_timestamp_analysis_report_generator_py --> hovercraft_data_analysis_timestamp_analysis_visualizer_py
    hovercraft_data_analysis_timestamp_analysis_timestamp_analyzer_py --> hovercraft_data_analysis_dashboard_app_data_loader_py
    hovercraft_data_analysis_timestamp_analysis_visualizer_py --> hovercraft_data_analysis_timestamp_analysis_timestamp_analyzer_py
    hovercraft_data_analysis_timestamp_analysis___main___py --> hovercraft_data_analysis_timestamp_analysis_main_py
    src_data_processing_py --> src_classes_py
    thesis_analysis_scripts_analyze_experiments_py --> thesis_analysis_plotting_experiment_plots_py
```

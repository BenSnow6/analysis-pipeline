# src/plotting.py
import matplotlib.pyplot as plt
import os

def plot_real_data(data_structure, variables, experiment_name, save_dir):
    for var in variables:
        # Handle dynamic column names for IMU-derived data
        if var in data_structure.columns:
            plt.figure()
            plt.plot(data_structure['time_from_sync'], data_structure[var], label='Real Data')
            plt.title(f"{experiment_name} - {var} (Real)")
            plt.xlabel("Time from Sync")
            plt.ylabel(f"{var} (units)")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{experiment_name}_{var}_real.png"))
            plt.close()
        else:
            # Check for sensor-specific variables
            sensor_vars = [col for col in data_structure.columns if col.startswith(var + '_')]
            for sensor_var in sensor_vars:
                plt.figure()
                plt.plot(data_structure['time_from_sync'], data_structure[sensor_var], label=f'Real Data - {sensor_var}')
                plt.title(f"{experiment_name} - {sensor_var} (Real)")
                plt.xlabel("Time from Sync")
                plt.ylabel(f"{var} (units)")
                plt.legend()
                plt.savefig(os.path.join(save_dir, f"{experiment_name}_{sensor_var}_real.png"))
                plt.close()

def plot_simulated_data(data_structure, variables, experiment_name, save_dir):
    for var in variables:
        if var in data_structure.columns:
            plt.figure()
            plt.plot(data_structure['time_from_sync'], data_structure[var], label='Simulated Data', color='orange')
            plt.title(f"{experiment_name} - {var} (Simulated)")
            plt.xlabel("Time from Sync")
            plt.ylabel(f"{var} (units)")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{experiment_name}_{var}_simulated.png"))
            plt.close()
        else:
            # Handle sensor-specific simulated data if applicable
            sensor_vars = [col for col in data_structure.columns if col.startswith(var + '_')]
            for sensor_var in sensor_vars:
                plt.figure()
                plt.plot(data_structure['time_from_sync'], data_structure[sensor_var], label=f'Simulated Data - {sensor_var}', color='orange')
                plt.title(f"{experiment_name} - {sensor_var} (Simulated)")
                plt.xlabel("Time from Sync")
                plt.ylabel(f"{var} (units)")
                plt.legend()
                plt.savefig(os.path.join(save_dir, f"{experiment_name}_{sensor_var}_simulated.png"))
                plt.close()

def plot_with_shadow_error(data_structure, variables, percentage, experiment_name, save_dir):
    for var in variables:
        if var in data_structure.columns:
            plt.figure()
            plt.plot(data_structure['time_from_sync'], data_structure[var], label='Real Data')
            error = data_structure[var] * (percentage / 100)
            plt.fill_between(data_structure['time_from_sync'], data_structure[var] - error, data_structure[var] + error, color='gray', alpha=0.2, label=f'±{percentage}%')
            plt.title(f"{experiment_name} - {var} with ±{percentage}% Error")
            plt.xlabel("Time from Sync")
            plt.ylabel(f"{var} (units)")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{experiment_name}_{var}_shadow_error.png"))
            plt.close()
        else:
            # Handle sensor-specific variables
            sensor_vars = [col for col in data_structure.columns if col.startswith(var + '_')]
            for sensor_var in sensor_vars:
                plt.figure()
                plt.plot(data_structure['time_from_sync'], data_structure[sensor_var], label='Real Data')
                error = data_structure[sensor_var] * (percentage / 100)
                plt.fill_between(data_structure['time_from_sync'], data_structure[sensor_var] - error, data_structure[sensor_var] + error, color='gray', alpha=0.2, label=f'±{percentage}%')
                plt.title(f"{experiment_name} - {sensor_var} with ±{percentage}% Error")
                plt.xlabel("Time from Sync")
                plt.ylabel(f"{var} (units)")
                plt.legend()
                plt.savefig(os.path.join(save_dir, f"{experiment_name}_{sensor_var}_shadow_error.png"))
                plt.close()

def plot_comparison(data_real, data_sim, variables, experiment_name, save_dir):
    for var in variables:
        if var in data_real.columns and var in data_sim.columns:
            plt.figure()
            plt.plot(data_real['time_from_sync'], data_real[var], label='Real Data')
            plt.plot(data_sim['time_from_sync'], data_sim[var], label='Simulated Data', linestyle='--')
            plt.title(f"{experiment_name} - {var} Comparison")
            plt.xlabel("Time from Sync")
            plt.ylabel(f"{var} (units)")
            plt.legend()
            plt.savefig(os.path.join(save_dir, f"{experiment_name}_{var}_comparison.png"))
            plt.close()
        else:
            # Handle sensor-specific comparison
            real_sensor_vars = [col for col in data_real.columns if col.startswith(var + '_')]
            sim_sensor_vars = [col for col in data_sim.columns if col.startswith(var + '_')]
            for real_var, sim_var in zip(real_sensor_vars, sim_sensor_vars):
                plt.figure()
                plt.plot(data_real['time_from_sync'], data_real[real_var], label='Real Data')
                plt.plot(data_sim['time_from_sync'], data_sim[sim_var], label='Simulated Data', linestyle='--')
                plt.title(f"{experiment_name} - {real_var} vs {sim_var} Comparison")
                plt.xlabel("Time from Sync")
                plt.ylabel(f"{var} (units)")
                plt.legend()
                plt.savefig(os.path.join(save_dir, f"{experiment_name}_{real_var}_{sim_var}_comparison.png"))
                plt.close()

def custom_plot(data_structure, x, y, experiment_name, plot_info, save_dir, zoom=None):
    plt.figure()
    plt.plot(data_structure[x], data_structure[y], label=f"{y} vs {x}")
    if zoom:
        plt.xlim(zoom['x_min'], zoom['x_max'])
        plt.ylim(zoom['y_min'], zoom['y_max'])
    plt.title(f"{experiment_name} - {plot_info}")
    plt.xlabel(f"{x} (units)")
    plt.ylabel(f"{y} (units)")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_{plot_info}.png"))
    plt.close()

def save_plot(fig, experiment_name, plot_info, save_dir):
    filename = f"{experiment_name}_{plot_info}.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

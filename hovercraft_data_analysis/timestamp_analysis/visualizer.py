"""
Visualization functions for timestamp analysis.

This module provides functions to create diagnostic plots for
timestamp jitter, gaps, and cross-sensor alignment.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import seaborn as sns

from .timestamp_analyzer import TimestampAnalysisResult

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_figure_with_subplots(num_subplots: int, 
                               figsize: Tuple[float, float] = (12, 8)) -> Tuple:
    """Create figure with specified number of subplots."""
    if num_subplots == 1:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        return fig, [ax]
    else:
        rows = (num_subplots + 1) // 2
        cols = min(2, num_subplots)
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0], figsize[1] * rows / 2))
        return fig, axes.flatten() if num_subplots > 1 else [axes]


def plot_timestamp_intervals(result: TimestampAnalysisResult, ax: plt.Axes) -> None:
    """
    Plot timestamp intervals over time.
    
    Args:
        result: Analysis result for a sensor
        ax: Matplotlib axes to plot on
    """
    if len(result.intervals) == 0:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{result.sensor_name} - Timestamp Intervals')
        return
    
    # Create time axis (using midpoints between timestamps)
    time_axis = result.timestamps[:-1] + np.diff(result.timestamps) / 2
    
    # Plot intervals
    ax.plot(time_axis, result.intervals, 'b-', alpha=0.7, linewidth=1, label='Actual')
    
    # Plot expected interval as horizontal line
    ax.axhline(y=result.expected_interval_ms, color='g', linestyle='--', 
               linewidth=2, label=f'Expected ({result.expected_rate_hz:.0f}Hz)')
    
    # Plot jitter threshold bounds
    ax.axhline(y=result.expected_interval_ms + result.jitter_threshold_ms, 
               color='r', linestyle=':', alpha=0.5, label='Jitter threshold')
    ax.axhline(y=result.expected_interval_ms - result.jitter_threshold_ms, 
               color='r', linestyle=':', alpha=0.5)
    
    # Highlight gaps
    for gap in result.gaps:
        ax.axvspan(gap['start_time'], gap['end_time'], 
                  alpha=0.3, color='red', label='Gap' if gap == result.gaps[0] else '')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Interval (ms)')
    ax.set_title(f'{result.sensor_name} - Timestamp Intervals\n'
                f'Rate: {result.actual_rate_hz:.1f}Hz (expected {result.expected_rate_hz}Hz)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_jitter_histogram(result: TimestampAnalysisResult, ax: plt.Axes) -> None:
    """
    Plot histogram of timestamp jitter.
    
    Args:
        result: Analysis result for a sensor
        ax: Matplotlib axes to plot on
    """
    if len(result.jitter) == 0:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{result.sensor_name} - Jitter Distribution')
        return
    
    # Create histogram
    bins = np.linspace(0, max(result.max_jitter_ms, result.jitter_threshold_ms * 1.5), 50)
    ax.hist(result.jitter, bins=bins, alpha=0.7, density=True, edgecolor='black')
    
    # Add vertical line for threshold
    ax.axvline(x=result.jitter_threshold_ms, color='r', linestyle='--', 
               linewidth=2, label=f'Threshold ({result.jitter_threshold_ms}ms)')
    
    # Add statistics text
    stats_text = (f'Mean: {result.mean_jitter_ms:.2f}ms\n'
                 f'Std: {result.std_jitter_ms:.2f}ms\n'
                 f'Max: {result.max_jitter_ms:.2f}ms\n'
                 f'Violations: {result.jitter_violations}')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Jitter (ms)')
    ax.set_ylabel('Density')
    ax.set_title(f'{result.sensor_name} - Jitter Distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_timeline_with_gaps(result: TimestampAnalysisResult, ax: plt.Axes) -> None:
    """
    Plot timeline view showing data coverage and gaps.
    
    Args:
        result: Analysis result for a sensor
        ax: Matplotlib axes to plot on
    """
    if len(result.timestamps) == 0:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{result.sensor_name} - Data Timeline')
        return
    
    # Create timeline segments
    y_pos = 0
    segment_height = 0.8
    
    # Plot continuous data segments
    if len(result.gaps) == 0:
        # No gaps, single continuous segment
        ax.barh(y_pos, result.duration_seconds, height=segment_height, 
                left=0, color='green', alpha=0.7, label='Data')
    else:
        # Plot segments between gaps
        start_time = 0
        for i, gap in enumerate(result.gaps):
            # Plot segment before gap
            segment_duration = gap['start_time'] - start_time
            if segment_duration > 0:
                ax.barh(y_pos, segment_duration, height=segment_height,
                       left=start_time, color='green', alpha=0.7,
                       label='Data' if i == 0 else '')
            
            # Plot gap
            ax.barh(y_pos, gap['duration_ms'] / 1000, height=segment_height,
                   left=gap['start_time'], color='red', alpha=0.7,
                   label='Gap' if i == 0 else '')
            
            start_time = gap['end_time']
        
        # Plot final segment after last gap
        final_duration = result.duration_seconds - start_time
        if final_duration > 0:
            ax.barh(y_pos, final_duration, height=segment_height,
                   left=start_time, color='green', alpha=0.7)
    
    # Add gap annotations
    for gap in result.gaps[:5]:  # Limit to first 5 gaps to avoid clutter
        ax.text(gap['start_time'] + gap['duration_ms'] / 2000, y_pos,
               f"{gap['duration_ms']:.0f}ms", 
               ha='center', va='center', fontsize=8)
    
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, result.duration_seconds)
    ax.set_xlabel('Time (seconds)')
    ax.set_yticks([y_pos])
    ax.set_yticklabels([result.sensor_name])
    ax.set_title(f'{result.sensor_name} - Data Timeline\n'
                f'{result.num_gaps} gaps detected, '
                f'{result.num_samples} samples over {result.duration_seconds:.1f}s')
    ax.legend(loc='best')
    ax.grid(True, axis='x', alpha=0.3)


def plot_cross_sensor_alignment(results: Dict[str, TimestampAnalysisResult],
                               ax: plt.Axes) -> None:
    """
    Plot alignment comparison between sensors.
    
    Args:
        results: Dictionary of analysis results
        ax: Matplotlib axes to plot on
    """
    if len(results) < 2:
        ax.text(0.5, 0.5, 'Need at least 2 sensors for alignment comparison', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cross-Sensor Alignment')
        return
    
    # Prepare data for plotting
    sensor_names = []
    start_times = []
    end_times = []
    durations = []
    
    for name, result in results.items():
        if len(result.timestamps) > 0:
            sensor_names.append(name)
            start_times.append(result.timestamps[0])
            end_times.append(result.timestamps[-1])
            durations.append(result.duration_seconds)
    
    if not sensor_names:
        ax.text(0.5, 0.5, 'No valid timestamp data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cross-Sensor Alignment')
        return
    
    # Normalize times to start from earliest sensor
    min_start = min(start_times)
    start_offsets = [t - min_start for t in start_times]
    
    # Create horizontal bar chart
    y_positions = range(len(sensor_names))
    
    # Plot bars
    bars = ax.barh(y_positions, durations, left=start_offsets, 
                   height=0.6, alpha=0.7)
    
    # Color bars by sensor type
    colors = ['blue' if 'gps' in name.lower() else 'green' for name in sensor_names]
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add start time annotations
    for i, (name, offset) in enumerate(zip(sensor_names, start_offsets)):
        if offset > 0.01:  # Only show if offset is significant
            ax.text(offset - 0.1, i, f'+{offset:.3f}s', 
                   ha='right', va='center', fontsize=8)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(sensor_names)
    ax.set_xlabel('Time (seconds from earliest start)')
    ax.set_title('Cross-Sensor Time Alignment')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.7, label='GPS'),
                      Patch(facecolor='green', alpha=0.7, label='IMU')]
    ax.legend(handles=legend_elements, loc='best')


def create_sensor_report_plots(result: TimestampAnalysisResult,
                              output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Create comprehensive plots for a single sensor.
    
    Args:
        result: Analysis result for a sensor
        output_dir: Directory to save plots (if None, displays instead)
        
    Returns:
        Path to saved figure if output_dir provided, None otherwise
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Timestamp intervals plot
    ax1 = fig.add_subplot(gs[0, :])
    plot_timestamp_intervals(result, ax1)
    
    # Jitter histogram
    ax2 = fig.add_subplot(gs[1, 0])
    plot_jitter_histogram(result, ax2)
    
    # Timeline with gaps
    ax3 = fig.add_subplot(gs[1, 1])
    plot_timeline_with_gaps(result, ax3)
    
    # Add overall title
    status = "✓ PASS" if result.within_spec else "✗ FAIL"
    fig.suptitle(f'Timestamp Analysis: {result.sensor_name} {status}', 
                fontsize=16, fontweight='bold')
    
    # Save or show
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f'timestamp_analysis_{result.sensor_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return filename
    else:
        plt.tight_layout()
        plt.show()
        return None


def create_experiment_summary_plots(results: Dict[str, TimestampAnalysisResult],
                                   experiment_name: str,
                                   output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Create summary plots for all sensors in an experiment.
    
    Args:
        results: Dictionary of analysis results
        experiment_name: Name of the experiment
        output_dir: Directory to save plots
        
    Returns:
        Path to saved figure if output_dir provided, None otherwise
    """
    num_sensors = len(results)
    if num_sensors == 0:
        return None
    
    # Create figure with appropriate number of subplots - increased spacing
    rows = ((num_sensors + 1) // 2 + 1)
    fig = plt.figure(figsize=(18, 5 * rows))
    gs = fig.add_gridspec(rows, 2, hspace=0.5, wspace=0.4)
    
    # Plot each sensor's intervals
    for i, (name, result) in enumerate(results.items()):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        plot_timestamp_intervals(result, ax)
    
    # Add cross-sensor alignment plot at the bottom
    ax_align = fig.add_subplot(gs[-1, :])
    plot_cross_sensor_alignment(results, ax_align)
    
    # Overall title
    num_pass = sum(1 for r in results.values() if r.within_spec)
    fig.suptitle(f'Timestamp Analysis Summary: {experiment_name}\n'
                f'{num_pass}/{num_sensors} sensors within specification',
                fontsize=16, fontweight='bold')
    
    # Save or show
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f'timestamp_analysis_summary_{experiment_name.replace("/", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return filename
    else:
        # Use constrained layout instead of tight_layout for better results
        fig.set_constrained_layout(True)
        plt.show()
        return None
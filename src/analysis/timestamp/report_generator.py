"""
Report generation for timestamp analysis.

This module creates comprehensive HTML reports with analysis results,
statistics, and embedded visualizations.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import base64
from io import BytesIO
import matplotlib.pyplot as plt

from .timestamp_analyzer import TimestampAnalysisResult
from .visualizer import create_sensor_report_plots, create_experiment_summary_plots


def encode_figure_to_base64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure to base64 for HTML embedding."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return image_base64


def generate_sensor_table_row(result: TimestampAnalysisResult) -> str:
    """Generate HTML table row for a sensor's results."""
    status_icon = "✓" if result.within_spec else "✗"
    status_class = "pass" if result.within_spec else "fail"
    
    row = f"""
    <tr class="{status_class}">
        <td>{result.sensor_name}</td>
        <td>{status_icon}</td>
        <td>{result.expected_rate_hz:.0f}</td>
        <td>{result.actual_rate_hz:.1f}</td>
        <td>{result.rate_deviation_percent:.1f}%</td>
        <td>{result.mean_jitter_ms:.2f}</td>
        <td>{result.max_jitter_ms:.2f}</td>
        <td>{result.jitter_threshold_ms:.0f}</td>
        <td>{result.jitter_violations}</td>
        <td>{result.num_gaps}</td>
        <td>{result.num_samples}</td>
        <td>{result.duration_seconds:.1f}</td>
    </tr>
    """
    return row


def generate_issues_section(result: TimestampAnalysisResult) -> str:
    """Generate HTML for issues/warnings section."""
    if result.within_spec and not result.issues:
        return ""
    
    issues_html = f"<h4>{result.sensor_name} Issues:</h4>\n<ul>\n"
    for issue in result.issues:
        issues_html += f"  <li>{issue}</li>\n"
    issues_html += "</ul>\n"
    return issues_html


def generate_gap_details(result: TimestampAnalysisResult) -> str:
    """Generate HTML for gap details section."""
    if result.num_gaps == 0:
        return ""
    
    gaps_html = f"""
    <h4>{result.sensor_name} Gap Details:</h4>
    <table class="gap-table">
        <tr>
            <th>Gap #</th>
            <th>Start Time (s)</th>
            <th>Duration (ms)</th>
            <th>Samples Before</th>
            <th>Samples After</th>
        </tr>
    """
    
    for i, gap in enumerate(result.gaps[:10]):  # Limit to first 10 gaps
        gaps_html += f"""
        <tr>
            <td>{i+1}</td>
            <td>{gap['start_time']:.3f}</td>
            <td>{gap['duration_ms']:.1f}</td>
            <td>{gap['samples_before']}</td>
            <td>{gap['samples_after']}</td>
        </tr>
        """
    
    if result.num_gaps > 10:
        gaps_html += f"""
        <tr>
            <td colspan="5">... and {result.num_gaps - 10} more gaps</td>
        </tr>
        """
    
    gaps_html += "</table>\n"
    return gaps_html


def generate_html_report(results: Dict[str, TimestampAnalysisResult],
                        experiment_name: str,
                        alignment_info: Optional[Dict] = None,
                        include_plots: bool = True) -> str:
    """
    Generate complete HTML report for timestamp analysis.
    
    Args:
        results: Dictionary of analysis results
        experiment_name: Name of the experiment
        alignment_info: Cross-sensor alignment information
        include_plots: Whether to include embedded plots
        
    Returns:
        HTML string for the report
    """
    # Generate timestamp
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Count passing sensors
    num_sensors = len(results)
    num_pass = sum(1 for r in results.values() if r.within_spec)
    overall_status = "PASS" if num_pass == num_sensors else "FAIL"
    overall_class = "pass" if num_pass == num_sensors else "fail"
    
    # Start HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Timestamp Analysis Report - {experiment_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #333;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .summary {{
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .status-pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .status-fail {{
            color: #dc3545;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background-color: #333;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        tr.pass {{
            background-color: #d4edda;
        }}
        tr.fail {{
            background-color: #f8d7da;
        }}
        .issues {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .plot-container {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
        }}
        .gap-table {{
            width: auto;
            margin: 10px 0;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Timestamp Analysis Report</h1>
        <h2>{experiment_name}</h2>
        <p>Generated: {report_time}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Overall Status: <span class="status-{overall_class}">{overall_status}</span></p>
        <p>Sensors Analyzed: {num_sensors}</p>
        <p>Sensors Within Specification: {num_pass}/{num_sensors}</p>
    </div>
    
    <h2>Sensor Analysis Results</h2>
    <table>
        <tr>
            <th>Sensor</th>
            <th>Status</th>
            <th>Expected Rate (Hz)</th>
            <th>Actual Rate (Hz)</th>
            <th>Rate Deviation</th>
            <th>Mean Jitter (ms)</th>
            <th>Max Jitter (ms)</th>
            <th>Jitter Threshold (ms)</th>
            <th>Jitter Violations</th>
            <th>Gaps Detected</th>
            <th>Samples</th>
            <th>Duration (s)</th>
        </tr>
"""
    
    # Add table rows for each sensor
    for sensor_name in sorted(results.keys()):
        html += generate_sensor_table_row(results[sensor_name])
    
    html += "    </table>\n"
    
    # Add cross-sensor alignment info if available
    if alignment_info and alignment_info['sensor_pairs']:
        html += """
    <h2>Cross-Sensor Alignment</h2>
    <div class="summary">
"""
        html += f"<p>Reference Sensor: {alignment_info['reference_sensor']}</p>\n"
        html += f"<p>Maximum Time Offset: {alignment_info['max_offset_ms']:.1f}ms</p>\n"
        
        if alignment_info['max_offset_ms'] > 100:
            html += '<p class="status-fail">⚠️ Warning: Large time offsets detected between sensors</p>\n'
        
        html += "</div>\n"
    
    # Add issues section
    issues_exist = any(not r.within_spec for r in results.values())
    if issues_exist:
        html += '<div class="issues">\n<h2>Issues and Warnings</h2>\n'
        for sensor_name in sorted(results.keys()):
            html += generate_issues_section(results[sensor_name])
        html += '</div>\n'
    
    # Add gap details for sensors with gaps
    gaps_exist = any(r.num_gaps > 0 for r in results.values())
    if gaps_exist:
        html += '<h2>Gap Details</h2>\n'
        for sensor_name in sorted(results.keys()):
            if results[sensor_name].num_gaps > 0:
                html += generate_gap_details(results[sensor_name])
    
    # Add plots if requested
    if include_plots:
        html += '<h2>Diagnostic Plots</h2>\n'
        
        # Create summary plot
        fig = plt.figure(figsize=(16, 10))
        from . import visualizer
        
        # Create a temporary summary plot
        summary_path = create_experiment_summary_plots(
            results, experiment_name, output_dir=None
        )
        
        # Instead of saving to file, we'll recreate it for embedding
        # This is a bit redundant but ensures we have the figure object
        num_sensors = len(results)
        if num_sensors > 0:
            fig = plt.figure(figsize=(16, 4 * ((num_sensors + 1) // 2 + 1)))
            gs = fig.add_gridspec(((num_sensors + 1) // 2 + 1), 2, hspace=0.4, wspace=0.3)
            
            for i, (name, result) in enumerate(results.items()):
                row = i // 2
                col = i % 2
                ax = fig.add_subplot(gs[row, col])
                visualizer.plot_timestamp_intervals(result, ax)
            
            ax_align = fig.add_subplot(gs[-1, :])
            visualizer.plot_cross_sensor_alignment(results, ax_align)
            
            num_pass = sum(1 for r in results.values() if r.within_spec)
            fig.suptitle(f'Timestamp Analysis Summary: {experiment_name}\n'
                        f'{num_pass}/{num_sensors} sensors within specification',
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Encode and embed
            img_data = encode_figure_to_base64(fig)
            html += f'''
        <div class="plot-container">
            <h3>Summary Plot</h3>
            <img src="data:image/png;base64,{img_data}" alt="Summary Plot">
        </div>
'''
    
    # Footer
    html += """
    <div class="footer">
        <p>Timestamp Analysis Tool v1.0.0</p>
        <p>Part of Hovercraft Data Analysis Pipeline</p>
    </div>
</body>
</html>
"""
    
    return html


def save_html_report(html_content: str, output_path: Path) -> Path:
    """
    Save HTML report to file.
    
    Args:
        html_content: HTML string
        output_path: Path to save the report
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def generate_summary_csv(results: Dict[str, TimestampAnalysisResult],
                        output_path: Path) -> Path:
    """
    Generate CSV summary of results for easy import to other tools.
    
    Args:
        results: Dictionary of analysis results
        output_path: Path to save CSV file
        
    Returns:
        Path to saved file
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = [
        'sensor_name', 'within_spec', 'expected_rate_hz', 'actual_rate_hz',
        'rate_deviation_percent', 'mean_jitter_ms', 'max_jitter_ms',
        'jitter_threshold_ms', 'jitter_violations', 'num_gaps',
        'num_samples', 'duration_seconds'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for sensor_name in sorted(results.keys()):
            result = results[sensor_name]
            writer.writerow({
                'sensor_name': result.sensor_name,
                'within_spec': result.within_spec,
                'expected_rate_hz': result.expected_rate_hz,
                'actual_rate_hz': round(result.actual_rate_hz, 2),
                'rate_deviation_percent': round(result.rate_deviation_percent, 2),
                'mean_jitter_ms': round(result.mean_jitter_ms, 3),
                'max_jitter_ms': round(result.max_jitter_ms, 3),
                'jitter_threshold_ms': result.jitter_threshold_ms,
                'jitter_violations': result.jitter_violations,
                'num_gaps': result.num_gaps,
                'num_samples': result.num_samples,
                'duration_seconds': round(result.duration_seconds, 1)
            })
    
    return output_path
"""
Test script to process a single experiment with WP-2.
This demonstrates the functionality on real data.
"""

import sys
from pathlib import Path
import subprocess

# Test experiments to process
test_experiments = [
    ("007_Fast_stbd_turn_1", "afternoon"),  # Dynamic maneuver
    ("011_Static_stbd_1", "afternoon"),      # Static test - should show idle RPM
    ("016_Straight_cruise_1", "afternoon"),  # Straight cruise
]

def main():
    print("WP-2 Test Processing")
    print("=" * 50)
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Process each test experiment
    for exp_name, session in test_experiments:
        print(f"\nProcessing {exp_name} ({session})...")
        
        # Build command
        cmd = [
            sys.executable,
            str(script_dir / "wp2_process.py"),
            "--experiment", exp_name,
            "--session", session,
            "--log-level", "INFO"
        ]
        
        # Run the processing
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Successfully processed {exp_name}")
                # Extract key results from output
                for line in result.stdout.split('\n'):
                    if 'Mean RPM:' in line or 'Availability:' in line:
                        print(f"  {line.strip()}")
            else:
                print(f"✗ Failed to process {exp_name}")
                print(f"  Error: {result.stderr}")
                
        except Exception as e:
            print(f"✗ Error processing {exp_name}: {e}")
    
    print("\n" + "=" * 50)
    print("Test processing complete!")
    print("\nCheck the following directories for results:")
    print("- results/wp2/afternoon/  (HDF5 files)")
    print("- results/wp2/plots/afternoon/  (diagnostic plots)")


if __name__ == "__main__":
    main()
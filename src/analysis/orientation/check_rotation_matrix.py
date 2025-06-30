#!/usr/bin/env python3
"""
Check what the rotation matrix should produce for gravity.
"""

import numpy as np

def check_rotation_expectation():
    """Check what gravity should look like in sensor frame."""
    
    # Expected gravity in body frame (pointing down)
    gravity_body = np.array([0.0, 0.0, 9.80665])  # m/s²
    
    # But wait - if data is in g's, let's use g's
    gravity_body_g = np.array([0.0, 0.0, 1.0])  # g
    
    # Recreate rotation matrix for Sensor_3
    # From the test output, we saw:
    # Generated R_bs for Sensor_3:
    # [[ 0  1  0]
    #  [ 0  0 -1]
    #  [-1  0  0]]
    R_bs = np.array([[0, 1, 0],
                     [0, 0, -1],
                     [-1, 0, 0]])
    print("Rotation matrix R_bs for Sensor_3:")
    print(R_bs)
    
    # Transform gravity from body to sensor frame
    # If R_bs is body-to-sensor: g_sensor = R_bs @ g_body
    # If R_bs is sensor-to-body: g_body = R_bs @ g_sensor, so g_sensor = R_bs.T @ g_body
    
    print("\n=== If R_bs is body-to-sensor ===")
    gravity_sensor_1 = R_bs @ gravity_body_g
    print(f"Expected gravity in sensor frame: {gravity_sensor_1}")
    
    print("\n=== If R_bs is sensor-to-body ===")
    gravity_sensor_2 = R_bs.T @ gravity_body_g
    print(f"Expected gravity in sensor frame: {gravity_sensor_2}")
    
    print("\n=== Actual measured gravity ===")
    print("Measured: [-0.004, -1.016, 0.087] g")
    
    print("\n=== Analysis ===")
    measured = np.array([-0.004, -1.016, 0.087])
    
    # Check which interpretation matches better
    error1 = np.linalg.norm(gravity_sensor_1 - measured)
    error2 = np.linalg.norm(gravity_sensor_2 - measured)
    
    print(f"Error if R_bs is body-to-sensor: {error1:.3f}")
    print(f"Error if R_bs is sensor-to-body: {error2:.3f}")
    
    # Check if we need to flip the sign
    print("\n=== Sign flip check ===")
    error1_neg = np.linalg.norm(-gravity_sensor_1 - measured)
    error2_neg = np.linalg.norm(-gravity_sensor_2 - measured)
    
    print(f"Error if R_bs is body-to-sensor (negated): {error1_neg:.3f}")
    print(f"Error if R_bs is sensor-to-body (negated): {error2_neg:.3f}")
    
    # The sensor configuration says:
    # X: Upward, Y: Forward, Z: Port
    # In body frame: X: Forward, Y: Starboard, Z: Down
    # So if gravity is down in body (+Z), it should be:
    # - If sensor X is body Z (up), then gravity should be -X in sensor
    # - If sensor Y is body X (forward), then no gravity component
    # - If sensor Z is body -Y (port = -starboard), then no gravity component
    
    print("\n=== Manual analysis ===")
    print("Sensor_3 axes mapping:")
    print("  Sensor X points Upward (body -Z)")
    print("  Sensor Y points Forward (body X)")
    print("  Sensor Z points Port (body -Y)")
    print("\nSo gravity (body +Z) should map to sensor -X")
    print("But we're seeing gravity in sensor -Y!")
    
    print("\n=== Conclusion ===")
    print("The rotation matrix or sensor mounting is incorrect.")
    print("The sensor seems to be mounted differently than expected.")

if __name__ == "__main__":
    check_rotation_expectation()
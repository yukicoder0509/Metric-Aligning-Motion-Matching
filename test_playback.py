#!/usr/bin/env python3
"""
Simple test script to verify individual motion playback functions
"""

import numpy as np
import synthetic_data
from synthetic_data import motion_playback_3d, motion_playback_1d

def test_individual_playback():
    """Test individual motion playback functions"""
    print("=== Testing Individual Motion Playback ===")
    
    # Generate test data
    X, Y = synthetic_data.generate_synthetic_data()
    print(f"Generated X shape: {X.shape}, Y shape: {Y.shape}")
    
    # Test 3D motion playback for X
    print("\nTesting 3D Motion Playback for X...")
    print("Close the animation window to continue...")
    ani_x = motion_playback_3d(X, "Test: Original Motion X")
    
    # Test 1D signal playback for Y  
    print("\nTesting 1D Signal Playback for Y...")
    print("Close the animation window to continue...")
    ani_y = motion_playback_1d(Y, "Test: Control Signal Y")
    
    print("\n=== Playback Tests Completed ===")
    
    return ani_x, ani_y

if __name__ == "__main__":
    test_individual_playback()
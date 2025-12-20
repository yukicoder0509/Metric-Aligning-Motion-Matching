import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import solver
import synthetic_data
import patches
import distance
from synthetic_data import motion_playback_3d, motion_playback_1d

# Test the solver with synthetic data generated from synthetic_data.py
def test_solver():
    print("=== Testing Solver ===")
    
    # 1. 準備數據
    X, Y = synthetic_data.generate_synthetic_data()
    # 為了測試，我們讓 Y = X (完全一樣的動作)
    Y_same = X.copy()
    
    # 提取 Patches
    X_patches = patches.extract_patches(X)
    Y_patches = patches.extract_patches(Y_same)
    
    # 計算距離矩陣
    D_X = distance.compute_normalized_distance(X_patches)
    D_Y = distance.compute_normalized_distance(Y_patches)
    
    print(f"Input Shapes: D_X {D_X.shape}, D_Y {D_Y.shape}")
    
    # 2. 執行 Solver
    # 這裡我們設 alpha=0.5 (混合權重), rho=100 (接近 Hard constraint)
    T = solver.solve_fsugw(D_X, D_Y, X_patches, Y_patches, 
                           alpha=0.5, rho=100.0, epsilon=0.01, num_iters=20)
    
    print(f"Output T shape: {T.shape}")
    
    # 3. 驗證與視覺化
    # 檢查每一列的總和是否為 1/N_y (Hard constraint check)
    row_sums = T.sum(axis=1)
    expected_sum = 1.0 / T.shape[0]
    error = np.mean(np.abs(row_sums - expected_sum))
    print(f"Marginal Constraint Error (Row sums): {error:.6f}")
    
    # # 視覺化 T 矩陣
    # plt.figure(figsize=(8, 6))
    # plt.imshow(T, cmap='hot', interpolation='nearest')
    # plt.colorbar(label='Probability Mass')
    # plt.title("Transport Plan T (Should be diagonal-ish for identical inputs)")
    # plt.xlabel("Original Motion Indices (X)")
    # plt.ylabel("Control Sequence Indices (Y)")
    # plt.show()
    
    # # 自動化判定
    # is_diagonal_dominant = np.all(np.argmax(T, axis=1) == np.arange(T.shape[0]))
    # # 注意：因為只有完全一樣長度才會有完美的對角線，
    # # 但如果長度一樣，最大值應該主要落在對角線上。
    # print(f"Is diagonal structure visible? (Visual check recommended)")

    X_prime = solver.blend_patches(T, X_patches)

    # playback original and aligned motions
    print(f"Original motion shape: {X.shape}")
    print(f"Aligned motion shape: {X_prime.shape}")
    
    # Analyze motion differences
    analyze_motion_differences(X, X_prime, Y_same)
    
    # Show individual motion playbacks
    show_individual_motions(X, Y, X_prime)
    
    # Create side-by-side comparison
    print("\n4. Playing Side-by-Side Comparison (X vs X')...")
    animate_comparison(X, X_prime)
    
    # Create triple comparison  
    print("\n5. Playing Triple Comparison (X, Y, X')...")
    create_triple_comparison(X, Y, X_prime)

def animate_comparison(X_original, X_aligned):
    """
    Animate original motion and aligned motion side by side for comparison
    """
    # Determine the minimum length for synchronization
    min_length = min(len(X_original), len(X_aligned))
    
    fig = plt.figure(figsize=(16, 6))

    # === Left Plot: Original Motion ===
    ax_orig = fig.add_subplot(1, 2, 1, projection='3d')
    ax_orig.set_title("Original Motion", fontsize=14)
    ax_orig.set_xlim(-1.5, 1.5)
    ax_orig.set_ylim(-1.5, 1.5)
    ax_orig.set_zlim(-1.5, 1.5)
    ax_orig.set_xlabel('X')
    ax_orig.set_ylabel('Y')
    ax_orig.set_zlabel('Z')

    # Initialize original motion objects
    line_orig, = ax_orig.plot([], [], [], lw=2, color='blue', alpha=0.7, label='Original')
    point_orig, = ax_orig.plot([], [], [], 'bo', markersize=10)
    trail_orig, = ax_orig.plot([], [], [], lw=1, color='lightblue', alpha=0.3)

    # === Right Plot: Aligned Motion ===
    ax_aligned = fig.add_subplot(1, 2, 2, projection='3d')
    ax_aligned.set_title("Aligned Motion (After FSUGW)", fontsize=14)
    ax_aligned.set_xlim(-1.5, 1.5)
    ax_aligned.set_ylim(-1.5, 1.5)
    ax_aligned.set_zlim(-1.5, 1.5)
    ax_aligned.set_xlabel('X')
    ax_aligned.set_ylabel('Y')
    ax_aligned.set_zlabel('Z')

    # Initialize aligned motion objects
    line_aligned, = ax_aligned.plot([], [], [], lw=2, color='red', alpha=0.7, label='Aligned')
    point_aligned, = ax_aligned.plot([], [], [], 'ro', markersize=10)
    trail_aligned, = ax_aligned.plot([], [], [], lw=1, color='pink', alpha=0.3)

    # Add legends
    ax_orig.legend()
    ax_aligned.legend()

    def update(frame):
        # Handle frame bounds
        if frame >= min_length:
            frame = min_length - 1
        
        # === Update Original Motion ===
        # Current position
        x_orig, y_orig, z_orig = X_original[frame]
        point_orig.set_data([x_orig], [y_orig])
        point_orig.set_3d_properties([z_orig])
        
        # Trail (past positions)
        if frame > 0:
            trail_orig.set_data(X_original[:frame, 0], X_original[:frame, 1])
            trail_orig.set_3d_properties(X_original[:frame, 2])

        # === Update Aligned Motion ===
        # Current position
        x_aligned, y_aligned, z_aligned = X_aligned[frame]
        point_aligned.set_data([x_aligned], [y_aligned])
        point_aligned.set_3d_properties([z_aligned])
        
        # Trail (past positions)
        if frame > 0:
            trail_aligned.set_data(X_aligned[:frame, 0], X_aligned[:frame, 1])
            trail_aligned.set_3d_properties(X_aligned[:frame, 2])

        return point_orig, trail_orig, point_aligned, trail_aligned

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=min_length, interval=100, blit=False, repeat=True
    )

    plt.tight_layout()
    plt.suptitle(f"Motion Comparison (Frames: Original={len(X_original)}, Aligned={len(X_aligned)})", 
                 fontsize=16, y=1.02)
    plt.show()
    
    return ani

def analyze_motion_differences(X, X_prime, Y):
    """
    Analyze and print differences between original and aligned motions
    """
    print("\n=== Motion Analysis ===")
    
    # Calculate differences
    min_len = min(len(X), len(X_prime))
    diff = X[:min_len] - X_prime[:min_len]
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    max_diff = np.max(np.abs(diff))
    
    print(f"Motion Alignment Quality:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Max Difference: {max_diff:.6f}")
    
    # Motion statistics
    print(f"\nMotion Statistics:")
    print(f"  Original X - Mean: {np.mean(X, axis=0)}, Std: {np.std(X, axis=0)}")
    print(f"  Aligned X' - Mean: {np.mean(X_prime, axis=0)}, Std: {np.std(X_prime, axis=0)}")
    print(f"  Control Y - Shape: {Y.shape}, Range: [{np.min(Y):.3f}, {np.max(Y):.3f}]")

def show_individual_motions(X, Y, X_prime):
    """
    Show individual motion playbacks for X, Y, and X'
    """
    print("\n=== Individual Motion Playbacks ===")
    print("Close each animation window to proceed to the next one...")
    
    # 1. Original Motion X
    print("\n1. Playing Original Motion (X)...")
    ani_x = motion_playback_3d(X, "Original Motion (X)")
    
    # 2. Control Signal Y
    print("\n2. Playing Control Signal (Y)...")
    ani_y = motion_playback_1d(Y, "Control Signal (Y)")
    
    # 3. Aligned Motion X'
    print("\n3. Playing Aligned Motion (X')...")
    ani_x_prime = motion_playback_3d(X_prime, "Aligned Motion (X')")
    
    return ani_x, ani_y, ani_x_prime

def create_triple_comparison(X, Y, X_prime):
    """
    Create a triple comparison animation showing X, Y, and X' together
    """
    min_length = min(len(X), len(X_prime))
    Y_flat = Y.flatten() if Y.ndim > 1 else Y
    
    fig = plt.figure(figsize=(18, 6))
    
    # === Left: Original Motion X ===
    ax_orig = fig.add_subplot(1, 3, 1, projection='3d')
    ax_orig.set_title("Original Motion (X)", fontsize=12)
    ax_orig.set_xlim(-1.5, 1.5)
    ax_orig.set_ylim(-1.5, 1.5)
    ax_orig.set_zlim(-1.5, 1.5)
    
    trail_orig, = ax_orig.plot([], [], [], lw=1, color='blue', alpha=0.5)
    point_orig, = ax_orig.plot([], [], [], 'bo', markersize=8)
    
    # === Middle: Control Signal Y ===
    ax_control = fig.add_subplot(1, 3, 2)
    ax_control.set_title("Control Signal (Y)", fontsize=12)
    ax_control.set_xlim(0, len(Y_flat))
    ax_control.set_ylim(np.min(Y_flat) * 1.2, np.max(Y_flat) * 1.2)
    ax_control.plot(Y_flat, color='lightgray', linestyle='--', alpha=0.5)
    ax_control.grid(True, alpha=0.3)
    
    line_control, = ax_control.plot([], [], 'g-', lw=2)
    point_control, = ax_control.plot([], [], 'go', markersize=8)
    
    # === Right: Aligned Motion X' ===
    ax_aligned = fig.add_subplot(1, 3, 3, projection='3d')
    ax_aligned.set_title("Aligned Motion (X')", fontsize=12)
    ax_aligned.set_xlim(-1.5, 1.5)
    ax_aligned.set_ylim(-1.5, 1.5)
    ax_aligned.set_zlim(-1.5, 1.5)
    
    trail_aligned, = ax_aligned.plot([], [], [], lw=1, color='red', alpha=0.5)
    point_aligned, = ax_aligned.plot([], [], [], 'ro', markersize=8)
    
    def update(frame):
        # Handle bounds
        frame = min(frame, min_length - 1)
        
        # Original Motion
        x, y, z = X[frame]
        point_orig.set_data([x], [y])
        point_orig.set_3d_properties([z])
        trail_orig.set_data(X[:frame+1, 0], X[:frame+1, 1])
        trail_orig.set_3d_properties(X[:frame+1, 2])
        
        # Control Signal
        if frame < len(Y_flat):
            line_control.set_data(np.arange(frame+1), Y_flat[:frame+1])
            point_control.set_data([frame], [Y_flat[frame]])
        
        # Aligned Motion
        x_prime, y_prime, z_prime = X_prime[frame]
        point_aligned.set_data([x_prime], [y_prime])
        point_aligned.set_3d_properties([z_prime])
        trail_aligned.set_data(X_prime[:frame+1, 0], X_prime[:frame+1, 1])
        trail_aligned.set_3d_properties(X_prime[:frame+1, 2])
        
        return point_orig, trail_orig, line_control, point_control, point_aligned, trail_aligned
    
    ani = animation.FuncAnimation(
        fig, update, frames=min_length, interval=120, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.suptitle("Triple Motion Comparison: X → Y → X'", fontsize=14, y=1.02)
    plt.show()
    
    return ani

if __name__ == "__main__":
    test_solver()
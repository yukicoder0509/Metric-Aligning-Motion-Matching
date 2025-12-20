import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_synthetic_data():
    # --- 1. Original Motion X (原始動作) ---
    # 模擬一個簡單的週期性動作 (例如: 畫圓)
    # 形狀: (Frames, Features). 這裡 Features = 3 (x, y, z)
    t_x = np.linspace(0, 4 * np.pi, 100) # 100 幀
    x_pos = np.cos(t_x)
    y_pos = np.sin(t_x)
    z_pos = np.zeros_like(t_x) # 假設在平面上
    X = np.stack([x_pos, y_pos, z_pos], axis=1)
    
    # --- 2. Control Sequence Y (控制訊號) ---
    # 使用 1D 波形控制。假設我們希望動作變慢或變快。
    # 形狀: (Frames, Features). 這裡 Features = 1 (振幅)
    # 產生一個不同頻率的波形
    t_y = np.linspace(0, 16 * np.pi, 120) # 120 幀 (長度可以跟 X 不同)
    Y = np.sin(t_y).reshape(-1, 1)
    
    return X, Y

def motion_playback_3d(X, title="3D Motion"):
    """
    Play 3D motion animation for X data
    Args:
        X: (frames, 3) - 3D motion data with x, y, z coordinates
        title: str - Title for the animation
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Initialize objects
    trail, = ax.plot([], [], [], lw=2, color='blue', alpha=0.7)
    point, = ax.plot([], [], [], 'ro', markersize=10)
    
    def update(frame):
        # Current position
        x, y, z = X[frame]
        point.set_data([x], [y])
        point.set_3d_properties([z])
        
        # Trail (past positions)
        trail.set_data(X[:frame+1, 0], X[:frame+1, 1])
        trail.set_3d_properties(X[:frame+1, 2])
        
        return point, trail

    ani = animation.FuncAnimation(
        fig, update, frames=len(X), interval=100, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    return ani

def motion_playback_1d(Y, title="Control Signal"):
    """
    Play 1D control signal animation for Y data
    Args:
        Y: (frames, 1) - 1D control signal data
        title: str - Title for the animation
    """
    Y_flat = Y.flatten() if Y.ndim > 1 else Y
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, len(Y_flat))
    ax.set_ylim(np.min(Y_flat) * 1.2, np.max(Y_flat) * 1.2)
    ax.set_xlabel('Time (frames)')
    ax.set_ylabel('Signal Value')
    ax.grid(True, alpha=0.3)

    # Plot full signal as background
    ax.plot(Y_flat, color='lightgray', linestyle='--', alpha=0.5, label='Full Signal')
    
    # Initialize animated objects
    line, = ax.plot([], [], 'b-', lw=2, label='Current Signal')
    point, = ax.plot([], [], 'ro', markersize=8, label='Current Position')
    
    ax.legend()
    
    def update(frame):
        # Current signal up to this frame
        line.set_data(np.arange(frame+1), Y_flat[:frame+1])
        
        # Current position
        point.set_data([frame], [Y_flat[frame]])
        
        return line, point

    ani = animation.FuncAnimation(
        fig, update, frames=len(Y_flat), interval=100, blit=False, repeat=True
    )
    
    plt.tight_layout()
    plt.show()
    return ani


if __name__ == "__main__":
    # show the synthetic data with animation
    
    X, Y = generate_synthetic_data()
    print(f"Original Motion X shape: {X.shape}") # 預期 (100, 3)
    print(f"Control Signal Y shape: {Y.shape}")  # 預期 (120, 1)

    fig = plt.figure(figsize=(12, 6))

    # 1. Setup 3D Plot for Motion X
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax3d.set_title("3D Motion (X)")
    ax3d.set_xlim(-1.5, 1.5)
    ax3d.set_ylim(-1.5, 1.5)
    ax3d.set_zlim(-1.5, 1.5)
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    # Initialize 3D objects
    # The "trail" (past movement)
    line_3d, = ax3d.plot([], [], [], lw=1, color='gray', alpha=0.5) 
    # The "head" (current position)
    point_3d, = ax3d.plot([], [], [], 'ro', markersize=8)           

    # 2. Setup 2D Plot for Control Signal Y
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.set_title("Control Signal (Y)")
    ax2d.set_xlim(0, len(X)) # Visualize up to length of X
    ax2d.set_ylim(np.min(Y)*1.2, np.max(Y)*1.2)
    ax2d.grid(True)

    # Initialize 2D objects
    # The full signal waveform (faint background)
    ax2d.plot(Y, color='lightgray', linestyle='--') 
    # The "cursor" moving along the signal
    point_2d, = ax2d.plot([], [], 'bo', markersize=8)
    # The signal trace up to current time
    line_2d, = ax2d.plot([], [], 'b-', lw=2)

    # --- Update Function ---
    def update(frame):
        # Update 3D Motion (X)
        # Current x, y, z
        cx, cy, cz = X[frame]
        
        # Update head
        point_3d.set_data([cx], [cy])
        point_3d.set_3d_properties([cz])
        
        # Update trail (history)
        line_3d.set_data(X[:frame+1, 0], X[:frame+1, 1])
        line_3d.set_3d_properties(X[:frame+1, 2])

        # Update Control Signal (Y)
        # Note: Y has 120 frames, X has 100. We protect against index errors.
        if frame < len(Y):
            cy_val = Y[frame]
            # Update dot
            point_2d.set_data([frame], [cy_val])
            # Update line trace
            line_2d.set_data(np.arange(frame+1), Y[:frame+1])

        return point_3d, line_3d, point_2d, line_2d

    # --- Create Animation ---
    # frames=len(X) ensures we stop when the motion X ends
    ani = animation.FuncAnimation(
        fig, update, frames=len(X), interval=50, blit=False
    )

    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt
import solver
import synthetic_data
import patches
import distance

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
    
    # 視覺化 T 矩陣
    plt.figure(figsize=(8, 6))
    plt.imshow(T, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Probability Mass')
    plt.title("Transport Plan T (Should be diagonal-ish for identical inputs)")
    plt.xlabel("Original Motion Indices (X)")
    plt.ylabel("Control Sequence Indices (Y)")
    plt.show()
    
    # 自動化判定
    is_diagonal_dominant = np.all(np.argmax(T, axis=1) == np.arange(T.shape[0]))
    # 注意：因為只有完全一樣長度才會有完美的對角線，
    # 但如果長度一樣，最大值應該主要落在對角線上。
    print(f"Is diagonal structure visible? (Visual check recommended)")

if __name__ == "__main__":
    test_solver()
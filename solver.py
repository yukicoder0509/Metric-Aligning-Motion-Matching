import numpy as np
from scipy.spatial.distance import cdist

def compute_wasserstein_cost(Y_patches, X_patches):
    """
    計算兩個 Patch 集合之間的 Cross-Domain Distance Matrix (Cost Matrix)。
    這是 L_W (Wasserstein Loss) 的基礎。
    
    Args:
        Y_patches: (N_y, Dim) - 控制序列的 patches (或當前生成的 patches)
        X_patches: (N_x, Dim) - 原始動作的 patches
        
    Returns:
        C_XY: (N_y, N_x) - 歐幾里得距離平方矩陣
    """
    # 使用 cdist 計算兩組點之間的距離
    # metric='sqeuclidean' 表示歐幾里得距離的平方，這是 OT 的標準 Cost 定義
    C_XY = cdist(Y_patches, X_patches, metric='sqeuclidean')
    
    # 正規化 Cost (這在數值穩定性上很重要)
    if C_XY.max() > 0:
        C_XY /= C_XY.max()
        
    return C_XY

def solve_fsugw(D_X, D_Y, X_patches, Y_patches, T_init=None, 
                alpha=0.5, rho=1.0, epsilon=0.01, num_iters=50):
    """
    求解 Fused Semi-Unbalanced Gromov-Wasserstein (FSUGW) 問題。
    
    對應論文 Equation (3):
    min L_FSUGW = alpha * L_GW + (1-alpha) * L_W + lambda * KL - epsilon * H(T)
    
    Args:
        D_X: (N_x, N_x) - 原始動作的內部距離矩陣
        D_Y: (N_y, N_y) - 控制序列的內部距離矩陣
        X_patches: (N_x, Dim) - 原始動作數據
        Y_patches: (N_y, Dim) - 目標/控制動作數據
        T_init: (N_y, N_x) - 初始傳輸計畫 (可選)
        alpha: (float) 0~1, 平衡 GW loss 和 Wasserstein loss。
               alpha 越大越重視結構對齊，越小越重視內容對齊。
        rho: (float) 對應論文的 lambda (Marginal relaxation)。
             rho 越大越接近 Hard Constraint (必須完全用到所有原始動作)。
             rho 越小允許越鬆散的對應 (Unbalanced)。
        epsilon: (float) Entropy regularization strength.
        num_iters: (int) Sinkhorn 迭代次數。
        
    Returns:
        T: (N_y, N_x) - 最佳化的 Transport Plan
    """
    N_y = D_Y.shape[0]
    N_x = D_X.shape[0]
    
    # 1. 定義 Marginals (邊際分佈目標)
    # 論文中 a 是均勻分佈 (1/Ly) [cite: 201]
    mu = np.ones(N_y) / N_y 
    # b 也是均勻分佈 (1/Lx)
    nu = np.ones(N_x) / N_x 
    
    # 初始化 T
    if T_init is None:
        # 初始化為外積 (Product of marginals)
        T = np.outer(mu, nu)
    else:
        T = T_init

    # 計算 Wasserstein Cost (L_W 的基礎) - 這部分在迴圈內通常是不變的(除非 Y_patches 變了)
    C_XY = compute_wasserstein_cost(Y_patches, X_patches)

    # Sinkhorn 迭代迴圈
    # 這裡我們使用簡易版的 Gradient Descent + Sinkhorn Projection 策略
    # 這是一種解 GW 問題的常見方法 (Mirror Descent)
    
    for i in range(num_iters):
        # --- A. 計算 Gradient (基於當前的 T) ---
        # GW Loss 的梯度近似為: D_Y * T * D_X^T (這是一個 tensor product 操作)
        # 這裡計算的是 "Local Cost"
        grad_GW = np.dot(D_Y, np.dot(T, D_X.T))
        
        # 由於我們是最小化 -<D_Y, T*D_X*T^t> (這是 GW 的內積形式)，
        # 我們將其轉換為 Cost 形式。
        # 簡單理解：如果 Y 的結構 D_Y 和 X 的結構 D_X 透過 T 對齊得好，這個值會大。
        # 因為 Sinkhorn 吃的是 "Cost" (越小越好)，所以我們取負號，或者使用 (D_Y - T D_X)^2 的展開式
        # 這裡使用標準的 GW 梯度形式: Cost_GW = -2 * D_Y * T * D_X
        C_GW = -2 * np.dot(D_Y, np.dot(T, D_X))
        
        # 總 Cost 矩陣 M
        # M = alpha * Cost_GW + (1 - alpha) * Cost_Wasserstein
        M = alpha * C_GW + (1 - alpha) * C_XY
        
        # --- B. Sinkhorn 投影 (解決熵正規化問題) ---
        # Kernel K = exp(-M / epsilon)
        # 為了數值穩定，減去 min
        M_stable = M - M.min()
        K = np.exp(-M_stable / epsilon)
        
        # 進行幾次 Sinkhorn scaling 來更新 T，使其符合邊際條件
        # 對於 Semi-Unbalanced:
        # Row sum (Source/Control side) 必須是 mu (Hard constraint) [cite: 203]
        # Col sum (Target/Original side) 接近 nu (Soft KL constraint) [cite: 203]
        
        # 初始化 scaling vectors
        u = np.ones(N_y)
        v = np.ones(N_x)
        
        # Sinkhorn 內迴圈 (通常 5-10 次就夠了)
        for _ in range(5):
            # 1. Update u (Rows): Hard constraint -> T * 1 = mu
            # u = mu / (K @ v)
            kv = np.dot(K, v)
            u = mu / (kv + 1e-10)
            
            # 2. Update v (Cols): Soft constraint -> KL penalty
            # 對於 Unbalanced OT，v 的更新公式是 (nu / K.T @ u) ^ (epsilon / (epsilon + rho))
            # 當 rho -> 無限大時，這變成標準的 v = nu / (K.T @ u)
            ktu = np.dot(K.T, u)
            
            if rho > 100: # 視為 Hard constraint
                v = nu / (ktu + 1e-10)
            else: # Soft constraint (Semi-unbalanced)
                exponent = rho / (rho + epsilon)
                v = (nu / (ktu + 1e-10)) ** exponent
        
        # 更新 T
        # T = diag(u) * K * diag(v)
        T = np.diag(u) @ K @ np.diag(v)
        
    return T

def blend_patches(T, X_patches, window_size=11, stride=1):
    """
    根據 Transport Plan (T) 和原始動作 Patches (X_patches)，
    合成出對齊後的動作序列 X_prime。
    
    對應論文 Algorithm 1: X' <- BlendPatches(T * X_tilde ...)
    對應論文 3.2.4 (3): X' ← BlendPatches(T·X·L_Y).
        With T fixed, update X' by matching weighted by the transport plan and blending the overlapping regions through averaging.
    
    Args:
        T: (N_y, N_x) - Transport Plan (由 FSUGW 算出)
           N_y 是控制序列(Control)的 patch 數
           N_x 是原始動作(Original)的 patch 數
        X_patches: (N_x, Flattened_Dim) - 原始動作的 patches
           Flattened_Dim = window_size * feature_dim
        window_size: (int) Patch 的時間長度 (e.g., 11)
        stride: (int) Patch 的步長 (通常為 1)
        
    Returns:
        X_prime: (L_new, feature_dim) - 最終對齊後的連續動作序列
    """
    
    # --- Step 1: 加權映射 (Barycentric Projection) ---
    # 計算每個新 Patch 是由哪些舊 Patches 組成的
    # 公式概念: X'_patches = (T @ X_patches) / row_sums_of_T
    
    # 1. 加權總和
    # (N_y, N_x) @ (N_x, Dim) -> (N_y, Dim)
    X_prime_patches_weighted = np.dot(T, X_patches)
    
    # 2. 正規化 (除以權重總和)
    # T 的每一列 (row) 代表一個 Y patch 對應到 X patches 的機率分佈
    # 我們需要除以該列的總和來做加權平均
    row_sums = T.sum(axis=1)[:, np.newaxis] # Shape: (N_y, 1)
    
    # 避免除以 0 (雖然理論上 T 不會全為 0，但為了數值穩定)
    row_sums[row_sums < 1e-10] = 1.0
    
    X_prime_patches = X_prime_patches_weighted / row_sums
    
    # --- Step 2: 重疊平均 (Reconstruction from Patches) ---
    # 將重疊的 Patches 拼回連續序列
    # blending the overlapping regions through averaging.
    
    num_patches_y = X_prime_patches.shape[0]
    flattened_dim = X_prime_patches.shape[1]
    feature_dim = flattened_dim // window_size
    
    # 計算最終序列的總長度
    # Length = (Strips - 1) * Stride + Window
    final_length = (num_patches_y - 1) * stride + window_size
    
    # 初始化累加器和計數器
    X_prime_accum = np.zeros((final_length, feature_dim))
    counts = np.zeros((final_length, 1))
    
    for i in range(num_patches_y):
        # 計算該 Patch 在時間軸上的起始與結束位置
        start_frame = i * stride
        end_frame = start_frame + window_size
        
        # 取出該 Patch 的數據並 Reshape 回 (Window, Features)
        # 因為輸入時是被 Flatten 過的
        patch_data = X_prime_patches[i].reshape(window_size, feature_dim)
        
        # 累加到對應的時間位置
        X_prime_accum[start_frame:end_frame] += patch_data
        counts[start_frame:end_frame] += 1.0
        
    # --- Step 3: 取平均 ---
    # 避免除以 0 (雖然邏輯上 counts 至少為 1)
    counts[counts == 0] = 1.0
    
    X_prime = X_prime_accum / counts
    
    return X_prime
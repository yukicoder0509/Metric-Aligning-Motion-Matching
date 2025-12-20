import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import zoom
from patches import extract_patches
from distance import compute_normalized_distance

def compute_wasserstein_cost(X_patches, X_prime_patches):
    """
    計算兩個 Patch 集合之間的 Distance Matrix (Cost Matrix)。
    這是 L_W (Wasserstein Loss) 的基礎。
    """
    # 使用 cdist 計算兩組點之間的距離
    # metric='sqeuclidean' 表示歐幾里得距離的平方，這是 OT 的標準 Cost 定義
    C_linear = cdist(X_patches, X_prime_patches, metric='sqeuclidean')
    
    # 正規化 Cost (這在數值穩定性上很重要)
    if C_linear.max() > 0:
        C_linear /= C_linear.max()
        
    return C_linear

def compute_gw_gradient(D_Y, D_X, T):
    """
    計算 Gromov-Wasserstein Loss 的梯度。
    這是基於當前的 Transport Plan T 計算的。
    """
    # 計算梯度: grad_GW = D_Y * T * D_X^T
    grad_GW = np.dot(D_Y, np.dot(T, D_X.T))
    return grad_GW

def solve_fsugw(D_X, D_Y, X_patches, X_prime_patches, 
                alpha=0.5, rho=1.0, epsilon=0.01, num_iters=50):
    """
    求解 Fused Semi-Unbalanced Gromov-Wasserstein (FSUGW) 問題。
    
    對應論文 Equation (3):
    min L_FSUGW = alpha * L_GW + (1-alpha) * L_W + lambda * KL - epsilon * H(T)
    
    Args:
        D_X: (N_x, N_x) - 原始動作的內部距離矩陣
        D_Y: (N_y, N_y) - 控制序列的內部距離矩陣
        X_patches: (N_x, Dim) - 原始動作數據
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
    
    # 初始化 T 為外積 (Product of marginals)
    T = np.outer(mu, nu)

    # 計算 Wasserstein Cost (L_W 的基礎) - 這部分在迴圈內通常是不變的
    if X_prime_patches is None:
        C_W = np.zeros((N_y, N_x))
    else:
        C_W = compute_wasserstein_cost(X_prime_patches, X_patches)

    # Sinkhorn 迭代迴圈
    # 這裡我們使用簡易版的 Gradient Descent + Sinkhorn Projection 策略
    # 這是一種解 GW 問題的常見方法 (Mirror Descent)
    
    for i in range(num_iters):
        # --- A. 計算 Gradient (基於當前的 T) ---
        # GW Loss 的梯度近似為: D_Y * T * D_X^T (這是一個 tensor product 操作)
        # 這裡計算的是 "Local Cost"
        C_GW = compute_gw_gradient(D_Y, D_X, T)
        
        # 總 Cost 矩陣 M
        # M = alpha * Cost_GW + (1 - alpha) * Cost_Wasserstein
        M = alpha * C_GW + (1 - alpha) * C_W
        
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

def upsample_motion(motion, target_length):
    """
    將動作序列 (Time, Dim) 放大到 target_length。
    用於將 Coarse 層級的結果作為 Fine 層級的初始值。
    """
    current_length = motion.shape[0]
    zoom_factor = target_length / current_length
    # zoom(input, zoom_factors, order=1(linear))
    # 我們只在時間軸 (axis 0) 放縮，特徵軸 (axis 1) 保持 1.0
    return zoom(motion, (zoom_factor, 1.0), order=1)

def downsample_motion(motion, scale_factor):
    """
    將動作序列 (Time, Dim) 依 scale_factor 下採樣。
    用於將 Fine 層級的動作降至 Coarse 層級。
    """
    return motion[::scale_factor]

def coarse_to_fine(X, Y, alpha=0.5, rho=1.0, epsilon=0.01, num_stage=6, num_iters=20):
    """
    solve FSUGW using a coarse-to-fine strategy.
    """

    X_prime = None

    for k in range(num_stage):
        print(f"--- Stage {k+1}/{num_stage} ---")
        
        # downsample X, Y to current scale
        scale_factor = 2 ** (num_stage - k - 1)
        
        if X.shape[0] // scale_factor < 11 or Y.shape[0] // scale_factor < 11:
            # size after downsampling is too small (< patch window_size) for patch extraction
            print("Skipping this stage due to small size after downsampling.")
            continue
        X_k = downsample_motion(X, scale_factor)
        Y_k = downsample_motion(Y, scale_factor)

        if X_prime is None:
            # initialize X_prime
            # paper did not specify how to initialize X_prime at the coarsest level
            # so we scale X to the length of Y as the initial X_prime.
            # To let the length of X_prime match Y, we upsample X to the current resolution. This is a bit different to the paper
            X_prime = upsample_motion(X, Y_k.shape[0])
        else: 
            X_prime = upsample_motion(X_prime, Y_k.shape[0])

        for m in range(num_iters):
            # extract patches
            X_patches = extract_patches(X_k, window_size=11, stride=1)
            Y_patches = extract_patches(Y_k, window_size=11, stride=1)
            X_prime_patches = extract_patches(X_prime, window_size=11, stride=1)

            # compute distance matrices
            D_X = compute_normalized_distance(X_patches)
            D_Y = compute_normalized_distance(Y_patches)

            # solve FSUGW
            T = solve_fsugw(D_X, D_Y, X_patches, X_prime_patches, alpha, rho, epsilon, num_iters=10)

            # blend patches to get new X_prime
            X_prime = blend_patches(T, X_patches, window_size=11, stride=1)

    return X_prime

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
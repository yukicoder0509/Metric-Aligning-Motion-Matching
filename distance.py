from scipy.spatial.distance import cdist

# base on section 3.2.3 of the paper
# in the calculation of GW loss, we need all possible pairwise distances of X and Y patches
def compute_normalized_distance(patches):
    """
    計算 Patch 兩兩之間的歐幾里得距離並正規化。
    對應論文 Eq (2) 中的 d_X 和 d_Y。
    """
    # cdist 計算 all possible pairwise distance
    D = cdist(patches, patches, metric='euclidean')
    
    # 正規化: 使用平均值而非最大值，對異常值更robust
    if D.mean() > 0:
        D = D / D.mean()
    
    return D
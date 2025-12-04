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
    
    # 正規化: 論文建議除以最大值 (或平均值)
    if D.max() > 0:
        D = D / D.max()
    
    return D
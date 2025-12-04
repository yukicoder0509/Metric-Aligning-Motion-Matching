import numpy as np

def extract_patches(sequence, window_size=11, stride=1):
    """
    將時間序列轉換為 Patch 矩陣。
    Input: (T, D)
    Output: (Num_Patches, window_size * D)
    """
    T, D = sequence.shape
    # 計算 Patch 數量
    num_patches = (T - window_size) // stride + 1
    
    patches = []
    for i in range(num_patches):
        start = i * stride
        end = start + window_size
        # 取出視窗內的資料並攤平 (Flatten)
        # 對應論文: 將每個 patch 視為高維空間中的一個點
        patch = sequence[start:end, :].flatten()
        patches.append(patch)
        
    return np.array(patches)
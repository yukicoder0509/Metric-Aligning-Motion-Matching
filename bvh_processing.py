import numpy as np
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
import os

def parse_bvh_to_mamm_format(bvh_file_path):
    """
    將 BVH 檔案轉換為 MAMM 論文定義的特徵格式 X。
    X = [Root_Velocity, Joint_Rotation_Matrices]
    """
    print(f"Loading BVH file: {bvh_file_path}...")
    with open(bvh_file_path) as f:
        mocap = Bvh(f.read())

    frames = mocap.nframes
    joints = mocap.get_joints_names()
    print(f"Found {frames} frames and {len(joints)} joints.")

    # --- 1. 提取 Root Position 並計算速度 V ---
    root_name = joints[0] 
    
    root_positions = []
    for i in range(frames):
        x = float(mocap.frame_joint_channel(i, root_name, 'Xposition'))
        y = float(mocap.frame_joint_channel(i, root_name, 'Yposition'))
        z = float(mocap.frame_joint_channel(i, root_name, 'Zposition'))
        root_positions.append([x, y, z])
    
    root_positions = np.array(root_positions)
    
    # 計算速度 V
    root_velocity = np.diff(root_positions, axis=0, prepend=root_positions[0:1])
    
    # --- 2. 提取 Joint Rotations 並轉為矩陣 R ---
    all_rotations = []
    
    # 預先解析每個關節的旋轉順序
    joint_channel_orders = {}
    for joint in joints:
        channels = mocap.joint_channels(joint)
        rot_channels = [c for c in channels if 'rotation' in c]
        if not rot_channels:
            continue
        
        # [Fix] 使用大寫 (UPPERCASE) 來表示 Intrinsic Rotations (e.g., 'ZXY')
        # 這對於正確還原 BVH 動作至關重要
        order_str = "".join([c[0].upper() for c in rot_channels])
        joint_channel_orders[joint] = order_str

    print("Extracting rotations and converting to matrices...")
    
    for i in range(frames):
        frame_rotations = []
        
        for joint in joints:
            if joint not in joint_channel_orders:
                continue
                
            order = joint_channel_orders[joint]
            angles = []
            
            # 根據順序提取角度
            for char in order:
                channel_name = f"{char}rotation" # e.g. Zrotation
                val = float(mocap.frame_joint_channel(i, joint, channel_name))
                angles.append(val)
            
            # Matrix conversion
            r = R.from_euler(order, angles, degrees=True)
            matrix = r.as_matrix() # shape (3, 3)
            frame_rotations.append(matrix.flatten())
            
        if frame_rotations:
            all_rotations.append(np.concatenate(frame_rotations))
        else:
            all_rotations.append(np.zeros(1))
        
    all_rotations = np.array(all_rotations)

    # --- 3. 合併 V 和 R ---
    X = np.concatenate([root_velocity, all_rotations], axis=1)
    
    print(f"Conversion complete. Output X shape: {X.shape}")
    return X

# X define the motion
# reference_bvh_path is used to get the skeleton structure and initial positions
def save_mamm_format_to_bvh(X, reference_bvh_path, output_bvh_path):
    """
    將 MAMM 格式的 X 轉回 BVH 檔案。
    """
    print(f"Converting Matrix to BVH: {output_bvh_path}...")
    
    with open(reference_bvh_path) as f:
        bvh_content = f.read()
        mocap = Bvh(bvh_content)

    joints = mocap.get_joints_names()
    frames = X.shape[0]
    
    # --- 1. 還原 Root Position ---
    root_velocity = X[:, 0:3]
    root_name = joints[0]
    
    # 獲取初始位置 (從 Reference 第一幀)
    start_pos = np.array([
        float(mocap.frame_joint_channel(0, root_name, 'Xposition')),
        float(mocap.frame_joint_channel(0, root_name, 'Yposition')),
        float(mocap.frame_joint_channel(0, root_name, 'Zposition'))
    ])
    
    # 積分: P_t = P_0 + sum(V)
    # 由於 parser 中 V[0]=0, cumsum 後第一項為 0, 加上 start_pos 剛好是正確的初始位置
    root_positions = np.cumsum(root_velocity, axis=0) + start_pos

    # --- 2. 準備 Header ---
    header_lines = []
    with open(reference_bvh_path) as f:
        for line in f:
            if "MOTION" in line:
                header_lines.append(line)
                break
            header_lines.append(line)
    
    with open(output_bvh_path, 'w') as f:
        f.writelines(header_lines)
        f.write(f"Frames: {frames}\n")
        f.write(f"Frame Time: {mocap.frame_time}\n")
        
        # --- 3. 逐幀寫入數據 ---
        rot_data_start_idx = 3 # Skip root velocity
        
        for i in range(frames):
            line_values = []
            current_rot_idx = rot_data_start_idx
            
            for joint in joints:
                channels = mocap.joint_channels(joint)
                if not channels:
                    continue
                
                # 準備該 Joint 的旋轉數據 (如果有的話)
                joint_euler_map = {} # {'Xrotation': val, ...}
                rot_channels = [c for c in channels if 'rotation' in c]
                
                if rot_channels:
                    # 提取 Matrix
                    rot_matrix_flat = X[i, current_rot_idx : current_rot_idx + 9]
                    rot_matrix = rot_matrix_flat.reshape(3, 3)
                    current_rot_idx += 9
                    
                    # [Fix] 使用大寫 (UPPERCASE) 來還原 Intrinsic Rotations
                    order_str = "".join([c[0].upper() for c in rot_channels])
                    
                    r = R.from_matrix(rot_matrix)
                    euler_angles = r.as_euler(order_str, degrees=True)
                    
                    # 將角度對應回 channel 名稱
                    for char, angle in zip(order_str, euler_angles):
                        joint_euler_map[f"{char}rotation"] = angle

                # 依序寫入 Channel 數據
                for channel in channels:
                    if 'position' in channel and joint == root_name:
                        # Root Position
                        if 'Xposition' == channel: line_values.append(root_positions[i][0])
                        elif 'Yposition' == channel: line_values.append(root_positions[i][1])
                        elif 'Zposition' == channel: line_values.append(root_positions[i][2])
                    
                    elif channel in joint_euler_map:
                        # Rotation
                        line_values.append(joint_euler_map[channel])
                    
                    else:
                        # Fallback (例如非 Root 的 Position，或 Scaling)
                        line_values.append(0.0)
            
            # 寫入一行
            line_str = " ".join([f"{val:.6f}" for val in line_values])
            f.write(line_str + "\n")
            
    print(f"Saved reconstructed BVH to {output_bvh_path}")

if __name__ == "__main__":
    # Test Block
    input_filename = "./original_motion/Hip Hop Dancing.bvh" 
    npy_filename = "./output/Hip Hop Dancing.npy"
    recon_filename = "./output/Hip Hop Dancing_recon.bvh"

    if os.path.exists(input_filename):
        try:
            # 1. BVH -> Matrix
            X_data = parse_bvh_to_mamm_format(input_filename)
            np.save(npy_filename, X_data)
            
            # 2. Matrix -> BVH
            save_mamm_format_to_bvh(X_data, input_filename, recon_filename)
            print(f"Test complete. Check '{recon_filename}'")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"File '{input_filename}' not found. Please verify.")
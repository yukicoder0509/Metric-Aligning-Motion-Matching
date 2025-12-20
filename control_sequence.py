import numpy as np
import tkinter as tk
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

SKETCH_CONTROL_SEQ_FILE = "./output/control_sequence_Y.npy"

def sin_wave_control_sequence(length: float, frequency: float, amplitude: float) -> np.ndarray:
    """Waveform to motion control sequence"""
    t = np.linspace(0, length - 1, length)
    control_sequence = amplitude * np.sin(2 * np.pi * frequency * t / length)
    return control_sequence

class SketchRecorder:
    def __init__(self, root, width=800, height=600):
        self.root = root
        self.root.title("MAMM Sketch Recorder")
        
        self.width = width
        self.height = height
        self.points = [] # 儲存原始滑鼠軌跡
        
        # --- UI Setup ---
        self.canvas = tk.Canvas(root, width=width, height=height, bg="white")
        self.canvas.pack(pady=10)
        
        # 按鈕區
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Process & Save", command=lambda: self.process_and_save(), bg="#ddffdd").pack(side=tk.LEFT, padx=5)
        
        # 綁定滑鼠事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonPress-1>", self.start_paint)

        # 狀態標籤
        self.lbl_status = tk.Label(root, text="Draw a curve on the canvas (simulating motion trajectory).")
        self.lbl_status.pack()

    def start_paint(self, event):
        # 每次重新畫時清空舊的 (如果你只想要單一軌跡)
        self.points = []
        self.canvas.delete("all")
        self.points.append((event.x, event.y))

    def paint(self, event):
        x, y = event.x, event.y
        self.points.append((x, y))
        
        # 簡單視覺化：畫線連接上一點
        if len(self.points) > 1:
            x0, y0 = self.points[-2]
            self.canvas.create_line(x0, y0, x, y, width=3, fill="black", capstyle=tk.ROUND, smooth=True)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.points = []
        self.lbl_status.config(text="Canvas cleared.")

    def process_and_save(self, target_frames=120, sampling: str="distance"):
        """
        Open a canva and let user draw a sketch as control sequence.
        
        :target_frames: 要產生的 control sequence 的長度
        :sampling: "temporal" or "distance" - 取樣方法, temporal: 等時間間隔取樣, distance: 基於距離的取樣(等距取樣)
        """
        if len(self.points) < 10:
            self.lbl_status.config(text="Trajectory too short! Please draw more.")
            return

        # 1. 轉為 Numpy Array
        raw_path = np.array(self.points) # Shape: (N, 2)
        
        # 2. 歸一化 (Normalize) 到 0~1 之間
        # 我們希望保留長寬比 (Aspect Ratio)，所以除以畫布長邊
        scale = max(self.width, self.height)
        norm_path = raw_path / scale
        # 翻轉 Y 軸 (Tkinter (0,0) 在左上，通常數學座標在左下，視需求而定)
        norm_path[:, 1] = 1.0 - norm_path[:, 1] 

        # 3. 重取樣 (Resample) - 對應論文 "constant time intervals"
        # 我們想把它變成固定的幀數，例如 120 幀 (配合你的動作長度)
        target_frames = 120 

        if sampling == "temporal":
            # 3.a 簡單線性插值 (Temporal Subsampling) - 直接等距取樣
            total_points = len(norm_path)
            indices = np.linspace(0, total_points - 1, target_frames).astype(int)
            Y_final = norm_path[indices]  # Shape: (120, 2)

        else: # sampling == "distance". if not specified, default to distance-based
            # 3.b 基於距離的插值 (Distance-based Interpolation)
            # 計算累積距離 (Arc length)
            # 這樣做是為了讓點在線上分佈均勻 (Constant speed assumption)
            dists = np.linalg.norm(np.diff(norm_path, axis=0), axis=1)
            cum_dist = np.insert(np.cumsum(dists), 0, 0)
            total_dist = cum_dist[-1]
            
            # 建立插值函數 (Distance -> Coordinate)
            # 這裡有兩個函數: dist -> x, dist -> y
            fx = interp1d(cum_dist, norm_path[:, 0], kind='linear')
            fy = interp1d(cum_dist, norm_path[:, 1], kind='linear')
            
            # 產生新的等距取樣點
            new_dists = np.linspace(0, total_dist, target_frames)
            resampled_x = fx(new_dists)
            resampled_y = fy(new_dists)
            
            Y_final = np.stack([resampled_x, resampled_y], axis=1) # Shape: (120, 2)
        
        # 4. 平滑化 (Smoothing) - 對應論文 "Gaussian smoothing"
        # sigma 控制平滑程度
        Y_final[:, 0] = gaussian_filter1d(Y_final[:, 0], sigma=2)
        Y_final[:, 1] = gaussian_filter1d(Y_final[:, 1], sigma=2)

        # 5. 存檔與檢查
        output_filename = SKETCH_CONTROL_SEQ_FILE
        np.save(output_filename, Y_final)
        
        self.lbl_status.config(text=f"Saved to {output_filename}. Shape: {Y_final.shape}")
        
        # 視覺化比較 (Raw vs Resampled)
        self.show_debug_plot(raw_path, Y_final, scale)

    def show_debug_plot(self, raw, processed, scale):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left chart - Raw Mouse Input
        ax1.plot(raw[:, 0], raw[:, 1], 'b.', alpha=0.6, label="Raw Mouse Input")
        ax1.set_title("Raw Mouse Input")
        ax1.set_aspect('equal')
        ax1.set_xlim(0, self.width)
        ax1.set_ylim(0, self.height)
        ax1.invert_yaxis()  # Tkinter 座標系
        ax1.legend()
        
        # Right chart - Processed (X, 1-Y) * scale
        proc_display_x = processed[:, 0] * scale
        proc_display_y = (1.0 - processed[:, 1]) * scale # 反轉回 Tkinter 座標系
        
        ax2.plot(proc_display_x, proc_display_y, 'r.', alpha=0.6, label="Resampled Control Signal (Y)")
        ax2.set_title("Resampled Control Signal")
        ax2.set_aspect('equal')
        ax2.set_xlim(0, self.width)
        ax2.set_ylim(0, self.height)
        ax2.invert_yaxis()  # Tkinter 座標系
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

def read_sketch_control_sequence(filename=SKETCH_CONTROL_SEQ_FILE) -> np.ndarray:
    """讀取手繪控制序列檔案"""
    return np.load(filename)

def sketch_control_sequence():
    root = tk.Tk()
    app = SketchRecorder(root)
    root.mainloop()

    return read_sketch_control_sequence()

if __name__ == "__main__":
    seq = sketch_control_sequence()
    print("Control Sequence Shape:", seq.shape)
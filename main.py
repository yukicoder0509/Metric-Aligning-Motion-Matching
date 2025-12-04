import synthetic_data
import patches

X, Y = synthetic_data.generate_synthetic_data()

X_patches = patches.extract_patches(X, window_size=11, stride=1)
print(f"Extracted Patches shape: {X_patches.shape}") # 預期 (90, 33)

Y_patches = patches.extract_patches(Y, window_size=11, stride=1)
print(f"Extracted Patches shape: {Y_patches.shape}") # 預期 (110, 11)
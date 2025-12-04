import synthetic_data
import patches
import distance
import matplotlib.pyplot as plt

# get synthetic data for testing
X, Y = synthetic_data.generate_synthetic_data()

# extract patches: section 3.2.1 of the paper
X_patches = patches.extract_patches(X, window_size=11, stride=1)
print(f"Extracted Patches shape: {X_patches.shape}") # 預期 (90, 33)

Y_patches = patches.extract_patches(Y, window_size=11, stride=1)
print(f"Extracted Patches shape: {Y_patches.shape}") # 預期 (110, 11)

# prepare for distance calculation section 3.2.3 of the paper
# pre compute pairwise distance matrices
D_X = distance.compute_normalized_distance(X_patches)
print(f"X Pairwise Distances shape: {D_X.shape}") # 預期 (90, 90)

D_Y = distance.compute_normalized_distance(Y_patches)
print(f"Y Pairwise Distances shape: {D_Y.shape}") # 預期 (110, 110)

# visualize the distance matrices for verification
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
ax1 = axes[0]
im1 = ax1.imshow(D_X, cmap='viridis')
ax1.set_title(f"Original Motion Distance Matrix ($D_X$)\nSize: {D_X.shape}")
plt.colorbar(im1, ax=ax1)

ax2 = axes[1]
im2 = ax2.imshow(D_Y, cmap='viridis')
ax2.set_title(f"Control Signal Distance Matrix ($D_Y$)\nSize: {D_Y.shape}")
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
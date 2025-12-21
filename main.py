import control_sequence
import bvh_processing
import patches
import solver
import distance

X = bvh_processing.parse_bvh_to_mamm_format("./original_motion/Hip Hop Dancing.bvh")
Y = control_sequence.sketch_control_sequence()

# # ==== No Coarse-to-Fine ====
# X_patches = patches.extract_patches(X, window_size=11, stride=1)
# Y_patches = patches.extract_patches(Y, window_size=11, stride=1)
# D_X = distance.compute_normalized_distance(X_patches)
# D_Y = distance.compute_normalized_distance(Y_patches)

# T = solver.solve_fsugw(D_X, D_Y, X_patches, X_prime_patches=None, alpha=0.5, rho=100.0, epsilon=0.01, num_iters=20)
# X_aligned = solver.blend_patches(T, X_patches, window_size=11, stride=1)
# solver.save_heatmaps(D_X, D_Y, T, 1, 1)
# # ====

#==== With Coarse-to-Fine ====
X_aligned = solver.coarse_to_fine(X, Y)
#====

bvh_processing.save_mamm_format_to_bvh(X_aligned, reference_bvh_path="./original_motion/Hip Hop Dancing.bvh", output_bvh_path="./output/Hip_Hop_Dancing_Aligned.bvh")
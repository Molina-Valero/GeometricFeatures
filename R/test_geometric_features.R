# Simple test for point cloud geometric features
# Make sure your C++ file is named "geometric_features.cpp" or adjust accordingly
#
# This script uses C++ parallelization for performance.
#
# ✅ Requirements:
#   - R packages: Rcpp
#   - A working C++ toolchain:
#       * Windows: RTools (with OpenMP support)
#       * macOS: install libomp (brew install libomp)
#       * Linux: g++ / build-essential

library(Rcpp)

# Compile the C++ code
sourceCpp("../src/geometric_features_optimized.cpp")

# Create a simple point cloud (a small cube)
set.seed(123)
n <- 50000
x <- runif(n, 0, 1)
y <- runif(n, 0, 1)
z <- runif(n, 0, 1)

# Test 1: Eigenvalue analysis on all points
cat("Test 1: Eigenvalue Analysis\n")
point_matrix <- cbind(x, y, z)
result_pca <- eigenvalue_analysis(point_matrix)
print(result_pca)

# Test 2: Geometric features for a few points
cat("\nTest 2: Geometric Features\n")
# Select 5 points to analyze
test_points <- cbind(
  point = 1:length(x),
  x = x,
  y = y,
  z = z
)

# Calculate features within radius of 2 units
time_taken <- system.time({
  features <- geometric_features_batch(
    points = test_points,
    x_all = x,
    y_all = y,
    z_all = z,
    dist = 2.0,
    num_threads = 1
  )
})
print(time_taken)

time_taken <- system.time({
  features <- geometric_features_batch(
    points = test_points,
    x_all = x,
    y_all = y,
    z_all = z,
    dist = 2.0,
    num_threads = parallel::detectCores()-2
  )
})
print(time_taken)

print(features)

cat("\nTest completed successfully!\n")

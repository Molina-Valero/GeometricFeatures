# Geometric features

```r
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
sourceCpp("../src/geometric_features_optimized_ultra.cpp")

# Load dataset
tree <- read.table("../data/tree.txt", header = TRUE)

# The plot file is here: https://drive.google.com/file/d/1GszdG8J9cRZsTtRkd80HQ_a7u10uxhKd/view?usp=sharing
plot <- read.table("../data/plot.txt", header = TRUE)



# Test: Geometric features
cat("\nTest 2: Geometric Features\n")

# Calculate features within radius of 2 units
time_taken <- system.time({
  features <- geometric_features_batch(
    points = as.matrix(tree),
    dist = 0.2,
    Verticality = TRUE,
    num_threads = 1
  )
})
print(time_taken)

# write.table(features, "../data/treeFeatures.txt", row.names = FALSE)


time_taken <- system.time({
  features <- geometric_features_batch(
    points = as.matrix(plot),
    dist = 0.2,
    Verticality = TRUE,
    num_threads = parallel::detectCores()-1
  )
})
print(time_taken)

print(features)

cat("\nTest completed successfully!\n")

# write.table(features, "../data/plotFeatures.txt", row.names = FALSE)
```

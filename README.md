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
sourceCpp("../src/geometric_features_dist.cpp")
sourceCpp("../src/geometric_features_knn.cpp")

# Load dataset
tree <- read.table("../data/tree.txt", header = TRUE)

# The plot file is here: https://drive.google.com/file/d/1GszdG8J9cRZsTtRkd80HQ_a7u10uxhKd/view?usp=sharing
# plot <- read.table("../data/plot.txt", header = TRUE)



# Test: Geometric features based on distance

time_taken <- system.time({
  features_dist <- geometric_features_dist(
    points = as.matrix(tree),
    dist = 0.155,
    Verticality = TRUE,
    Surface_variation = TRUE,
    Planarity = TRUE,
    num_threads = 10
  )
})
print(time_taken)
summary(features_dist)

write.table(features_dist, "../data/treeFeatures_dist.txt", row.names = FALSE)


# Test: Geometric features based on knn

# Calculate features within radius of 2 units
time_taken <- system.time({
  features_knn <- geometric_features_knn(
    points = as.matrix(tree),
    k = 30,
    Verticality = TRUE,
    Surface_variation = TRUE,
    Planarity = TRUE,
    num_threads = 10
  )
})
print(time_taken)
summary(features_knn)


write.table(features_knn, "../data/treeFeatures_knn.txt", row.names = FALSE)

cat("\nTest completed successfully!\n")

# write.table(features, "../data/plotFeatures.txt", row.names = FALSE)


treeFeatures_CC <- read.table("../data/treeFeatures_CC.txt", header = TRUE)


# Verticality

plot(density(features_dist$Verticality, na.rm = TRUE), col = 2, lwd = 3)

lines(density(features_knn$Verticality), col = 3, lwd=3, lty = 2)

lines(density(treeFeatures_CC$Verticality_.0.155122., na.rm = TRUE), col = 4, lwd=3, lty = 3)

legend(x = "topleft",
       legend = c("Dist", "Knn", "CloudCompare"),
       lwd = 3, col = c(2, 3, 4), lty = c(1, 2, 3), bty = "n")

# Planarity

plot(density(features_dist$Planarity, na.rm = TRUE), col = 2, lwd = 3)

lines(density(features_knn$Planarity), col = 3, lwd=3, lty = 2)

lines(density(treeFeatures_CC$Planarity_.0.155122., na.rm = TRUE), col = 4, lwd=3, lty = 3)


legend(x = "topleft",
       legend = c("Dist", "Knn", "CloudCompare"),
       lwd = 3, col = c(2, 3, 4), lty = c(1, 2, 3), bty = "n")


# Surface variation

plot(density(features_dist$Surface_variation, na.rm = TRUE), col = 2, lwd = 3)

lines(density(features_knn$Surface_variation), col = 3, lwd=3, lty = 2)

lines(density(treeFeatures_CC$Surface_variation_.0.155122., na.rm = TRUE), col = 4, lwd=3, lty = 3)


legend(x = "topleft",
       legend = c("Dist", "Knn", "CloudCompare"),
       lwd = 3, col = c(2, 3, 4), lty = c(1, 2, 3), bty = "n")
```

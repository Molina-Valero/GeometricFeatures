// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <limits>
#include <array>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// ----- Type aliases -----
template <typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using PointCloud = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>; // Nx3 point cloud, RowMajor for cache locality

template <typename real_t>
struct PCAResult {
  Vec3<real_t> val;  // eigenvalues (descending)
  Vec3<real_t> v0;   // eigenvector for largest EV
  Vec3<real_t> v1;
  Vec3<real_t> v2;   // eigenvector for smallest EV (z-oriented positive)
};

// Structure to hold computed features (avoids std::map overhead)
struct GeometricFeatures {
  double point_id;
  double lambda1, lambda2, lambda3;
  double eigenvalue_sum;
  double normal_x, normal_y, normal_z;
  double pca1, pca2;
  double anisotropy, eigenentropy, linearity;
  double omnivariance, planarity, sphericity;
  double surface_variation, normal_change_rate, verticality;
  double n_points, surface_density, volume_density;
  int count;
  
  // Initialize all to NA
  GeometricFeatures() : 
    point_id(NA_REAL), lambda1(NA_REAL), lambda2(NA_REAL), lambda3(NA_REAL),
    eigenvalue_sum(NA_REAL), normal_x(NA_REAL), normal_y(NA_REAL), normal_z(NA_REAL),
    pca1(NA_REAL), pca2(NA_REAL), anisotropy(NA_REAL), eigenentropy(NA_REAL),
    linearity(NA_REAL), omnivariance(NA_REAL), planarity(NA_REAL), sphericity(NA_REAL),
    surface_variation(NA_REAL), normal_change_rate(NA_REAL), verticality(NA_REAL), 
    n_points(NA_REAL), surface_density(NA_REAL), volume_density(NA_REAL), count(0) {}
};

// ----- AUXILIARY FUNCTIONS -----

// Inline squared distance (avoid sqrt when possible)
inline double SquaredDistance(double x1, double y1, double z1,
                              double x2, double y2, double z2) {
  const double dx = x2 - x1;
  const double dy = y2 - y1;
  const double dz = z2 - z1;
  return dx * dx + dy * dy + dz * dz;
}

// Euclidean distance (use squared distance when comparing)
inline double EuclideanDistance(double x1, double y1, double z1,
                                double x2, double y2, double z2) {
  return std::sqrt(SquaredDistance(x1, y1, z1, x2, y2, z2));
}

// ----- Eigenvalue analysis (optimized) -----
template <typename real_t>
static inline PCAResult<real_t>
eigenvalue_analysis_core(const PointCloud<real_t>& cloud)
{
  const Eigen::Index N = cloud.rows();
  if (N < 2)
    Rcpp::stop("Need at least 2 points");
  
  // Compute centroid
  const Vec3<real_t> centroid = cloud.colwise().mean();
  
  // Compute covariance matrix directly without creating centered matrix (saves memory)
  Eigen::Matrix<real_t, 3, 3> cov = Eigen::Matrix<real_t, 3, 3>::Zero();
  
  for (Eigen::Index i = 0; i < N; ++i) {
    const Vec3<real_t> diff = cloud.row(i).transpose() - centroid;
    cov.noalias() += diff * diff.transpose();
  }
  cov /= static_cast<real_t>(N - 1);
  
  // Eigen-decomposition (symmetric)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<real_t, 3, 3>> es(cov);
  if (es.info() != Eigen::Success)
    Rcpp::stop("Eigen decomposition failed");
  
  const auto ev = es.eigenvalues();   // ascending
  const auto evecs = es.eigenvectors();  // columns = eigenvectors
  
  // Sort indices by descending eigenvalue (optimized sorting network for 3 elements)
  std::array<int, 3> idx{2, 1, 0}; // Start with descending order
  if (ev(1) > ev(2)) std::swap(idx[0], idx[1]);
  if (ev(0) > ev(1)) std::swap(idx[1], idx[2]);
  if (ev(1) > ev(2)) std::swap(idx[0], idx[1]);
  
  PCAResult<real_t> out;
  out.val << std::max<real_t>(ev(idx[0]), real_t(0)),
             std::max<real_t>(ev(idx[1]), real_t(0)),
             std::max<real_t>(ev(idx[2]), real_t(0));
  
  out.v0 = evecs.col(idx[0]);
  out.v1 = evecs.col(idx[1]);
  out.v2 = evecs.col(idx[2]);
  
  // Orient smallest eigenvector to have positive Z
  if (out.v2(2) < real_t(0)) out.v2 = -out.v2;
  
  // Ensure right-handed frame
  if (out.v0.cross(out.v1).dot(out.v2) < real_t(0))
    out.v1 = -out.v1;
  
  return out;
}

// ----- R interface -----
// [[Rcpp::export]]
Rcpp::List eigenvalue_analysis(Rcpp::NumericMatrix X)
{
  if (X.ncol() != 3)
    Rcpp::stop("Input must be an N x 3 matrix (point cloud)");
  
  // Check for finite values
  for (int i = 0; i < X.size(); ++i) {
    if (!R_finite(X[i]))
      Rcpp::stop("Input contains non-finite values");
  }
  
  // Map input to Eigen (row-major for better cache locality)
  using Mapped = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>;
  Mapped cloud_map(reinterpret_cast<const double*>(X.begin()), X.nrow(), 3);
  
  auto res = eigenvalue_analysis_core<double>(cloud_map);
  
  return Rcpp::List::create(
    Rcpp::Named("EigenValues") = Rcpp::NumericVector::create(res.val(0), res.val(1), res.val(2)),
    Rcpp::Named("v0")          = Rcpp::NumericVector::create(res.v0(0), res.v0(1), res.v0(2)),
    Rcpp::Named("v1")          = Rcpp::NumericVector::create(res.v1(0), res.v1(1), res.v1(2)),
    Rcpp::Named("v2")          = Rcpp::NumericVector::create(res.v2(0), res.v2(1), res.v2(2))
  );
}

// ----- Geometric features (optimized) -----
GeometricFeatures geometric_features_core(const Eigen::Map<const Eigen::VectorXd>& x,
                                          const Eigen::Map<const Eigen::VectorXd>& y,
                                          const Eigen::Map<const Eigen::VectorXd>& z,
                                          double x_pto, double y_pto, double z_pto,
                                          double point_id, double dist,
                                          bool compute_density_features,
                                          bool compute_normal_change) {
  
  GeometricFeatures features;
  features.point_id = point_id;
  
  const Eigen::Index SIZE = x.size();
  const double dist_sq = dist * dist;  // Use squared distance for comparison (avoids sqrt)
  
  // First pass: count and collect indices
  std::vector<Eigen::Index> indices_in_range;
  indices_in_range.reserve(std::min<Eigen::Index>(SIZE, 1000)); // Reserve reasonable amount
  
  for (Eigen::Index i = 0; i < SIZE; ++i) {
    const double d_sq = SquaredDistance(x_pto, y_pto, z_pto, x(i), y(i), z(i));
    if (d_sq <= dist_sq) {
      indices_in_range.push_back(i);
    }
  }
  
  const int count = static_cast<int>(indices_in_range.size());
  features.count = count;
  features.n_points = static_cast<double>(count);
  
  // Early exit if insufficient points
  if (count < 3) {
    return features;
  }
  
  // Allocate and fill neighborhood matrix efficiently
  Eigen::MatrixXd neighborhood(count, 3);
  for (int idx = 0; idx < count; ++idx) {
    const Eigen::Index i = indices_in_range[idx];
    neighborhood(idx, 0) = x(i);
    neighborhood(idx, 1) = y(i);
    neighborhood(idx, 2) = z(i);
  }
  
  // Compute PCA
  const auto pca = eigenvalue_analysis_core<double>(neighborhood);
  
  const double lambda1 = pca.val(0);
  const double lambda2 = pca.val(1);
  const double lambda3 = pca.val(2);
  const double lambda_sum = pca.val.sum();
  
  // Store eigenvalues
  features.lambda1 = lambda1;
  features.lambda2 = lambda2;
  features.lambda3 = lambda3;
  features.eigenvalue_sum = lambda_sum;
  
  // Store normal vector
  features.normal_x = pca.v2(0);
  features.normal_y = pca.v2(1);
  features.normal_z = pca.v2(2);
  
  // Compute features with safety checks (avoid division by zero)
  constexpr double epsilon = 1e-10;
  
  if (lambda_sum > epsilon) {
    features.pca1 = lambda1 / lambda_sum;
    features.pca2 = lambda2 / lambda_sum;
    features.surface_variation = lambda3 / lambda_sum;
  }
  
  if (lambda1 > epsilon) {
    features.anisotropy = (lambda1 - lambda3) / lambda1;
    features.linearity = (lambda1 - lambda2) / lambda1;
    features.planarity = (lambda2 - lambda3) / lambda1;
    features.sphericity = lambda3 / lambda1;
  }
  
  // Eigenentropy - check all eigenvalues are positive and normalize properly
  if (lambda1 > epsilon && lambda2 > epsilon && lambda3 > epsilon) {
    const double norm_l1 = lambda1 / lambda_sum;
    const double norm_l2 = lambda2 / lambda_sum;
    const double norm_l3 = lambda3 / lambda_sum;
    features.eigenentropy = -(norm_l1 * std::log(norm_l1) + 
                              norm_l2 * std::log(norm_l2) + 
                              norm_l3 * std::log(norm_l3));
  }
  
  // Omnivariance - geometric mean of eigenvalues (use cbrt for better precision)
  if (lambda1 > epsilon && lambda2 > epsilon && lambda3 > epsilon) {
    features.omnivariance = std::cbrt(lambda1 * lambda2 * lambda3);
  }
  
  // Verticality
  features.verticality = 1.0 - std::abs(pca.v2(2));
  
  // Normal change rate (optional, computationally expensive)
  if (compute_normal_change) {
    // Placeholder - would need additional computation
    features.normal_change_rate = NA_REAL;
  }
  
  // Density features (only compute if requested)
  if (compute_density_features && dist > epsilon) {
    constexpr double PI = 3.14159265358979323846;
    features.surface_density = static_cast<double>(count) / (4.0 * PI * dist * dist);
    features.volume_density = static_cast<double>(count) / ((4.0 / 3.0) * PI * dist * dist * dist);
  }
  
  return features;
}

// [[Rcpp::export]]
Rcpp::DataFrame geometric_features_batch(Rcpp::NumericMatrix points,
                                         Rcpp::NumericVector x_all,
                                         Rcpp::NumericVector y_all,
                                         Rcpp::NumericVector z_all,
                                         double dist,
                                         bool Anisotropy = true,
                                         bool Eigenentropy = true,
                                         bool Eigenvalue_sum = true,
                                         bool First_eigenvalue = true,
                                         bool Linearity = true,
                                         bool Normal_change_rate = false,
                                         bool Normal_x = true,
                                         bool Normal_y = true,
                                         bool Normal_z = true,
                                         bool Number_of_points = false,
                                         bool Omnivariance = true,
                                         bool PCA_1 = true,
                                         bool PCA_2 = true,
                                         bool Planarity = true,
                                         bool Second_eigenvalue = true,
                                         bool Sphericity = true,
                                         bool Surface_density = false,
                                         bool Surface_variation = true,
                                         bool Third_eigenvalue = true,
                                         bool Verticality = true,
                                         bool Volume_density = false,
                                         int solver_thresh = 50000,
                                         int num_threads = 0) {
  
  // Validate input
  if (points.ncol() != 4) {
    Rcpp::stop("Input 'points' must be a matrix with 4 columns: point, x, y, z");
  }
  
  const int n_points = points.nrow();
  
  if (n_points == 0) {
    return Rcpp::DataFrame();
  }
  
  // Set number of threads
#ifdef _OPENMP
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
#endif
  
  // Map the full point cloud vectors (const for safety)
  const Eigen::Map<const Eigen::VectorXd> x_map(x_all.begin(), x_all.size());
  const Eigen::Map<const Eigen::VectorXd> y_map(y_all.begin(), y_all.size());
  const Eigen::Map<const Eigen::VectorXd> z_map(z_all.begin(), z_all.size());
  
  // Check if density features are needed (to avoid unnecessary computation)
  const bool compute_density = Surface_density || Volume_density;
  const bool compute_normal_change = Normal_change_rate;
  
  // Pre-allocate result vectors (initialize with NA_REAL for optional features)
  std::vector<double> vec_point(n_points), vec_x(n_points), vec_y(n_points), vec_z(n_points);
  std::vector<double> vec_first_ev(n_points, NA_REAL), vec_second_ev(n_points, NA_REAL);
  std::vector<double> vec_third_ev(n_points, NA_REAL), vec_ev_sum(n_points, NA_REAL);
  std::vector<double> vec_normal_x(n_points, NA_REAL), vec_normal_y(n_points, NA_REAL);
  std::vector<double> vec_normal_z(n_points, NA_REAL);
  std::vector<double> vec_pca1(n_points, NA_REAL), vec_pca2(n_points, NA_REAL);
  std::vector<double> vec_anisotropy(n_points, NA_REAL), vec_eigenentropy(n_points, NA_REAL);
  std::vector<double> vec_linearity(n_points, NA_REAL), vec_omnivariance(n_points, NA_REAL);
  std::vector<double> vec_planarity(n_points, NA_REAL), vec_sphericity(n_points, NA_REAL);
  std::vector<double> vec_surface_var(n_points, NA_REAL), vec_normal_change(n_points, NA_REAL);
  std::vector<double> vec_verticality(n_points, NA_REAL);
  std::vector<double> vec_n_points(n_points, NA_REAL);
  std::vector<double> vec_surf_density(n_points, NA_REAL), vec_vol_density(n_points, NA_REAL);
  
  // Parallel loop over each point (with dynamic scheduling and chunk size optimization)
#pragma omp parallel for schedule(dynamic, 64) if(n_points > 100)
  for (int i = 0; i < n_points; ++i) {
    
    // Check for user interrupt periodically (only in single-threaded or from thread 0)
#ifdef _OPENMP
    if (omp_get_thread_num() == 0 && i % 1000 == 0) {
#else
    if (i % 1000 == 0) {
#endif
      Rcpp::checkUserInterrupt();
    }
    
    const double point_id = points(i, 0);
    const double x_pto = points(i, 1);
    const double y_pto = points(i, 2);
    const double z_pto = points(i, 3);
    
    // Compute features for this point
    const auto features = geometric_features_core(x_map, y_map, z_map,
                                                   x_pto, y_pto, z_pto,
                                                   point_id, dist,
                                                   compute_density,
                                                   compute_normal_change);
    
    // Store basic info
    vec_point[i] = features.point_id;
    vec_x[i] = x_pto;
    vec_y[i] = y_pto;
    vec_z[i] = z_pto;
    
    // Store computed features (only if requested)
    if (First_eigenvalue) vec_first_ev[i] = features.lambda1;
    if (Second_eigenvalue) vec_second_ev[i] = features.lambda2;
    if (Third_eigenvalue) vec_third_ev[i] = features.lambda3;
    if (Eigenvalue_sum) vec_ev_sum[i] = features.eigenvalue_sum;
    if (Normal_x) vec_normal_x[i] = features.normal_x;
    if (Normal_y) vec_normal_y[i] = features.normal_y;
    if (Normal_z) vec_normal_z[i] = features.normal_z;
    if (PCA_1) vec_pca1[i] = features.pca1;
    if (PCA_2) vec_pca2[i] = features.pca2;
    if (Anisotropy) vec_anisotropy[i] = features.anisotropy;
    if (Eigenentropy) vec_eigenentropy[i] = features.eigenentropy;
    if (Linearity) vec_linearity[i] = features.linearity;
    if (Omnivariance) vec_omnivariance[i] = features.omnivariance;
    if (Planarity) vec_planarity[i] = features.planarity;
    if (Sphericity) vec_sphericity[i] = features.sphericity;
    if (Surface_variation) vec_surface_var[i] = features.surface_variation;
    if (Normal_change_rate) vec_normal_change[i] = features.normal_change_rate;
    if (Verticality) vec_verticality[i] = features.verticality;
    if (Number_of_points) vec_n_points[i] = features.n_points;
    if (Surface_density) vec_surf_density[i] = features.surface_density;
    if (Volume_density) vec_vol_density[i] = features.volume_density;
  }
  
  // Build the DataFrame efficiently
  Rcpp::List result;
  result["point"] = vec_point;
  result["x"] = vec_x;
  result["y"] = vec_y;
  result["z"] = vec_z;
  
  if (Anisotropy) result["Anisotropy"] = vec_anisotropy;
  if (Eigenentropy) result["Eigenentropy"] = vec_eigenentropy;
  if (Eigenvalue_sum) result["Eigenvalue_sum"] = vec_ev_sum;
  if (First_eigenvalue) result["First_eigenvalue"] = vec_first_ev;
  if (Linearity) result["Linearity"] = vec_linearity;
  if (Normal_change_rate) result["Normal_change_rate"] = vec_normal_change;
  if (Normal_x) result["Normal_x"] = vec_normal_x;
  if (Normal_y) result["Normal_y"] = vec_normal_y;
  if (Normal_z) result["Normal_z"] = vec_normal_z;
  if (Number_of_points) result["Number_of_points"] = vec_n_points;
  if (Omnivariance) result["Omnivariance"] = vec_omnivariance;
  if (PCA_1) result["PCA_1"] = vec_pca1;
  if (PCA_2) result["PCA_2"] = vec_pca2;
  if (Planarity) result["Planarity"] = vec_planarity;
  if (Second_eigenvalue) result["Second_eigenvalue"] = vec_second_ev;
  if (Sphericity) result["Sphericity"] = vec_sphericity;
  if (Surface_density) result["Surface_density"] = vec_surf_density;
  if (Surface_variation) result["Surface_variation"] = vec_surface_var;
  if (Third_eigenvalue) result["Third_eigenvalue"] = vec_third_ev;
  if (Verticality) result["Verticality"] = vec_verticality;
  if (Volume_density) result["Volume_density"] = vec_vol_density;
  
  return Rcpp::DataFrame(result);
}

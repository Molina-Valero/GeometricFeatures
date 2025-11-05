// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;
using namespace std;
using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::VectorXd;

// ----- Type aliases -----
template <typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using PointCloud = Eigen::Matrix<T, Eigen::Dynamic, 3>;

template <typename real_t>
struct PCAResult {
  Vec3<real_t> val;
  Vec3<real_t> v0;
  Vec3<real_t> v1;
  Vec3<real_t> v2;
};

// ----- Euclidean distance -----
inline double EuclideanDistance(double x1, double y1, double z1,
                                double x2, double y2, double z2) {
  double dx = x2 - x1;
  double dy = y2 - y1;
  double dz = z2 - z1;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// ----- eigenvalue analysis -----
template <typename real_t>
static inline PCAResult<real_t>
eigenvalue_analysis_core(const PointCloud<real_t>& cloud)
{
  const auto N = cloud.rows();
  if (N < 2)
    Rcpp::stop("Need at least 2 points");
  
  // Center
  PointCloud<real_t> centered = cloud.rowwise() - cloud.colwise().mean();
  
  // Covariance (unbiased, divide by N-1)
  Eigen::Matrix<real_t,3,3> cov;
  cov.noalias() = centered.adjoint() * centered;
  cov /= static_cast<real_t>(N - 1);
  
  // Eigen-decomposition (symmetric)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<real_t,3,3>> es(cov);
  if (es.info() != Eigen::Success)
    Rcpp::stop("Eigen decomposition failed");
  
  auto ev    = es.eigenvalues();
  auto evecs = es.eigenvectors();
  
  // Sort indices by descending eigenvalue
  std::array<int,3> idx {0,1,2};
  std::sort(idx.begin(), idx.end(), [&](int i, int j){ return ev(i) > ev(j); });
  
  PCAResult<real_t> out;
  out.val << std::max<real_t>(ev(idx[0]), 0),
             std::max<real_t>(ev(idx[1]), 0),
             std::max<real_t>(ev(idx[2]), 0);
  
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

// ----- Geometric features with K-NEAREST NEIGHBORS -----
std::map<std::string, double> geometric_features_core(const Eigen::MatrixXd& xyz_all,
                                                      int point_idx,
                                                      int k_neighbors,
                                                      bool Anisotropy = false,
                                                      bool Eigenentropy = false,
                                                      bool Eigenvalue_sum = false,
                                                      bool First_eigenvalue = false,
                                                      bool Linearity = false,
                                                      bool Normal_x = false,
                                                      bool Normal_y = false,
                                                      bool Normal_z = false,
                                                      bool Number_of_points = false,
                                                      bool Omnivariance = false,
                                                      bool PCA_1 = false,
                                                      bool PCA_2 = false,
                                                      bool Planarity = false,
                                                      bool Second_eigenvalue = false,
                                                      bool Sphericity = false,
                                                      bool Surface_variation = false,
                                                      bool Third_eigenvalue = false,
                                                      bool Verticality = false) {
  
  const int SIZE = xyz_all.rows();
  double x_pto = xyz_all(point_idx, 1);
  double y_pto = xyz_all(point_idx, 2);
  double z_pto = xyz_all(point_idx, 3);
  
  // Calculate distances to all points
  std::vector<std::pair<double, int>> distances;
  distances.reserve(SIZE);
  
  for (int i = 0; i < SIZE; ++i) {
    double distance = EuclideanDistance(x_pto, y_pto, z_pto, 
                                       xyz_all(i, 1), xyz_all(i, 2), xyz_all(i, 3));
    distances.push_back(std::make_pair(distance, i));
  }
  
  // Sort to get k nearest neighbors
  int k = std::min(k_neighbors, SIZE);
  std::partial_sort(distances.begin(), 
                    distances.begin() + k, 
                    distances.end());
  
  std::map<std::string, double> features;
  features["point"] = xyz_all(point_idx, 0);
  
  // Not enough neighbors
  if (k < 3) {
    if (Anisotropy) features["Anisotropy"] = NA_REAL;
    if (Eigenentropy) features["Eigenentropy"] = NA_REAL;
    if (Eigenvalue_sum) features["Eigenvalue_sum"] = NA_REAL;
    if (First_eigenvalue) features["First_eigenvalue"] = NA_REAL;
    if (Linearity) features["Linearity"] = NA_REAL;
    if (Normal_x) features["Normal_x"] = NA_REAL;
    if (Normal_y) features["Normal_y"] = NA_REAL;
    if (Normal_z) features["Normal_z"] = NA_REAL;
    if (Number_of_points) features["Number_of_points"] = NA_REAL;
    if (Omnivariance) features["Omnivariance"] = NA_REAL;
    if (PCA_1) features["PCA_1"] = NA_REAL;
    if (PCA_2) features["PCA_2"] = NA_REAL;
    if (Planarity) features["Planarity"] = NA_REAL;
    if (Second_eigenvalue) features["Second_eigenvalue"] = NA_REAL;
    if (Sphericity) features["Sphericity"] = NA_REAL;
    if (Surface_variation) features["Surface_variation"] = NA_REAL;
    if (Third_eigenvalue) features["Third_eigenvalue"] = NA_REAL;
    if (Verticality) features["Verticality"] = NA_REAL;
    return features;
  }
  
  // Build local point cloud from k nearest neighbors
  Eigen::MatrixXd local_cloud(k, 3);
  for (int i = 0; i < k; ++i) {
    int idx = distances[i].second;
    local_cloud(i, 0) = xyz_all(idx, 1);
    local_cloud(i, 1) = xyz_all(idx, 2);
    local_cloud(i, 2) = xyz_all(idx, 3);
  }
  
  // PCA
  auto pca = eigenvalue_analysis_core<double>(local_cloud);
  
  double lambda1 = pca.val(0);
  double lambda2 = pca.val(1);
  double lambda3 = pca.val(2);
  
  // Compute features
  if (Anisotropy) features["Anisotropy"] = (lambda1 - lambda3) / lambda1;
  if (Eigenentropy) features["Eigenentropy"] = -(lambda1 * log(lambda1 + 1e-10) + lambda2 * log(lambda2 + 1e-10) + lambda3 * log(lambda3 + 1e-10));
  if (Eigenvalue_sum) features["Eigenvalue_sum"] = pca.val.sum();
  if (First_eigenvalue) features["First_eigenvalue"] = lambda1;
  if (Linearity) features["Linearity"] = (lambda1 - lambda2) / lambda1;
  if (Normal_x) features["Normal_x"] = pca.v2(0);
  if (Normal_y) features["Normal_y"] = pca.v2(1);
  if (Normal_z) features["Normal_z"] = pca.v2(2);
  if (Number_of_points) features["Number_of_points"] = static_cast<double>(k);
  if (Omnivariance) features["Omnivariance"] = pow(lambda1 * lambda2 * lambda3, 1.0 / 3.0);
  if (PCA_1) features["PCA_1"] = lambda1 / pca.val.sum();
  if (PCA_2) features["PCA_2"] = lambda2 / pca.val.sum();
  if (Planarity) features["Planarity"] = (lambda2 - lambda3) / lambda1;
  if (Second_eigenvalue) features["Second_eigenvalue"] = lambda2;
  if (Sphericity) features["Sphericity"] = lambda3 / lambda1;
  if (Surface_variation) features["Surface_variation"] = lambda3 / pca.val.sum();
  if (Third_eigenvalue) features["Third_eigenvalue"] = lambda3;
  if (Verticality) features["Verticality"] = 1.0 - abs(pca.v2(2));

  return features;
}


// [[Rcpp::export]]
Rcpp::DataFrame geometric_features_knn(Rcpp::NumericMatrix points,
                                       int k = 30,
                                       bool Anisotropy = false,
                                   bool Eigenentropy = false,
                                   bool Eigenvalue_sum = false,
                                   bool First_eigenvalue = false,
                                   bool Linearity = false,
                                   bool Normal_x = false,
                                   bool Normal_y = false,
                                   bool Normal_z = false,
                                   bool Number_of_points = false,
                                   bool Omnivariance = false,
                                   bool PCA_1 = false,
                                   bool PCA_2 = false,
                                   bool Planarity = false,
                                   bool Second_eigenvalue = false,
                                   bool Sphericity = false,
                                   bool Surface_variation = false,
                                   bool Third_eigenvalue = false,
                                   bool Verticality = false,
                                   int num_threads = 0) {
  
  // Validate input
  if (points.ncol() != 4) {
    Rcpp::stop("Input 'points' must be a matrix with 4 columns: point, x, y, z");
  }
  
  const int n_points = points.nrow();
  
  // Set number of threads
#ifdef _OPENMP
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
#endif
  
  // Convert to Eigen matrix
  Eigen::Map<Eigen::MatrixXd> xyz_map(points.begin(), n_points, 4);
  
  // Pre-allocate vectors to store results
  std::vector<double> vec_point(n_points), vec_x(n_points), vec_y(n_points), vec_z(n_points);
  std::vector<double> vec_first_ev(n_points), vec_second_ev(n_points), vec_third_ev(n_points), vec_ev_sum(n_points);
  std::vector<double> vec_normal_x(n_points), vec_normal_y(n_points), vec_normal_z(n_points);
  std::vector<double> vec_pca1(n_points), vec_pca2(n_points);
  std::vector<double> vec_anisotropy(n_points), vec_eigenentropy(n_points), vec_linearity(n_points);
  std::vector<double> vec_omnivariance(n_points), vec_planarity(n_points), vec_sphericity(n_points);
  std::vector<double> vec_surface_var(n_points), vec_verticality(n_points);
  std::vector<double> vec_n_points(n_points);
  
  // Parallel loop over each point
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n_points; ++i) {
    
    // Check for user interrupt periodically
    if (i % 1000 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // Compute features for this point
    auto features = geometric_features_core(xyz_map, i, k,
                                           Anisotropy, Eigenentropy, Eigenvalue_sum,
                                           First_eigenvalue, Linearity,
                                           Normal_x, Normal_y, Normal_z,
                                           Number_of_points, Omnivariance, PCA_1, PCA_2,
                                           Planarity, Second_eigenvalue, Sphericity,
                                           Surface_variation, Third_eigenvalue,
                                           Verticality);
    
    // Store results
    vec_point[i] = features["point"];
    vec_x[i] = points(i, 1);
    vec_y[i] = points(i, 2);
    vec_z[i] = points(i, 3);
    
    if (Anisotropy) vec_anisotropy[i] = features["Anisotropy"];
    if (Eigenentropy) vec_eigenentropy[i] = features["Eigenentropy"];
    if (Eigenvalue_sum) vec_ev_sum[i] = features["Eigenvalue_sum"];
    if (First_eigenvalue) vec_first_ev[i] = features["First_eigenvalue"];
    if (Linearity) vec_linearity[i] = features["Linearity"];
    if (Normal_x) vec_normal_x[i] = features["Normal_x"];
    if (Normal_y) vec_normal_y[i] = features["Normal_y"];
    if (Normal_z) vec_normal_z[i] = features["Normal_z"];
    if (Number_of_points) vec_n_points[i] = features["Number_of_points"];
    if (Omnivariance) vec_omnivariance[i] = features["Omnivariance"];
    if (PCA_1) vec_pca1[i] = features["PCA_1"];
    if (PCA_2) vec_pca2[i] = features["PCA_2"];
    if (Planarity) vec_planarity[i] = features["Planarity"];
    if (Second_eigenvalue) vec_second_ev[i] = features["Second_eigenvalue"];
    if (Sphericity) vec_sphericity[i] = features["Sphericity"];
    if (Surface_variation) vec_surface_var[i] = features["Surface_variation"];
    if (Third_eigenvalue) vec_third_ev[i] = features["Third_eigenvalue"];
    if (Verticality) vec_verticality[i] = features["Verticality"];
  }
  
  // Build the DataFrame
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
  if (Surface_variation) result["Surface_variation"] = vec_surface_var;
  if (Third_eigenvalue) result["Third_eigenvalue"] = vec_third_ev;
  if (Verticality) result["Verticality"] = vec_verticality;
  
  return Rcpp::DataFrame(result);
}

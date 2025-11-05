// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <limits>
#include <array>
#include <algorithm>
#include <queue>
#include <memory>
#include <iomanip>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// ----- PROGRESS BAR UTILITY -----
class ProgressBar {
private:
  int total_;
  int bar_width_;
  int last_percent_;
  
public:
  ProgressBar(int total, int bar_width = 50) 
    : total_(total), bar_width_(bar_width), last_percent_(-1) {}
  
  void update(int current) {
    int percent = static_cast<int>((current * 100.0) / total_);
    
    if (percent == last_percent_) return;
    last_percent_ = percent;
    
    int filled = static_cast<int>((current * bar_width_) / total_);
    
    std::ostringstream bar;
    bar << "\r[";
    for (int i = 0; i < bar_width_; ++i) {
      if (i < filled) bar << "=";
      else if (i == filled) bar << ">";
      else bar << " ";
    }
    bar << "] " << percent << "% (" << current << "/" << total_ << ")";
    
    Rcpp::Rcout << bar.str() << std::flush;
  }
  
  void finish() {
    Rcpp::Rcout << std::endl;
  }
};

// ----- Type aliases -----
template <typename T>
using Vec3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
using PointCloud = Eigen::Matrix<T, Eigen::Dynamic, 3, Eigen::RowMajor>;

template <typename real_t>
struct PCAResult {
  Vec3<real_t> val;
  Vec3<real_t> v0, v1, v2;
};

struct GeometricFeatures {
  double point_id;
  double lambda1, lambda2, lambda3;
  double eigenvalue_sum;
  double normal_x, normal_y, normal_z;
  double pca1, pca2;
  double anisotropy, eigenentropy, linearity;
  double omnivariance, planarity, sphericity;
  double surface_variation, verticality;
  double n_points;
  int count;
  
  GeometricFeatures() : 
    point_id(NA_REAL), lambda1(NA_REAL), lambda2(NA_REAL), lambda3(NA_REAL),
    eigenvalue_sum(NA_REAL), normal_x(NA_REAL), normal_y(NA_REAL), normal_z(NA_REAL),
    pca1(NA_REAL), pca2(NA_REAL), anisotropy(NA_REAL), eigenentropy(NA_REAL),
    linearity(NA_REAL), omnivariance(NA_REAL), planarity(NA_REAL), sphericity(NA_REAL),
    surface_variation(NA_REAL), verticality(NA_REAL), n_points(NA_REAL), count(0) {}
};

// ----- AUXILIARY FUNCTIONS -----

inline double SquaredDistance(double x1, double y1, double z1,
                              double x2, double y2, double z2) {
  const double dx = x2 - x1;
  const double dy = y2 - y1;
  const double dz = z2 - z1;
  return dx * dx + dy * dy + dz * dz;
}

// ----- Eigenvalue analysis (optimized) -----
template <typename real_t>
static inline PCAResult<real_t>
eigenvalue_analysis_core(const PointCloud<real_t>& cloud)
{
  const Eigen::Index N = cloud.rows();
  if (N < 2) Rcpp::stop("Need at least 2 points");
  
  const Vec3<real_t> centroid = cloud.colwise().mean();
  
  // Covariance matrix computation
  Eigen::Matrix<real_t, 3, 3> cov = Eigen::Matrix<real_t, 3, 3>::Zero();
  
  for (Eigen::Index i = 0; i < N; ++i) {
    const Vec3<real_t> diff = cloud.row(i).transpose() - centroid;
    cov.noalias() += diff * diff.transpose();
  }
  cov /= static_cast<real_t>(N - 1);
  
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<real_t, 3, 3>> es(cov);
  if (es.info() != Eigen::Success) Rcpp::stop("Eigen decomposition failed");
  
  const auto ev = es.eigenvalues();
  const auto evecs = es.eigenvectors();
  
  // Sort by descending eigenvalue
  std::array<int, 3> idx{2, 1, 0};
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
  
  if (out.v2(2) < real_t(0)) out.v2 = -out.v2;
  if (out.v0.cross(out.v1).dot(out.v2) < real_t(0)) out.v1 = -out.v1;
  
  return out;
}

// ----- Core feature computation with KNN -----
GeometricFeatures geometric_features_knn_core(
    const Eigen::Map<const Eigen::VectorXd>& x,
    const Eigen::Map<const Eigen::VectorXd>& y,
    const Eigen::Map<const Eigen::VectorXd>& z,
    double x_pto, double y_pto, double z_pto,
    double point_id, int k_neighbors) {
  
  const int SIZE = x.size();
  constexpr double epsilon = 1e-10;
  
  GeometricFeatures features;
  features.point_id = point_id;
  
  // Use priority queue to find k nearest neighbors efficiently
  std::priority_queue<std::pair<double, int>> knn_queue;
  
  // Find k nearest neighbors
  for (int i = 0; i < SIZE; ++i) {
    const double sq_dist = SquaredDistance(x_pto, y_pto, z_pto, x[i], y[i], z[i]);
    
    if (static_cast<int>(knn_queue.size()) < k_neighbors) {
      knn_queue.push(std::make_pair(sq_dist, i));
    } else if (sq_dist < knn_queue.top().first) {
      knn_queue.pop();
      knn_queue.push(std::make_pair(sq_dist, i));
    }
  }
  
  const int count = static_cast<int>(knn_queue.size());
  features.count = count;
  features.n_points = static_cast<double>(count);
  
  if (count < 3) {
    return features;  // All values remain NA_REAL
  }
  
  // Extract neighborhood points
  PointCloud<double> neighborhood(count, 3);
  int idx = count - 1;
  while (!knn_queue.empty()) {
    const int point_idx = knn_queue.top().second;
    neighborhood(idx, 0) = x[point_idx];
    neighborhood(idx, 1) = y[point_idx];
    neighborhood(idx, 2) = z[point_idx];
    knn_queue.pop();
    --idx;
  }
  
  // PCA
  const auto pca = eigenvalue_analysis_core<double>(neighborhood);
  
  const double lambda1 = pca.val(0);
  const double lambda2 = pca.val(1);
  const double lambda3 = pca.val(2);
  const double lambda_sum = lambda1 + lambda2 + lambda3;
  
  features.lambda1 = lambda1;
  features.lambda2 = lambda2;
  features.lambda3 = lambda3;
  features.eigenvalue_sum = lambda_sum;
  
  features.normal_x = pca.v2(0);
  features.normal_y = pca.v2(1);
  features.normal_z = pca.v2(2);
  
  features.pca1 = lambda1 / lambda_sum;
  features.pca2 = lambda2 / lambda_sum;
  
  // Geometric features
  if (lambda_sum > epsilon) {
    features.linearity = (lambda1 - lambda2) / lambda_sum;
    features.planarity = (lambda2 - lambda3) / lambda_sum;
    features.sphericity = lambda3 / lambda_sum;
    features.anisotropy = (lambda1 - lambda3) / lambda_sum;
    features.surface_variation = lambda3 / lambda_sum;
  }
  
  if (lambda1 > epsilon && lambda2 > epsilon && lambda3 > epsilon) {
    const double norm_l1 = lambda1 / lambda_sum;
    const double norm_l2 = lambda2 / lambda_sum;
    const double norm_l3 = lambda3 / lambda_sum;
    features.eigenentropy = -(norm_l1 * std::log(norm_l1) + 
                              norm_l2 * std::log(norm_l2) + 
                              norm_l3 * std::log(norm_l3));
    features.omnivariance = std::cbrt(lambda1 * lambda2 * lambda3);
  }
  
  features.verticality = 1.0 - std::abs(pca.v2(2));
  
  return features;
}

// [[Rcpp::export]]
Rcpp::DataFrame geometric_features_knn(
    Rcpp::NumericMatrix points,
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
  
  if (points.ncol() != 4) {
    Rcpp::stop("Input 'points' must be a matrix with 4 columns: point, x, y, z");
  }
  
  const int n_points = points.nrow();
  if (n_points == 0) return Rcpp::DataFrame();
  
  // Ensure k is valid
  const int effective_k = std::min(k, n_points);
  
#ifdef _OPENMP
  if (num_threads > 0) {
    omp_set_num_threads(num_threads);
  }
#endif
  
  // Extract coordinates
  Rcpp::NumericVector x_all(n_points);
  Rcpp::NumericVector y_all(n_points);
  Rcpp::NumericVector z_all(n_points);
  
  for (int i = 0; i < n_points; ++i) {
    x_all[i] = points(i, 1);
    y_all[i] = points(i, 2);
    z_all[i] = points(i, 3);
  }
  
  const Eigen::Map<const Eigen::VectorXd> x_map(x_all.begin(), x_all.size());
  const Eigen::Map<const Eigen::VectorXd> y_map(y_all.begin(), y_all.size());
  const Eigen::Map<const Eigen::VectorXd> z_map(z_all.begin(), z_all.size());
  
  // Pre-allocate result vectors
  std::vector<double> vec_point(n_points), vec_x(n_points), vec_y(n_points), vec_z(n_points);
  std::vector<double> vec_first_ev(n_points, NA_REAL), vec_second_ev(n_points, NA_REAL);
  std::vector<double> vec_third_ev(n_points, NA_REAL), vec_ev_sum(n_points, NA_REAL);
  std::vector<double> vec_normal_x(n_points, NA_REAL), vec_normal_y(n_points, NA_REAL);
  std::vector<double> vec_normal_z(n_points, NA_REAL);
  std::vector<double> vec_pca1(n_points, NA_REAL), vec_pca2(n_points, NA_REAL);
  std::vector<double> vec_anisotropy(n_points, NA_REAL), vec_eigenentropy(n_points, NA_REAL);
  std::vector<double> vec_linearity(n_points, NA_REAL), vec_omnivariance(n_points, NA_REAL);
  std::vector<double> vec_planarity(n_points, NA_REAL), vec_sphericity(n_points, NA_REAL);
  std::vector<double> vec_surface_var(n_points, NA_REAL), vec_verticality(n_points, NA_REAL);
  std::vector<double> vec_n_points(n_points, NA_REAL);
  
  // Process points in parallel with progress bar
  Rcpp::Rcout << "Computing geometric features (KNN)..." << std::endl;
  ProgressBar compute_progress(n_points);
  
#pragma omp parallel for schedule(dynamic, 64) if(n_points > 100)
  for (int i = 0; i < n_points; ++i) {
    
    // Update progress bar (only from master thread to avoid conflicts)
#ifdef _OPENMP
    if (omp_get_thread_num() == 0 && i % 1000 == 0) {
      compute_progress.update(i);
    }
#else
    if (i % 1000 == 0) {
      compute_progress.update(i);
      Rcpp::checkUserInterrupt();
    }
#endif
    
    const double point_id = points(i, 0);
    const double x_pto = points(i, 1);
    const double y_pto = points(i, 2);
    const double z_pto = points(i, 3);
    
    // Compute features using KNN
    const auto features = geometric_features_knn_core(
      x_map, y_map, z_map,
      x_pto, y_pto, z_pto, point_id, effective_k);
    
    // Store results
    vec_point[i] = features.point_id;
    vec_x[i] = x_pto;
    vec_y[i] = y_pto;
    vec_z[i] = z_pto;
    
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
    if (Verticality) vec_verticality[i] = features.verticality;
    if (Number_of_points) vec_n_points[i] = features.n_points;
  }
  
  compute_progress.update(n_points);
  compute_progress.finish();
  
  // Build DataFrame
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

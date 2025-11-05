// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <limits>  // For NaN
#include <map>
#include <string>

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
using PointCloud = Eigen::Matrix<T, Eigen::Dynamic, 3>; // Nx3 point cloud

template <typename real_t>
struct PCAResult {
  Vec3<real_t> val;  // eigenvalues (descending)
  Vec3<real_t> v0;   // eigenvector for largest EV
  Vec3<real_t> v1;
  Vec3<real_t> v2;   // eigenvector for smallest EV (z-oriented positive)
};



// ----- AUXILIARY FUNCTIONS -----

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
  
  auto ev    = es.eigenvalues();   // ascending
  auto evecs = es.eigenvectors();  // columns = eigenvectors
  
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

// ----- R interface -----
// [[Rcpp::export]]
Rcpp::List eigenvalue_analysis(Rcpp::NumericMatrix X)
{
  if (X.ncol() != 3)
    Rcpp::stop("Input must be an N x 3 matrix (point cloud)");
  
  for (int i = 0; i < X.size(); ++i)
    if (!R_finite(X[i]))
      Rcpp::stop("Input contains non-finite values");
    
    using Mapped = Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor>>;
    Mapped cloud_map(reinterpret_cast<const double*>(X.begin()), X.nrow(), 3);
    
    auto res = eigenvalue_analysis_core<double>(cloud_map);
    
    return Rcpp::List::create(
      Rcpp::Named("EigenValues") = Rcpp::NumericVector::create(res.val(0), res.val(1), res.val(2)),
      Rcpp::Named("v0")          = Rcpp::NumericVector::create(res.v0(0), res.v0(1), res.v0(2)),
      Rcpp::Named("v1")          = Rcpp::NumericVector::create(res.v1(0), res.v1(1), res.v1(2)),
      Rcpp::Named("v2")          = Rcpp::NumericVector::create(res.v2(0), res.v2(1), res.v2(2))
    );
}



// ----- Geometric features -----
std::map<std::string, double> geometric_features_core(const Eigen::Map<Eigen::VectorXd>& x,
                                                      const Eigen::Map<Eigen::VectorXd>& y,
                                                      const Eigen::Map<Eigen::VectorXd>& z,
                                                      double x_pto,
                                                      double y_pto,
                                                      double z_pto,
                                                      double point_pto,
                                                      double dist,
                                                      bool Anisotropy = false,
                                                      bool Eigenentropy = false,
                                                      bool Eigenvalue_sum = false,
                                                      bool First_eigenvalue = false,
                                                      bool Linearity = false,
                                                      bool Normal_change_rate = false,
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
                                                      bool Surface_density = false,
                                                      bool Surface_variation = false,
                                                      bool Third_eigenvalue = false,
                                                      bool Verticality = false,
                                                      bool Volume_density = false,
                                                      int solver_thresh = 50000) {
  const int SIZE = x.size();
  
  // First pass: count points within distance
  int count = 0;
  std::vector<int> indices_in_range;
  indices_in_range.reserve(SIZE);
  
  // Collect indices of points within the distance from the point of interest
  for (int i = 0; i < SIZE; ++i) {
    double distance = EuclideanDistance(x_pto, y_pto, z_pto, x[i], y[i], z[i]);
    
    if (distance <= dist) {
      indices_in_range.push_back(i);
      ++count;
    }
  }
  
  // Initialize feature map
  std::map<std::string, double> features;
  features["point"] = point_pto;
  
  // Insufficient points for eigenanalysis
  if (count < 3) {
    if (Anisotropy) features["Anisotropy"] = NA_REAL;
    if (Eigenentropy) features["Eigenentropy"] = NA_REAL;
    if (Eigenvalue_sum) features["Eigenvalue_sum"] = NA_REAL;
    if (First_eigenvalue) features["First_eigenvalue"] = NA_REAL;
    if (Linearity) features["Linearity"] = NA_REAL;
    if (Normal_change_rate) features["Normal_change_rate"] = NA_REAL;
    if (Normal_x) features["Normal_x"] = NA_REAL;
    if (Normal_y) features["Normal_y"] = NA_REAL;
    if (Normal_z) features["Normal_z"] = NA_REAL;
    if (Number_of_points) features["Number_of_points"] = static_cast<double>(count);
    if (Omnivariance) features["Omnivariance"] = NA_REAL;
    if (PCA_1) features["PCA_1"] = NA_REAL;
    if (PCA_2) features["PCA_2"] = NA_REAL;
    if (Planarity) features["Planarity"] = NA_REAL;
    if (Second_eigenvalue) features["Second_eigenvalue"] = NA_REAL;
    if (Sphericity) features["Sphericity"] = NA_REAL;
    if (Surface_density) features["Surface_density"] = (count > 0) ? static_cast<double>(count) / (4.0 * M_PI * std::pow(dist, 2.0)) : NA_REAL;
    if (Surface_variation) features["Surface_variation"] = NA_REAL;
    if (Third_eigenvalue) features["Third_eigenvalue"] = NA_REAL;
    if (Verticality) features["Verticality"] = NA_REAL;
    if (Volume_density) features["Volume_density"] = (count > 0) ? static_cast<double>(count) / ((4.0/3.0) * M_PI * std::pow(dist, 3.0)) : NA_REAL;
    return features;
  }
  
  // Extract points within range
  PointCloud<double> cloud(count, 3);
  for (int i = 0; i < count; ++i) {
    int idx = indices_in_range[i];
    cloud(i, 0) = x[idx];
    cloud(i, 1) = y[idx];
    cloud(i, 2) = z[idx];
  }
  
  // Perform PCA
  auto pca = eigenvalue_analysis_core<double>(cloud);
  
  // Eigenvalues
  double lambda1 = pca.val(0);
  double lambda2 = pca.val(1);
  double lambda3 = pca.val(2);
  
  // Compute features only if requested
  if (Anisotropy) features["Anisotropy"] = (lambda1 - lambda3) / lambda1;
  if (Eigenentropy) features["Eigenentropy"] = -(lambda1 * log(lambda1) + lambda2 * log(lambda2) + lambda3 * log(lambda3));
  if (Eigenvalue_sum) features["Eigenvalue_sum"] = pca.val.sum();
  if (First_eigenvalue) features["First_eigenvalue"] = lambda1;
  if (Linearity) features["Linearity"] = (lambda1 - lambda2) / lambda1;
  if (Normal_x) features["Normal_x"] = pca.v2(0);
  if (Normal_y) features["Normal_y"] = pca.v2(1);
  if (Normal_z) features["Normal_z"] = pca.v2(2);
  if (Number_of_points) features["Number_of_points"] = static_cast<double>(count);
  if (Omnivariance) features["Omnivariance"] = pow(lambda1 * lambda2 * lambda3, 1.0 / 3.0);
  if (PCA_1) features["PCA_1"] = lambda1 / pca.val.sum();
  if (PCA_2) features["PCA_2"] = lambda2 / pca.val.sum();
  if (Planarity) features["Planarity"] = (lambda2 - lambda3) / lambda1;
  if (Second_eigenvalue) features["Second_eigenvalue"] = lambda2;
  if (Sphericity) features["Sphericity"] = lambda3 / lambda1;
  if (Surface_density) features["Surface_density"] = static_cast<double>(count) / (4.0 * M_PI * std::pow(dist, 2.0));
  if (Surface_variation) features["Surface_variation"] = lambda3 / pca.val.sum();
  if (Third_eigenvalue) features["Third_eigenvalue"] = lambda3;
  if (Verticality) features["Verticality"] = 1.0 - abs(pca.v2(2));
  if (Volume_density) features["Volume_density"] = static_cast<double>(count) / ((4.0/3.0) * M_PI * std::pow(dist, 3.0));

  return features;
  
}


// [[Rcpp::export]]
Rcpp::DataFrame geometric_features_dist(Rcpp::NumericMatrix points,
                                        double dist,
                                        bool Anisotropy = false,
                                        bool Eigenentropy = false,
                                        bool Eigenvalue_sum = false,
                                        bool First_eigenvalue = false,
                                        bool Linearity = false,
                                        bool Normal_change_rate = false,
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
                                        bool Surface_density = false,
                                        bool Surface_variation = false,
                                        bool Third_eigenvalue = false,
                                        bool Verticality = false,
                                        bool Volume_density = false,
                                        int solver_thresh = 50000,
                                        int num_threads = 0) {
  
  // Validate input - expecting 4 columns (point, x, y, z)
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
  
  // Extract X, Y, Z columns from the matrix (columns 1, 2, 3)
  Rcpp::NumericVector x_all(n_points);
  Rcpp::NumericVector y_all(n_points);
  Rcpp::NumericVector z_all(n_points);
  
  for (int i = 0; i < n_points; ++i) {
    x_all[i] = points(i, 1);  // x column
    y_all[i] = points(i, 2);  // y column
    z_all[i] = points(i, 3);  // z column
  }
  
  // Map the full point cloud vectors
  Eigen::Map<Eigen::VectorXd> x_map(x_all.begin(), x_all.size());
  Eigen::Map<Eigen::VectorXd> y_map(y_all.begin(), y_all.size());
  Eigen::Map<Eigen::VectorXd> z_map(z_all.begin(), z_all.size());
  
  // Pre-allocate vectors to store results
  std::vector<double> vec_point(n_points), vec_x(n_points), vec_y(n_points), vec_z(n_points);
  std::vector<double> vec_first_ev(n_points), vec_second_ev(n_points), vec_third_ev(n_points), vec_ev_sum(n_points);
  std::vector<double> vec_normal_x(n_points), vec_normal_y(n_points), vec_normal_z(n_points);
  std::vector<double> vec_pca1(n_points), vec_pca2(n_points);
  std::vector<double> vec_anisotropy(n_points), vec_eigenentropy(n_points), vec_linearity(n_points);
  std::vector<double> vec_omnivariance(n_points), vec_planarity(n_points), vec_sphericity(n_points);
  std::vector<double> vec_surface_var(n_points), vec_normal_change(n_points), vec_verticality(n_points);
  std::vector<double> vec_n_points(n_points), vec_surf_density(n_points), vec_vol_density(n_points);
  
  // Parallel loop over each point
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n_points; ++i) {
    
    // Check for user interrupt periodically
    if (i % 1000 == 0) {
      Rcpp::checkUserInterrupt();
    }
    
    // Use point ID from the first column
    double point_id = points(i, 0);
    double x_pto = points(i, 1);
    double y_pto = points(i, 2);
    double z_pto = points(i, 3);
    
    // Compute features for this point
    auto features = geometric_features_core(x_map, y_map, z_map,
                                            x_pto, y_pto, z_pto,
                                            point_id, dist,
                                            Anisotropy, Eigenentropy, Eigenvalue_sum,
                                            First_eigenvalue, Linearity, Normal_change_rate,
                                            Normal_x, Normal_y, Normal_z,
                                            Number_of_points, Omnivariance, PCA_1, PCA_2,
                                            Planarity, Second_eigenvalue, Sphericity,
                                            Surface_density, Surface_variation, Third_eigenvalue,
                                            Verticality, Volume_density,
                                            solver_thresh);
    
    // Store results directly into pre-allocated vectors
    vec_point[i] = features["point"];
    vec_x[i] = x_pto;
    vec_y[i] = y_pto;
    vec_z[i] = z_pto;
    
    if (Anisotropy) vec_anisotropy[i] = features["Anisotropy"];
    if (Eigenentropy) vec_eigenentropy[i] = features["Eigenentropy"];
    if (Eigenvalue_sum) vec_ev_sum[i] = features["Eigenvalue_sum"];
    if (First_eigenvalue) vec_first_ev[i] = features["First_eigenvalue"];
    if (Linearity) vec_linearity[i] = features["Linearity"];
    if (Normal_change_rate) vec_normal_change[i] = features["Normal_change_rate"];
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
    if (Surface_density) vec_surf_density[i] = features["Surface_density"];
    if (Surface_variation) vec_surface_var[i] = features["Surface_variation"];
    if (Third_eigenvalue) vec_third_ev[i] = features["Third_eigenvalue"];
    if (Verticality) vec_verticality[i] = features["Verticality"];
    if (Volume_density) vec_vol_density[i] = features["Volume_density"];
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

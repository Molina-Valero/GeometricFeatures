// [[Rcpp::depends(RcppEigen)]]
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <limits>  // For NaN
#include <map>
#include <string>

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
                                                      bool First_eigenvalue = true,
                                                      bool Second_eigenvalue = true,
                                                      bool Third_eigenvalue = true,
                                                      bool Eigenvalue_sum = true,
                                                      bool Normal_x = true,
                                                      bool Normal_y = true,
                                                      bool Normal_z = true,
                                                      bool PCA_1 = true,
                                                      bool PCA_2 = true,
                                                      bool Anisotropy = true,
                                                      bool Eigenentropy = true,
                                                      bool Linearity = true,
                                                      bool Omnivariance = true,
                                                      bool Planarity = true,
                                                      bool Sphericity = true,
                                                      bool Surface_variation = true,
                                                      bool Normal_change_rate = false,
                                                      bool Verticality = true,
                                                      bool Number_of_points = false,
                                                      bool surface_density = false,
                                                      bool volume_density = false,
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

  // Assign features to a map
  std::map<std::string, double> features;

  // Return NA for all features if there are fewer than 3 points
  if (count < 3) {

    features["point"] = point_pto;
    if (First_eigenvalue) features["First_eigenvalue"] = NA_REAL;
    if (Second_eigenvalue) features["Second_eigenvalue"] = NA_REAL;
    if (Third_eigenvalue) features["Third_eigenvalue"] = NA_REAL;
    if (Eigenvalue_sum) features["Eigenvalue_sum"] = NA_REAL;
    if (Normal_x) features["Normal_x"] = NA_REAL;
    if (Normal_y) features["Normal_y"] = NA_REAL;
    if (Normal_z) features["Normal_z"] = NA_REAL;
    if (PCA_1) features["PCA_1"] = NA_REAL;
    if (PCA_2) features["PCA_2"] = NA_REAL;
    if (Anisotropy) features["Anisotropy"] = NA_REAL;
    if (Eigenentropy) features["Eigenentropy"] = NA_REAL;
    if (Linearity) features["Linearity"] = NA_REAL;
    if (Omnivariance) features["Omnivariance"] = NA_REAL;
    if (Planarity) features["Planarity"] = NA_REAL;
    if (Sphericity) features["Sphericity"] = NA_REAL;
    if (Surface_variation) features["Surface_variation"] = NA_REAL;
    if (Normal_change_rate) features["Normal_change_rate"] = NA_REAL;
    if (Verticality) features["Verticality"] = NA_REAL;
    if (Number_of_points) features["Number_of_points"] = NA_REAL;
    if (surface_density) features["surface_density"] = NA_REAL;
    if (volume_density) features["volume_density"] = NA_REAL;

    return features;
  }

  // Allocate matrix (m) based on the number of points within range
  Eigen::MatrixXd m(count, 3);

  // Fill the matrix (m) with the coordinates of points within the range
  for (int idx = 0; idx < count; ++idx) {
    int i = indices_in_range[idx];
    m(idx, 0) = x[i];
    m(idx, 1) = y[i];
    m(idx, 2) = z[i];
  }

  // call your robust PCA core (returns descending eigenvalues in val)
  auto pca = eigenvalue_analysis_core<double>(m);

  double lambda1 = pca.val(0);
  double lambda2 = pca.val(1);
  double lambda3 = pca.val(2);

  features["point"] = point_pto;
  if (First_eigenvalue) features["First_eigenvalue"] = lambda1;
  if (Second_eigenvalue) features["Second_eigenvalue"] = lambda2;
  if (Third_eigenvalue) features["Third_eigenvalue"] = lambda3;
  if (Eigenvalue_sum) features["Eigenvalue_sum"] = pca.val.sum();

  if (Normal_x) features["Normal_x"] = pca.v2(0);
  if (Normal_y) features["Normal_y"] = pca.v2(1);
  if (Normal_z) features["Normal_z"] = pca.v2(2);

  if (PCA_1) features["PCA_1"] = (PCA_1) ? lambda1 / pca.val.sum() : NA_REAL;
  if (PCA_2) features["PCA_2"] = (PCA_2) ? lambda2 / pca.val.sum() : NA_REAL;

  if (Anisotropy) features["Anisotropy"] = (Anisotropy) ? (lambda1 - lambda3) / lambda1 : NA_REAL;
  if (Eigenentropy) features["Eigenentropy"] = (Eigenentropy) ? -(lambda1 * log(lambda1) + lambda2 * log(lambda2) + lambda3 * log(lambda3)) : NA_REAL;
  if (Linearity) features["Linearity"] = (Linearity) ? (lambda1 - lambda2) / lambda1 : NA_REAL;
  if (Omnivariance) features["Omnivariance"] = (Omnivariance) ? pow(lambda1 * lambda2 * lambda3, 1.0 / 3.0) : NA_REAL;
  if (Planarity) features["Planarity"] = (Planarity) ? (lambda2 - lambda3) / lambda1 : NA_REAL;
  if (Sphericity) features["Sphericity"] = (Sphericity) ? lambda3 / lambda1 : NA_REAL;
  if (surface_density) features["surface_density"] = (surface_density) ? static_cast<double>(count) / (4.0 * M_PI * std::pow(dist, 2.0))  : NA_REAL;
  if (Surface_variation) features["Surface_variation"] = (Surface_variation) ? lambda3 / pca.val.sum() : NA_REAL;
  if (Verticality) features["Verticality"] = (Verticality) ? 1.0 - abs(pca.v2(2)) : NA_REAL;

  return features;

}


// [[Rcpp::export]]
Rcpp::DataFrame geometric_features_batch(Rcpp::NumericMatrix points,
                                         Rcpp::NumericVector x_all,
                                         Rcpp::NumericVector y_all,
                                         Rcpp::NumericVector z_all,
                                         double dist,
                                         bool First_eigenvalue = true,
                                         bool Second_eigenvalue = true,
                                         bool Third_eigenvalue = true,
                                         bool Eigenvalue_sum = true,
                                         bool Normal_x = true,
                                         bool Normal_y = true,
                                         bool Normal_z = true,
                                         bool PCA_1 = true,
                                         bool PCA_2 = true,
                                         bool Anisotropy = true,
                                         bool Eigenentropy = true,
                                         bool Linearity = true,
                                         bool Omnivariance = true,
                                         bool Planarity = true,
                                         bool Sphericity = true,
                                         bool Surface_variation = true,
                                         bool Normal_change_rate = false,
                                         bool Verticality = true,
                                         bool Number_of_points = false,
                                         bool surface_density = false,
                                         bool volume_density = false,
                                         int solver_thresh = 50000) {

  // Validate input
  if (points.ncol() != 4) {
    Rcpp::stop("Input 'points' must be a matrix with 4 columns: point, x, y, z");
  }

  const int n_points = points.nrow();

  // Map the full point cloud vectors
  Eigen::Map<Eigen::VectorXd> x_map(x_all.begin(), x_all.size());
  Eigen::Map<Eigen::VectorXd> y_map(y_all.begin(), y_all.size());
  Eigen::Map<Eigen::VectorXd> z_map(z_all.begin(), z_all.size());

  // Pre-allocate vectors to store results
  std::vector<double> vec_point, vec_x, vec_y, vec_z;
  std::vector<double> vec_first_ev, vec_second_ev, vec_third_ev, vec_ev_sum;
  std::vector<double> vec_normal_x, vec_normal_y, vec_normal_z;
  std::vector<double> vec_pca1, vec_pca2;
  std::vector<double> vec_anisotropy, vec_eigenentropy, vec_linearity;
  std::vector<double> vec_omnivariance, vec_planarity, vec_sphericity;
  std::vector<double> vec_surface_var, vec_normal_change, vec_verticality;
  std::vector<double> vec_n_points, vec_surf_density, vec_vol_density;

  vec_point.reserve(n_points);
  vec_x.reserve(n_points);
  vec_y.reserve(n_points);
  vec_z.reserve(n_points);
  if (First_eigenvalue) vec_first_ev.reserve(n_points);
  if (Second_eigenvalue) vec_second_ev.reserve(n_points);
  if (Third_eigenvalue) vec_third_ev.reserve(n_points);
  if (Eigenvalue_sum) vec_ev_sum.reserve(n_points);
  if (Normal_x) vec_normal_x.reserve(n_points);
  if (Normal_y) vec_normal_y.reserve(n_points);
  if (Normal_z) vec_normal_z.reserve(n_points);
  if (PCA_1) vec_pca1.reserve(n_points);
  if (PCA_2) vec_pca2.reserve(n_points);
  if (Anisotropy) vec_anisotropy.reserve(n_points);
  if (Eigenentropy) vec_eigenentropy.reserve(n_points);
  if (Linearity) vec_linearity.reserve(n_points);
  if (Omnivariance) vec_omnivariance.reserve(n_points);
  if (Planarity) vec_planarity.reserve(n_points);
  if (Sphericity) vec_sphericity.reserve(n_points);
  if (Surface_variation) vec_surface_var.reserve(n_points);
  if (Normal_change_rate) vec_normal_change.reserve(n_points);
  if (Verticality) vec_verticality.reserve(n_points);
  if (Number_of_points) vec_n_points.reserve(n_points);
  if (surface_density) vec_surf_density.reserve(n_points);
  if (volume_density) vec_vol_density.reserve(n_points);

  // Iterate over each point
  for (int i = 0; i < n_points; ++i) {
    double point_id = points(i, 0);
    double x_pto = points(i, 1);
    double y_pto = points(i, 2);
    double z_pto = points(i, 3);

    // Compute features for this point
    auto features = geometric_features_core(x_map, y_map, z_map,
                                            x_pto, y_pto, z_pto,
                                            point_id, dist,
                                            First_eigenvalue, Second_eigenvalue, Third_eigenvalue,
                                            Eigenvalue_sum, Normal_x, Normal_y, Normal_z,
                                            PCA_1, PCA_2, Anisotropy, Eigenentropy,
                                            Linearity, Omnivariance, Planarity, Sphericity,
                                            Surface_variation, Normal_change_rate, Verticality,
                                            Number_of_points, surface_density, volume_density,
                                            solver_thresh);

    // Extract results from the map
    vec_point.push_back(features["point"]);
    vec_x.push_back(x_pto);
    vec_y.push_back(y_pto);
    vec_z.push_back(z_pto);
    if (First_eigenvalue) vec_first_ev.push_back(features["First_eigenvalue"]);
    if (Second_eigenvalue) vec_second_ev.push_back(features["Second_eigenvalue"]);
    if (Third_eigenvalue) vec_third_ev.push_back(features["Third_eigenvalue"]);
    if (Eigenvalue_sum) vec_ev_sum.push_back(features["Eigenvalue_sum"]);
    if (Normal_x) vec_normal_x.push_back(features["Normal_x"]);
    if (Normal_y) vec_normal_y.push_back(features["Normal_y"]);
    if (Normal_z) vec_normal_z.push_back(features["Normal_z"]);
    if (PCA_1) vec_pca1.push_back(features["PCA_1"]);
    if (PCA_2) vec_pca2.push_back(features["PCA_2"]);
    if (Anisotropy) vec_anisotropy.push_back(features["Anisotropy"]);
    if (Eigenentropy) vec_eigenentropy.push_back(features["Eigenentropy"]);
    if (Linearity) vec_linearity.push_back(features["Linearity"]);
    if (Omnivariance) vec_omnivariance.push_back(features["Omnivariance"]);
    if (Planarity) vec_planarity.push_back(features["Planarity"]);
    if (Sphericity) vec_sphericity.push_back(features["Sphericity"]);
    if (Surface_variation) vec_surface_var.push_back(features["Surface_variation"]);
    if (Normal_change_rate) vec_normal_change.push_back(features["Normal_change_rate"]);
    if (Verticality) vec_verticality.push_back(features["Verticality"]);
    if (Number_of_points) vec_n_points.push_back(features["Number_of_points"]);
    if (surface_density) vec_surf_density.push_back(features["surface_density"]);
    if (volume_density) vec_vol_density.push_back(features["volume_density"]);
  }

  // Build the DataFrame
  Rcpp::List result;
  result["point"] = vec_point;
  result["x"] = vec_x;
  result["y"] = vec_y;
  result["z"] = vec_z;
  if (First_eigenvalue) result["First_eigenvalue"] = vec_first_ev;
  if (Second_eigenvalue) result["Second_eigenvalue"] = vec_second_ev;
  if (Third_eigenvalue) result["Third_eigenvalue"] = vec_third_ev;
  if (Eigenvalue_sum) result["Eigenvalue_sum"] = vec_ev_sum;
  if (Normal_x) result["Normal_x"] = vec_normal_x;
  if (Normal_y) result["Normal_y"] = vec_normal_y;
  if (Normal_z) result["Normal_z"] = vec_normal_z;
  if (PCA_1) result["PCA_1"] = vec_pca1;
  if (PCA_2) result["PCA_2"] = vec_pca2;
  if (Anisotropy) result["Anisotropy"] = vec_anisotropy;
  if (Eigenentropy) result["Eigenentropy"] = vec_eigenentropy;
  if (Linearity) result["Linearity"] = vec_linearity;
  if (Omnivariance) result["Omnivariance"] = vec_omnivariance;
  if (Planarity) result["Planarity"] = vec_planarity;
  if (Sphericity) result["Sphericity"] = vec_sphericity;
  if (Surface_variation) result["Surface_variation"] = vec_surface_var;
  if (Normal_change_rate) result["Normal_change_rate"] = vec_normal_change;
  if (Verticality) result["Verticality"] = vec_verticality;
  if (Number_of_points) result["Number_of_points"] = vec_n_points;
  if (surface_density) result["surface_density"] = vec_surf_density;
  if (volume_density) result["volume_density"] = vec_vol_density;

  return Rcpp::DataFrame(result);
}

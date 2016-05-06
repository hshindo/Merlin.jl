#define EIGEN_NO_DEBUG
//#define EIGEN_USE_MKL_ALL

#include "eigen/Eigen/Core"

using namespace Eigen;

extern "C" {
  void eigen2d_f32(float *x, int size1, int size2) {
    return Map<MatrixXf>(x, size1, size2);
  }
}

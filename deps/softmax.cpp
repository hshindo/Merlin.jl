#define EIGEN_NO_DEBUG
#define EIGEN_USE_MKL_ALL

#include <iostream>
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

template<typename T>
void softmax() {

}

extern "C" {
  void softmax_fwd_f32() {
    //softmax_fwd(x, params, y, size_x1, size_x2);
  }
  void softmax_bwd_f32() {
    //softmax_bwd(params, gy, gx, size_x1, size_x2);
  }
  void logsoftmax_fwd_f32() {
    //softmax_fwd(x, params, y, size_x1, size_x2);
  }
  void logsoftmax_bwd_f32() {
    //softmax_bwd(params, gy, gx, size_x1, size_x2);
  }
}

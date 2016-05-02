#define EIGEN_NO_DEBUG
//#define EIGEN_USE_MKL_ALL

#include <iostream>
#include "eigen/Eigen/Core"
#include <vector>

using namespace std;
using namespace Eigen;

template<typename T>
void softmax_fw(T *x, T *y) {
  //Matrix4d A = Map<MatrixXd>(x);
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}

extern "C" {
  void softmax_fw_f32(float a) {
    softmax_fw(a);
  }
  void softmax_bw_f32() {
    //softmax_bwd(params, gy, gx, size_x1, size_x2);
  }
  void logsoftmax_fw_f32() {
    //softmax_fwd(x, params, y, size_x1, size_x2);
  }
  void logsoftmax_bw_f32() {
    //softmax_bwd(params, gy, gx, size_x1, size_x2);
  }
}

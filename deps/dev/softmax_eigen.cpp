#include <iostream>
#include "eigen/Eigen/Core"
#include <vector>

using namespace std;
using namespace Eigen;

template<typename T>
void softmax_fw(T *x, int size1, int size2, T *y) {
  Matrix<T,Dynamic,Dynamic> mx = Map< Matrix<T,Dynamic,Dynamic> >(x, size1, size2);
  //Matrix<T,Dynamic,Dynamic> maxm = mx.colwise().maxCoeff();
  mx.exp().rowwise() -= mx.colwise().maxCoeff();
  //Matrix4d A = Map<MatrixXd>(x);
  //MatrixXd m(2,2);
  //m(0,0) = 3;
  //m(1,0) = 2.5;
  //m(0,1) = -1;
  //m(1,1) = m(1,0) + m(0,1);
  std::cout << mx << std::endl;
}

extern "C" {
  void softmax_fw_f32(float *x, int size1, int size2, float *y) {
    softmax_fw(x, size1, size2, y);
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

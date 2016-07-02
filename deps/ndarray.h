template<typename T>
struct NDArray {
  T *value;
  int *dims;
  int *strides;
};

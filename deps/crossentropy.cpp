#include <arrayfire.h>

using namespace af;

void crossentropy_fwd2() {
  array zeros = constant(0, 3);
  //return zeros;
}

extern "C" {
  void crossentropy_fwd() {
    crossentropy_fwd2();
  }
}

#include <arrayfire.h>
#include <stdio.h>
#include <math.h>
#include <cstdlib>

//using namespace af;

void crossentropy_fwd2() {
  af::array zeros;
  //array zeros = constant(0, 3);
  //return zeros;
}

extern "C" {
  void crossentropy_fwd() {
    crossentropy_fwd2();
  }
}

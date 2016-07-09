#include <math.h>
#include <stdint.h>

// Assumed IEEE754.
inline float exp_approx(float x) {
  float exp_cst1 = float(((1<<8) - 1) * (1<<23));
  float exp_cst2 = 0.f;
  union { int i; float f; } xu, xu2;

  float val = (1<<23) / log(2.f) * x + ((1<<7) - 1) * (1<<23);
  val = val < exp_cst1 ? val : exp_cst1;
  val = val > exp_cst2 ? val : exp_cst2;
  int vali = (int)val;
  xu.i = vali & 0x7F800000; // exponent part
  xu2.i = (vali & 0x7FFFFF) | 0x3F800000; // 111... | coefficient
  float b = xu2.f;
  return
    xu.f * (0.510397365625862338668154f + b *
            (0.310670891004095530771135f + b *
             (0.168143436463395944830000f + b *
              (-2.88093587581985443087955e-3f + b *
               1.3671023382430374383648148e-2f))));
}

inline double exp_approx(double x) {
  double exp_cst1 = double(((1LL<<11) - 1LL) * (1LL<<52));
  double exp_cst2 = 0.;
  union { long long i; double f; } xu, xu2;

  double val = (1LL<<52) / log(2.) * x + ((1LL<<10) - 1LL) * (1LL<<52);
  val = val < exp_cst1 ? val : exp_cst1;
  val = val > exp_cst2 ? val : exp_cst2;
  long long vali = (long long)val;
  xu.i = vali & 0x7FF0000000000000;
  xu2.i = (vali & 0xFFFFFFFFFFFFF) | 0x3FF0000000000000;
  double b = xu2.f;
  return
    xu.f * (0.510397365625862338668154 + b *
            (0.310670891004095530771135 + b *
             (0.168143436463395944830000 + b *
              (-2.88093587581985443087955e-3 + b *
               1.3671023382430374383648148e-2))));
}

inline float log_approx(float x) {
  if (x <= 0.0f) return -(float)INFINITY;
  union { float f; int i; } valu;
  valu.f = x;
  float e = valu.i >> 23;
  valu.i = (valu.i & 0x7FFFFF) | 0x3F800000;
  float f = valu.f;
  // 89.970756366f = 127 * log(2) - constant term of polynomial (-1.94106443489)
  return
    f * (3.529304993f + f * (-2.461222105f +
      f * (1.130626167f + f * (-0.288739945f +
        f * 3.110401639e-2f))))
    + (-89.970756366f + 0.69314718055995f*e);
}

inline double log_approx(double x) {
  if (x <= 0.0) return -(double)INFINITY;
  union { double f; long long i; } valu;
  valu.f = x;
  double e = valu.i >> 52;
  valu.i = (valu.i & 0xFFFFFFFFFFFFF) | 0x3FF0000000000000;
  double f = valu.f;
  // 711.030630148 = 1023 * log(2) - constant term of polynomial
  return
    f * (3.529304993 + f * (-2.461222105 +
      f * (1.130626167 + f * (-0.288739945 +
        f * 3.110401639e-2))))
    + (-711.030630148 + 0.69314718055995*e);
}

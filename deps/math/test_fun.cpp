#define _USE_MATH_DEFINES
#include <cstdio>
#include <random>
#include <ctime>

#include "simd_math_prims.h"
#include "icsilog.h"

using namespace std;

const int N_samp = 1<<10;

float vals_test[N_samp];

float buf[N_samp];

void compare_fun_f(float (*f1)(float), float (*f2)(float), float low, float high,
                   bool abs, bool rel) {
  double sum_err = 0, sum_abs_err = 0, sum_sq_err = 0,
    min_err = +INFINITY, max_err = -INFINITY;
  double sum_rel_err = 0, sum_abs_rel_err = 0, sum_sq_rel_err = 0,
    min_rel_err = +INFINITY, max_rel_err = -INFINITY;
  for(int i = 0; i < N_samp; i++) {
    float x = high*vals_test[i]+low*(1-vals_test[i]);
    float y1 = f1(x), y2 = f2(x);
    double err = (double)y1 - y2;
    sum_err += err;
    sum_abs_err += fabs(err);
    sum_sq_err += err*err;
    if(err < min_err) min_err = err;
    if(err > max_err) max_err = err;

    double rel_err = err/y2;
    sum_rel_err += rel_err;
    sum_abs_rel_err += fabs(rel_err);
    sum_sq_rel_err += rel_err*rel_err;
    if(rel_err < min_rel_err) min_rel_err = rel_err;
    if(rel_err > max_rel_err) max_rel_err = rel_err;
  }
  if(abs) {
    printf("Bias:\t\t\t%le\n", sum_err/N_samp);
    printf("Mean absolute error:\t%le\n", sum_abs_err/N_samp);
    printf("RMS error:\t\t%le\n", sqrt(sum_sq_err/N_samp));
    printf("Min difference:\t\t%le\n", min_err);
    printf("Max difference:\t\t%le\n", max_err);
  }

  if(rel) {
    printf("Relative bias:\t\t\t%le\n", sum_rel_err/N_samp);
    printf("Mean relative error:\t\t%le\n", sum_abs_rel_err/N_samp);
    printf("RMS relative error:\t\t%le\n", sqrt(sum_sq_rel_err/N_samp));
    printf("Min relative difference:\t%le\n", min_rel_err);
    printf("Max relative difference:\t%le\n", max_rel_err);
  }
}

#define compare_fun(f1, f2, low, high, abs, rel) {                      \
    printf("Comparing the behavior of " #f1 " against " #f2 ", in the " \
           "interval [%g, %g]:\n", (float)low, (float)high);            \
    compare_fun_f(f1, f2, low, high, abs, rel);                         \
  }

#define bench_fun(f, cycles) {                                          \
    printf("Benchmarking " #f "...");                                   \
    clock_t begin = clock();                                            \
    for(int i = 0; i < cycles; i++)                                     \
      for(int j = 0; j < N_samp; j++)                                   \
        buf[j] = f(vals_test[j]);                                       \
    clock_t end = clock();                                              \
    double dt = double(end - begin) / CLOCKS_PER_SEC;                   \
    double throughput = cycles*N_samp/dt;                               \
    printf("    %.1fM/s\n", throughput/1e6);                            \
}

int main() {
  mt19937 r;
  uniform_real_distribution<float> d;

  for(int i = 0; i < N_samp; i++)
    vals_test[i] = d(r);

  printf("Sin functions:\n--------------\n");
  compare_fun(sinapprox, sinf, -M_PI, M_PI, true, false);
  printf("\n");
  bench_fun(sinf, 100000);
  bench_fun(sinapprox, 1000000L);

  printf("\n\nCos functions:\n--------------\n");
  compare_fun(cosapprox, cosf, -M_PI, M_PI, true, false);
  printf("\n");
  bench_fun(cosf, 100000);
  bench_fun(cosapprox, 1000000L);

  printf("\n\nLog functions:\n--------------\n");
  compare_fun(logapprox, logf, 1e-10, 10, true, false);
  printf("\n");
  compare_fun(icsi_log, logf, 1e-10, 10, true, false);
  printf("\n");
  bench_fun(logf, 100000);
  bench_fun(icsi_log, 100000L);
  bench_fun(logapprox, 1000000L);

  printf("\n\nExp functions:\n--------------\n");
  compare_fun(expapprox, expf, -10, 10, false, true);
  printf("\n");
  bench_fun(expf, 100000);
  bench_fun(expapprox, 1000000L);

  return 0;
}

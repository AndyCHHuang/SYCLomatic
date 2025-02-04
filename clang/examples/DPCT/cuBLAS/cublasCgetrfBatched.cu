#include "cublas_v2.h"

void test(cublasHandle_t handle, int n, cuComplex *const *a, int lda, int *ipiv,
          int *info, int group_count) {
  // Start
  cublasCgetrfBatched(handle /*cublasHandle_t*/, n /*int*/,
                      a /*cuComplex *const **/, lda /*int*/, ipiv /*int **/,
                      info /*int **/, group_count /*int*/);
  // End
}

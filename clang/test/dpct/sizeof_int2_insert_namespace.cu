// RUN: dpct -out-root %T %s -- -x cuda --cuda-host-only --cuda-path="%cuda-path"
// RUN: FileCheck --input-file %T/sizeof_int2_insert_namespace.dp.cpp --match-full-lines %s

void fun() {
  // CHECK:  cl::sycl::int2 a, b, c, d[2], *e[2];
  int2 a, b, c, d[2], *e[2];
  // CHECK:  int i = sizeof(cl::sycl::int2);
  int i = sizeof(int2);
  // CHECK:  int j = sizeof(int);
  int j = sizeof(int);
  // CHECK:  cl::sycl::int2 k;
  int2 k;
  // CHECK:  int kk = sizeof(k);
  int kk = sizeof(k);
}

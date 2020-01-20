// RUN: cd %T
// RUN: mkdir foo
// RUN: cat %s > %T/foo/use_format_file_with_p2.cu
// RUN: echo "ColumnLimit: 50" > %T/foo/.clang-format
// RUN: echo "[" > %T/foo/compile_commands.json
// RUN: echo "  {" >> %T/foo/compile_commands.json
// RUN: echo "    \"command\": \"nvcc -c -m64 -o use_format_file_with_p2.o use_format_file_with_p2.cu\"," >> %T/foo/compile_commands.json
// RUN: echo "    \"directory\": \"%T/foo\"," >> %T/foo/compile_commands.json
// RUN: echo "    \"file\": \"%T/foo/use_format_file_with_p2.cu\"" >> %T/foo/compile_commands.json
// RUN: echo "  }" >> %T/foo/compile_commands.json
// RUN: echo "]" >> %T/foo/compile_commands.json
// RUN: dpct -p=./foo ./foo/use_format_file_with_p2.cu --out-root=%T/out --cuda-include-path="%cuda-path/include" -- --cuda-host-only
// RUN: FileCheck -strict-whitespace %s --match-full-lines --input-file %T/out/use_format_file_with_p2.dp.cpp
// RUN: rm -rf ./*
#include "cuda.h"

void bar();
#define SIZE 100

size_t size = 1234567 * sizeof(float);
float *h_A = (float *)malloc(size);
float *d_A = NULL;

     //CHECK:void foo1() try {
//CHECK-NEXT:  for(;;)
//CHECK-NEXT:    /*
//CHECK-NEXT:    DPCT1003:0: Migrated api does not return error
//CHECK-NEXT:    code. (*, 0) is inserted. You may need to
//CHECK-NEXT:    rewrite this code.
//CHECK-NEXT:    */
//CHECK-NEXT:    int a =
//CHECK-NEXT:        (dpct::get_default_queue_wait()
//CHECK-NEXT:             .memcpy(d_A, h_A,
//CHECK-NEXT:                     sizeof(double) * SIZE * SIZE)
//CHECK-NEXT:             .wait(),
//CHECK-NEXT:         0);
//CHECK-NEXT:}
//CHECK-NEXT:catch (cl::sycl::exception const &exc) {
//CHECK-NEXT:  std::cerr << exc.what()
//CHECK-NEXT:            << "Exception caught at file:"
//CHECK-NEXT:            << __FILE__ << ", line:" << __LINE__
//CHECK-NEXT:            << std::endl;
//CHECK-NEXT:  std::exit(1);
//CHECK-NEXT:}
void foo1() {
  for(;;)
    int a = cudaMemcpy( d_A, h_A, sizeof(double)*SIZE*SIZE, cudaMemcpyDeviceToHost );
}
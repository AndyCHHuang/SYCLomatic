// UNSUPPORTED: -windows-
// RUN: cd %T
// RUN: mkdir hh_constant_db
// RUN: cd hh_constant_db
// RUN: cat %s > hh_constant_db.cpp
// RUN: cat %S/constant_header.h > constant_header.h
// RUN: echo "[" > compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"c++ hh_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/hh_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/hh_constant_db/hh_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }," >> compile_commands.json
// RUN: echo "    {" >> compile_commands.json
// RUN: echo "        \"command\": \"c++ hh_constant_db.cpp\"," >> compile_commands.json
// RUN: echo "        \"directory\": \"%/T/hh_constant_db\"," >> compile_commands.json
// RUN: echo "        \"file\": \"%/T/hh_constant_db/hh_constant_db.cpp\"" >> compile_commands.json
// RUN: echo "    }" >> compile_commands.json
// RUN: echo "]" >> compile_commands.json
// RUN: dpct -p=. --out-root=./out --cuda-include-path="%cuda-path/include"
// RUN: FileCheck %s --match-full-lines --input-file %T/hh_constant_db/out/constant_header.h
// RUN: cd ..
// RUN: rm -rf ./hh_constant_db

// CHECK: /*
// CHECK-NEXT: DPCT1056:{{[0-9]+}}: Intel(R) DPC++ Compatibility Tool did not detect that variable aaa
// CHECK-NEXT: is used in device code. If this variable is also used in device code, you need
// CHECK-NEXT: to rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: static const float aaa = (float)(1ll << 40);
// CHECK-NEXT: /*
// CHECK-NEXT: DPCT1056:{{[0-9]+}}: Intel(R) DPC++ Compatibility Tool did not detect that variable bbb
// CHECK-NEXT: is used in device code. If this variable is also used in device code, you need
// CHECK-NEXT: to rewrite the code.
// CHECK-NEXT: */
// CHECK-NEXT: static const float bbb = (float)(1ll << 20);

#include "constant_header.h"

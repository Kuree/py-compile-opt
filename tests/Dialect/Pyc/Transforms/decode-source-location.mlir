// RUN: pyc-opt %s --decode-source-location --mlir-print-debuginfo %s --split-input-file | FileCheck %s

pyc.code_obj @test1 {
  pyc.code {
    pyc.resume 0 : i8
    pyc.push_null 0 : i8
    pyc.load_name 0 : i8
    pyc.load_const 0 : i8
    pyc.call 1 : i8
    pyc.cache 0 : i8
    pyc.cache 0 : i8
    pyc.cache 0 : i8
    pyc.pop_top 0 : i8
    pyc.return_const 1 : i8
  }
  pyc.constant "/test/test.py" Filename
  pyc.constant "\F0\03\01\01\01\F1\08\00\01\06\80m\D5\00\14" LineTable
} {pyc.first_line_no = 1 : i32}

// CHECK-LABEL: @test1
// CHECK: pyc.resume 0 : i8 loc(#[[LOC0:.*]])
// CHECK: pyc.push_null 0 : i8 loc(#[[LOC1:.*]])
// CHECK: pyc.load_name 0 : i8 loc(#[[LOC1]])
// CHECK: pyc.load_const 0 : i8 loc(#[[LOC2:.*]])
// CHECK: pyc.call 1 : i8 loc(#[[LOC3:.*]])
// CHECK: pyc.cache 0 : i8 loc(#[[LOC3]])
// CHECK: pyc.cache 0 : i8 loc(#[[LOC3]])
// CHECK: pyc.cache 0 : i8 loc(#[[LOC3]])
// CHECK: pyc.pop_top 0 : i8 loc(#[[LOC3]])
// CHECK: pyc.return_const 1 : i8 loc(#[[LOC3]])

// CHECK: #[[LOC0]] = loc("/test/test.py":0:1)
// CHECK: #[[LOC1]] = loc("/test/test.py":4:1)
// CHECK: #[[LOC2]] = loc("/test/test.py":4:6)
// CHECK: #[[LOC3]] = loc("/test/test.py":4:0)

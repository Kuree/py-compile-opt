// RUN: pyc-opt --decode-byte-code %s | FileCheck %s

pyc.code_obj {
  pyc.constant "\97\00" Code
}

// CHECK: pyc.code_obj
// CHECK-NEXT: pyc.code
// CHECK-NEXT: pyc.resume 0 : i8

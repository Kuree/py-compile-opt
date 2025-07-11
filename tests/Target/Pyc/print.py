# RUN: cpython-pyc %s -o %t
# RUN: pyc-translate --import-pyc %t | FileCheck %s

print("hello world")


# CHECK: pyc.constant {{.*}} Code
# CHECK: pyc.collection tuple Constants
# CHECK: pyc.constant "hello world"

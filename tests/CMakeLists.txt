set(CPYTHON_PYC_DRIVER_PATH ${CMAKE_SOURCE_DIR}/utils/cpython-pyc/cpython-pyc.py)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cmake.cfg.py
)

set (MLIR_TEST_DEPENDS
        FileCheck count not
        pyc-translate
        pyc-opt
)

add_lit_testsuite(check-py "Running the py-compile-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MLIR_TEST_DEPENDS}
)

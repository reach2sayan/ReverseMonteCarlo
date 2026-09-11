# The _core extension module. One translation unit per subsystem, mirroring
# include/RMC/, so a binding lives next to the header group it binds.
pybind11_add_module(_core
        src/module.cpp)

# Linking the targets, not naming their include directories, keeps _core's ABI
# in step with the library's: cxx_std_23, RMC_USE_TBB and the global -march
# (which fixes Eigen's fixed-size alignment, i.e. the layout of every Matrix3d
# crossing this boundary) all arrive through RMC. Never restate them here.
target_link_libraries(_core PRIVATE RMC mcsqs_core)
target_compile_definitions(_core PRIVATE RMC_VERSION_STRING="${PROJECT_VERSION}")

# RMC, seitz and mcsqs_core are static archives whenever RMC_BUILD_PYTHON is
# on, so they are absorbed into _core rather than sitting beside it: one
# module, one set of Boost.LEAF thread-local error slots, nothing to locate at
# import.
set_target_properties(_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_CURRENT_BINARY_DIR}/rmc>")

# The pure-Python half, mirrored next to the freshly built _core so that
# PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR} is an importable package with no wheel
# and no install step -- which is what the CTest registration below relies on.
file(GLOB RMC_PY CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/rmc/*.py"
        "${CMAKE_CURRENT_SOURCE_DIR}/rmc/py.typed")
add_custom_command(TARGET _core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/rmc"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RMC_PY} "${CMAKE_CURRENT_BINARY_DIR}/rmc"
        # The _core stubs are a directory (the module has an mcsqs submodule, so
        # pybind11-stubgen writes a package); tolerated absent before generation.
        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/rmc/_core" "${CMAKE_CURRENT_BINARY_DIR}/rmc/_core"
        COMMENT "Mirroring the pure-Python package next to _core")

# Only _core: it already contains the library. COMPONENT python so the wheel
# build installs the extension without headers or a CMake package config.
install(TARGETS _core
        COMPONENT python
        LIBRARY DESTINATION rmc
        RUNTIME DESTINATION rmc)

# The pytest suite, registered so `ctest` stays the single verb for "run the
# tests". Skipped rather than failed when pytest is absent, so a plain
# -DRMC_BUILD_PYTHON=ON build without the uv dev group still configures.
execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE RMC_NO_PYTEST
        OUTPUT_QUIET ERROR_QUIET)
if (RMC_NO_PYTEST)
    message(STATUS "RMC: pytest not importable, skipping python.pytest")
else ()
    set(RMC_PYTEST_ENV
            "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
            "PYTHONDONTWRITEBYTECODE=1"
            "RMC_PACKAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}/rmc")

    # LD_PRELOAD, for two independent reasons, both about the interpreter being
    # something we load into rather than something we built:
    # 1. libstdc++. We compile with GCC 15, but a conda or system interpreter
    #    may load an older libstdc++ first (numpy pulls it in), and then _core
    #    cannot resolve the GLIBCXX_3.4.3x symbols it was built against.
    #    Preloading the compiler's own copy settles the order; a no-op when the
    #    interpreter's is already new enough.
    # 2. The ASan runtime under ENABLE_SANITIZERS, which must come FIRST.
    execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so.6
            OUTPUT_VARIABLE RMC_LIBSTDCXX
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(RMC_PRELOAD "${RMC_LIBSTDCXX}")
    if (ENABLE_SANITIZERS)
        execute_process(
                COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
                OUTPUT_VARIABLE RMC_LIBASAN
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(RMC_PRELOAD "${RMC_LIBASAN}:${RMC_PRELOAD}")
        # CPython and pybind11 leave allocations live at exit by design.
        list(APPEND RMC_PYTEST_ENV "ASAN_OPTIONS=detect_leaks=0")
    endif ()
    list(APPEND RMC_PYTEST_ENV "LD_PRELOAD=${RMC_PRELOAD}")

    add_test(NAME python.pytest
            COMMAND ${Python_EXECUTABLE} -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    set_tests_properties(python.pytest PROPERTIES ENVIRONMENT "${RMC_PYTEST_ENV}")
endif ()

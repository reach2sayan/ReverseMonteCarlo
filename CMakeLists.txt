cmake_minimum_required(VERSION 3.28)
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()

project(ReverseMonteCarlo VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(RMC_BUILD_TESTS "Build Catch2 test suite" ON)
option(ENABLE_NATIVE_ARCH "Optimise for the current machine" ON)
option(ENABLE_MKL "Use Intel MKL backend for Eigen" ON)

if (NOT BOOST_ROOT AND EXISTS "/opt/boost")
    set(BOOST_ROOT "/opt/boost" CACHE PATH "Boost root directory" FORCE)
endif ()
if (BOOST_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")
endif ()

set(RMC_KNOWN_COMPILER OFF)
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(RMC_KNOWN_COMPILER ON)
    set(RMC_COMPILER_MSVC ON)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(RMC_KNOWN_COMPILER ON)
    set(RMC_COMPILER_GCC ON)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(RMC_KNOWN_COMPILER ON)
    set(RMC_COMPILER_CLANG ON)
else ()
    message(STATUS "Compiler ${CMAKE_CXX_COMPILER_ID}: using portable C++23 defaults only")
endif ()

include(CheckCXXCompilerFlag)
if (RMC_COMPILER_MSVC)
    check_cxx_compiler_flag(/arch:AVX2 HAVE_AVX2)
    if (HAVE_AVX2)
        add_compile_options(/arch:AVX2)
        message(STATUS "AVX2: enabled (/arch:AVX2)")
    endif ()
else ()
    check_cxx_compiler_flag(-mavx2 HAVE_AVX2)
    if (HAVE_AVX2)
        if (ENABLE_NATIVE_ARCH)
            add_compile_options(-march=native)
            message(STATUS "AVX2: enabled (-march=native)")
        else ()
            add_compile_options(-march=x86-64-v3)
            message(STATUS "AVX2: enabled (-march=x86-64-v3)")
        endif ()
    else ()
        message(STATUS "AVX2: not supported by this compiler")
    endif ()
endif ()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost 1.83 REQUIRED CONFIG COMPONENTS
        log
        log_setup
        program_options
        serialization
        container
        context
)

add_library(eigen_backend INTERFACE)
if (ENABLE_MKL)
    find_package(MKL CONFIG QUIET)
    if (MKL_FOUND)
        message(STATUS "MKL: found (oneAPI) — enabling Eigen MKL backend")
        target_compile_definitions(eigen_backend INTERFACE EIGEN_USE_MKL_ALL)
        target_link_libraries(eigen_backend INTERFACE MKL::MKL)
    else ()
        find_path(MKL_INCLUDE_DIR mkl.h PATHS /usr/include/mkl /usr/include PATH_SUFFIXES mkl)
        find_library(MKL_RT_LIB mkl_rt)
        if (MKL_INCLUDE_DIR AND MKL_RT_LIB)
            message(STATUS "MKL: found (legacy) — enabling Eigen MKL backend")
            if (NOT TARGET MKL::MKL)
                add_library(MKL::MKL IMPORTED INTERFACE)
                set_target_properties(MKL::MKL PROPERTIES
                        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
                        INTERFACE_LINK_LIBRARIES "${MKL_RT_LIB}")
            endif ()
            target_compile_definitions(eigen_backend INTERFACE EIGEN_USE_MKL_ALL)
            target_link_libraries(eigen_backend INTERFACE MKL::MKL)
        else ()
            message(STATUS "MKL: not found — using default Eigen backend")
            target_compile_definitions(eigen_backend INTERFACE EIGEN_NO_MKL)
        endif ()
    endif ()
else ()
    message(STATUS "MKL: disabled by user")
    target_compile_definitions(eigen_backend INTERFACE EIGEN_NO_MKL)
endif ()

if (RMC_COMPILER_CLANG)
    foreach (_root /opt/rocm/lib/llvm /opt/rocm-7.0.2/lib/llvm /usr/lib/llvm-18 /usr/lib/llvm-17 /usr/local)
        if (EXISTS "${_root}/lib/libomp.so")
            set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "" FORCE)
            set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
            set(OpenMP_omp_LIBRARY "${_root}/lib/libomp.so" CACHE FILEPATH "" FORCE)
            set(_omp_include_dir "${_root}/include")
            break()
        endif ()
    endforeach ()
endif ()
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: ${OpenMP_CXX_VERSION} found")
else ()
    message(STATUS "OpenMP: not found — Eigen will run single-threaded")
endif ()

find_package(Threads REQUIRED)

add_library(RMC SHARED
        src/core/Structure.cpp
        src/constraints/PairConstraintBase.cpp
        src/constraints/StructureFactorConstraint.cpp
        src/io/PdbReader.cpp
        src/io/DataReader.cpp
        src/io/Checkpoint.cpp
        src/Engine.cpp
)
target_include_directories(RMC PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(RMC PUBLIC
        Eigen3::Eigen
        eigen_backend
        Boost::headers
        Boost::log
        Boost::log_setup
        Boost::serialization
        Boost::container
        Boost::context
        Threads::Threads
)

if (OpenMP_CXX_FOUND)
    target_link_libraries(RMC PUBLIC OpenMP::OpenMP_CXX)
    if (DEFINED _omp_include_dir AND EXISTS "${_omp_include_dir}/omp.h")
        target_include_directories(RMC PUBLIC "${_omp_include_dir}")
    endif ()
endif ()

if (RMC_COMPILER_MSVC)
    target_compile_options(RMC PRIVATE /W4 /permissive- /Zc:__cplusplus /external:W0 /external:anglebrackets)
elseif (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG)
    target_compile_options(RMC PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()

add_executable(RMC_run src/main.cpp)
target_link_libraries(RMC_run PRIVATE RMC Boost::program_options)

if (RMC_COMPILER_MSVC)
    target_compile_options(RMC_run PRIVATE /W4 /permissive-)
elseif (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG)
    target_compile_options(RMC_run PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()
# BOOST_LEAF_AUTO expands to a GNU statement-expression; silence on Clang.
if (RMC_COMPILER_CLANG)
    target_compile_options(RMC_run PRIVATE -Wno-gnu-statement-expression-from-macro-expansion)
endif ()

option(RMC_BUILD_MCSQS "Build mcsqs_rmc SQS extension" ON)
if (RMC_BUILD_MCSQS)
    add_subdirectory(extensions/mcsqs)
endif ()

include(CTest)
if (BUILD_TESTING AND RMC_BUILD_TESTS)
    find_package(Catch2 3 REQUIRED)
    add_executable(RMC_tests
            tests/test_boundary_conditions.cpp
            tests/test_generators.cpp
            tests/test_selectors.cpp
            tests/test_constraints.cpp
            tests/test_engine.cpp
            tests/test_examples.cpp
            tests/bench_scaling.cpp
    )
    target_link_libraries(RMC_tests PRIVATE RMC Catch2::Catch2WithMain)
    if (RMC_COMPILER_MSVC)
        target_compile_options(RMC_tests PRIVATE
                /W4 /permissive- /Zc:__cplusplus /external:W0 /external:anglebrackets)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
        foreach (tgt Catch2 Catch2WithMain Catch2::Catch2 Catch2::Catch2WithMain)
            if (TARGET ${tgt})
                target_compile_options(${tgt} PRIVATE /W0)
                set_target_properties(${tgt} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
            endif ()
        endforeach ()
    elseif (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG)
        target_compile_options(RMC_tests PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
    endif ()
    include(Catch)
    catch_discover_tests(RMC_tests DISCOVERY_MODE PRE_TEST)
endif ()

option(ENABLE_SANITIZERS "Enable AddressSanitizer + UBSan on all targets" OFF)
if (ENABLE_SANITIZERS AND (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG))
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "Sanitizers: ASan + UBSan enabled")
endif ()

cmake_minimum_required(VERSION 3.28)
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()

project(ReverseMonteCarlo VERSION 1.0.0 LANGUAGES CXX)

option(ENABLE_LTO "Enable link-time optimization" ON)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

if (ENABLE_LTO AND ipo_supported)
    message(STATUS "LTO: enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(STATUS "LTO: not supported: ${ipo_error}")
endif ()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_CCACHE "Use ccache to speed up rebuilds when available" ON)
if (ENABLE_CCACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_PROGRAM ccache)
    if (CCACHE_PROGRAM)
        message(STATUS "ccache: enabled (${CCACHE_PROGRAM})")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    else ()
        message(STATUS "ccache: not found")
    endif ()
endif ()

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

if (RMC_COMPILER_CLANG)
    find_program(LLD_LINKER ld.lld lld)
    if (LLD_LINKER)
        add_link_options(-fuse-ld=lld)
        message(STATUS "LLD: enabled")
    else ()
        message(STATUS "LLD: not found")
    endif ()
endif ()

# Boost components pull in private .so dependencies (e.g. libboost_filesystem
# behind Boost::serialization) that we never link directly. The linker default
# DT_RUNPATH is not searched when the loader resolves a dependency's own indirect
# deps, so emit the older DT_RPATH instead — it is inherited transitively and
# lets the loader find Boost's private .so deps in ${BOOST_ROOT}/lib at runtime.
if (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG)
    add_link_options(-Wl,--disable-new-dtags)
endif ()

# ── Sanitizer flags ──────────────────────────────────────────────────────────
# These MUST precede the first add_library/add_executable: add_compile_options /
# add_link_options only apply to targets created *after* the call, so placing
# them after the targets (as before) silently no-ops them.
option(ENABLE_SANITIZERS "Enable AddressSanitizer + UBSan on all targets" OFF)
if (ENABLE_SANITIZERS AND (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG))
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "Sanitizers: ASan + UBSan enabled")
endif ()

# Eigen 5 is always fetched from source (never the system package) so every
# platform builds against the exact same pinned version. It is header-only, so
# we skip Eigen's CMake/test machinery: fetch only (no add_subdirectory) and
# expose the headers as a plain interface target under the canonical
# Eigen3::Eigen name. FetchContent caches the download under the build tree
# (override with -DFETCHCONTENT_SOURCE_DIR_EIGEN to reuse a local checkout).
include(FetchContent)
# SOURCE_SUBDIR points at a directory with no CMakeLists, so MakeAvailable
# downloads the source but does NOT add_subdirectory it — we want only the
# headers, not Eigen's test/doc/install targets.
FetchContent_Declare(eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.0
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR _headers_only_no_cmake)
FetchContent_MakeAvailable(eigen)
add_library(eigen_fetched INTERFACE)
target_include_directories(eigen_fetched SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
add_library(Eigen3::Eigen ALIAS eigen_fetched)
message(STATUS "Eigen: fetched 5.0.0 source at ${eigen_SOURCE_DIR}")
# 1.87 is the true floor: src/io/*.cpp use Boost.Parser (added in 1.87) and
# extensions/mcsqs uses Boost.Process v2's compiled library (added in 1.86).
find_package(Boost 1.87 REQUIRED CONFIG COMPONENTS
        program_options
        serialization
        container
)
find_package(spdlog REQUIRED)

add_library(eigen_backend INTERFACE)
if (ENABLE_MKL)
    # Eigen's LAPACKE backend supports ONLY the LP64 (32-bit int) MKL interface.
    # oneAPI defaults MKL::MKL to ILP64, which makes lapack_int 64-bit and breaks
    # Eigen's QR/permutation bindings (lapack_int* = int* in
    # ColPivHouseholderQR_LAPACKE.h). Force LP64 before find_package.
    set(MKL_INTERFACE lp64 CACHE STRING "MKL integer interface (Eigen needs lp64)")
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

find_package(Threads REQUIRED)

add_library(RMC SHARED
        src/core/Structure.cpp
        src/core/BoundaryConditions.cpp
        src/core/NeighborGrid.cpp
        src/core/Parallel.cpp
        src/constraints/PairConstraintBase.cpp
        src/constraints/StructureFactorConstraint.cpp
        src/constraints/ReducedStructureFactorConstraint.cpp
        src/constraints/ConstraintCollection.cpp
        src/constraints/CoordinationConstraint.cpp
        src/constraints/ClusterCorrelationConstraint.cpp
        src/constraints/AngleConstraint.cpp
        src/constraints/AngularDistributionConstraint.cpp
        src/constraints/BondConstraint.cpp
        src/constraints/DihedralAngleConstraint.cpp
        src/constraints/ImproperAngleConstraint.cpp
        src/callbacks/Chi2CollectorCallback.cpp
        src/callbacks/PDBSnapshotCallback.cpp
        src/callbacks/HistogramCallback.cpp
        src/selectors/DirectionalOrderSelector.cpp
        src/io/PdbReader.cpp
        src/io/DataReader.cpp
        src/io/LammpsReader.cpp
        src/io/VaspReader.cpp
        src/io/StructFormat.cpp
        src/io/Checkpoint.cpp
        src/analysis/RadialDistribution.cpp
        src/analysis/AngularDistribution.cpp
        src/core/RandomStructure.cpp
        src/Engine.cpp
        src/Ensemble.cpp
        src/RMCRunner.cpp
)
target_include_directories(RMC PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(RMC PUBLIC
        Eigen3::Eigen
        eigen_backend
        Boost::headers
        spdlog::spdlog
        Boost::program_options
        Boost::serialization
        Boost::container
        Threads::Threads
)

option(RMC_USE_TBB "Use Intel TBB for parallel kernels (par_unseq) inside a shared task arena" ON)
if (RMC_USE_TBB)
    find_package(TBB REQUIRED)
    target_link_libraries(RMC PUBLIC TBB::tbb)
    target_compile_definitions(RMC PUBLIC RMC_USE_TBB)
    message(STATUS "TBB: enabled — parallel kernels run via par_unseq in a shared arena")
else ()
    message(STATUS "TBB: disabled (RMC_USE_TBB=OFF) — parallel kernels run serially")
endif ()

if (RMC_COMPILER_MSVC)
    target_compile_options(RMC PRIVATE /W4 /permissive- /Zc:__cplusplus /external:W0 /external:anglebrackets)
elseif (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG)
    target_compile_options(RMC PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()
# RMCRunner.cpp (now in the library) uses BOOST_LEAF_AUTO, which expands to a GNU
# statement-expression; silence the resulting Clang diagnostic.
if (RMC_COMPILER_CLANG)
    target_compile_options(RMC PRIVATE -Wno-gnu-statement-expression-from-macro-expansion)
endif ()

# RMC_run is a thin CLI front-end: the driver (RMCRunner) and the shared
# RMCConfig/build_engine API live in the RMC library.
add_executable(RMC_run src/main.cpp)
target_link_libraries(RMC_run PRIVATE RMC)

if (RMC_COMPILER_MSVC)
    target_compile_options(RMC_run PRIVATE /W4 /permissive-)
elseif (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG)
    target_compile_options(RMC_run PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()

option(RMC_BUILD_MCSQS "Build mcsqs_rmc SQS extension" ON)
if (RMC_BUILD_MCSQS)
    # Resolves corrdump (external tool, run via boost::process — never linked,
    # ATAT not vendored). Provider is selected with RMC_ATAT_PROVIDER; see
    # cmake/Atat.cmake. Default SYSTEM uses an installed corrdump / --corrdump.
    include(${CMAKE_SOURCE_DIR}/cmake/Atat.cmake)
    add_subdirectory(extensions/mcsqs)
endif ()

option(RMC_BUILD_EXAMPLES "Build the bundled C++ examples" ON)
if (RMC_BUILD_EXAMPLES)
    add_subdirectory(examples)
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
            tests/test_species_swap.cpp
            tests/test_sampler.cpp
            tests/test_atat_pipeline.cpp
            tests/test_io_readers.cpp
            tests/test_radial_distribution.cpp
            tests/test_angular_distribution.cpp
            tests/test_callbacks.cpp
            tests/bench_scaling.cpp
            tests/bench_thf.cpp
            # ATAT pipeline sources under test (corrdump-free: no boost::process).
            extensions/mcsqs/AtatFormats.cpp
            extensions/mcsqs/ClusterEnumerator.cpp
    )
    target_include_directories(RMC_tests PRIVATE extensions/mcsqs)
    target_compile_definitions(RMC_tests PRIVATE
            THF_BENCHMARK_DATA_DIR="${CMAKE_SOURCE_DIR}/examples/benchmark/data")
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
    # POST_BUILD (the default) enumerates each TEST_CASE right after the binary
    # links, so IDEs (CLion) populate the test tree without having to run first.
    # PRE_TEST defers discovery to test-run time, which leaves CLion's browser empty.
    catch_discover_tests(RMC_tests DISCOVERY_MODE POST_BUILD)
endif ()


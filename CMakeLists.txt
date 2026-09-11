cmake_minimum_required(VERSION 3.28)
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()

project(ReverseMonteCarlo VERSION 1.0.2 LANGUAGES CXX)

option(ENABLE_LTO "Enable link-time optimization" ON)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)

# Never on MSVC. CMAKE_INTERPROCEDURAL_OPTIMIZATION is set here, before seitz
# is fetched, so it initialises INTERPROCEDURAL_OPTIMIZATION on the seitz target
# as well -- putting /GL on its 70k-line constexpr table TUs, which seitz itself
# declines on MSVC for the link-time cost.
if (ENABLE_LTO AND ipo_supported AND NOT MSVC)
    message(STATUS "LTO: enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif (ENABLE_LTO AND MSVC)
    message(STATUS "LTO: disabled (MSVC — seitz's tables would compile with /GL)")
else ()
    message(STATUS "LTO: not supported: ${ipo_error}")
endif ()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
# RMC uses no C++20 modules. With C++23 CMake would still scan every TU for
# them (-fdeps-format=p1689r5 -fmodules-ts), which costs a scan step per file
# and breaks under the manylinux image's ccache 3.x: it handles -MD/-MF itself,
# leaving GCC the scan flags alone ("to generate dependencies you must specify
# either -M or -MM"). seitz turns it off for the same reason.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Not on MSVC: ccache's cl.exe support is narrow. CI passes
# -DCMAKE_CXX_COMPILER_LAUNCHER=sccache there, which this block leaves alone.
option(ENABLE_CCACHE "Use ccache to speed up rebuilds when available" ON)
if (ENABLE_CCACHE AND NOT MSVC AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
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
option(RMC_BUILD_PYTHON "Build the pybind11 extension module (rmc._core)" OFF)
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

include(FetchContent)

# The Python extension. Its find_package(Python) runs before anything else asks:
# the first call caches the interpreter, and seitz's configure step (and
# pybind11's) would otherwise pin whatever python is first on PATH -- conda's
# 3.10 on a machine that has one. The python preset points it at .venv.
if (RMC_BUILD_PYTHON)
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
    # seitz as a static archive too, so _core absorbs it like RMC.
    set(SEITZ_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
    # Everything linked into _core.so (RMC, mcsqs_core, a fetched spdlog) has
    # to be relocatable.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif ()

# Eigen3 and Boost are declared before seitz, so its own FetchContent_Declare
# calls for them are ignored (first declaration wins) and it links these.
# Eigen3::Eigen is IMPORTED so seitz's install(EXPORT) accepts it.
FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 5.0.0
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR _headers_only_no_cmake)
FetchContent_MakeAvailable(Eigen3)
add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${eigen3_SOURCE_DIR}")
message(STATUS "Eigen: fetched 5.0.0 source at ${eigen3_SOURCE_DIR}")

# find_package first (locally /opt/boost, in CI ~/boost), and the CMake-ready
# release archive only when that fails -- which is how Windows gets a Boost with
# no b2 script to maintain. Deliberately no REQUIRED in FIND_PACKAGE_ARGS: it
# makes a failed find fatal, so CMake would error out before reaching the URL.
# The superproject builds static, so Boost is absorbed into RMC_run.exe and
# _core.pyd and oneTBB is the only DLL left to ship.
# BOOST_INCLUDE_LIBRARIES is the union of the modules this tree includes and the
# ones seitz links; the superproject resolves each module's own dependencies.
# seitz install(EXPORT)s a target whose usage requirements name these Boost
# targets, and CMake refuses to export a target that depends on one which is in
# no export set. Boost's own install rules put them in one, so they must not be
# skipped. seitz sets this too, but its Dependencies.cmake runs after the
# MakeAvailable below -- by then Boost has already been configured, and the
# variable no longer has any effect. It is inert when find_package wins.
set(BOOST_SKIP_INSTALL_RULES OFF CACHE BOOL "" FORCE)
set(BOOST_INCLUDE_LIBRARIES
        program_options serialization container graph nowide
        leaf flyweight parser algorithm range
        assert describe dynamic_bitset math mp11 mpl random
        stl_interfaces type_erasure)
FetchContent_Declare(Boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.89.0/boost-1.89.0-cmake.tar.xz
        DOWNLOAD_EXTRACT_TIMESTAMP ON
        SYSTEM
        FIND_PACKAGE_ARGS 1.88 CONFIG COMPONENTS
        program_options serialization container graph nowide)
FetchContent_MakeAvailable(Boost)
# A b2-installed Boost ships CMake configs only for compiled libraries; the
# header-only ones seitz names get interface targets over Boost::headers. When
# the superproject above supplied them instead, these are already real targets.
foreach (lib leaf flyweight parser algorithm range)
    if (NOT TARGET Boost::${lib})
        add_library(Boost::${lib} INTERFACE IMPORTED GLOBAL)
        target_link_libraries(Boost::${lib} INTERFACE Boost::headers)
    endif ()
endforeach ()
# The system spdlog when there is one (locally, in CI); otherwise a static one,
# which is what the manylinux wheel container gets.
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "")
FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.15.3
        GIT_SHALLOW TRUE
        FIND_PACKAGE_ARGS 1.12)
FetchContent_MakeAvailable(spdlog)

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

# seitz (crystallography), linked against the Eigen3 and Boost declared above.
set(SEITZ_EXTERNAL_EIGEN_AND_BOOST ON)
set(SEITZ_BUILD_TESTS OFF)
set(SEITZ_BUILD_DEMO OFF)
FetchContent_Declare(Seitz
        GIT_REPOSITORY https://github.com/reach2sayan/Seitz.git
        GIT_TAG b2c96d194a469daec93b4160cdd47861f8fb8c91 # v0.1.4 + SEITZ_STATIC_LIBRARY
        SYSTEM)
FetchContent_MakeAvailable(Seitz)

# pybind11 3.x (py::native_enum), pinned like seitz pins it. FETCHCONTENT_SOURCE_
# DIR_PYBIND11 points it at a local clone for offline builds.
if (RMC_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)
    set(PYBIND11_INSTALL OFF CACHE BOOL "" FORCE)
    set(PYBIND11_TEST OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v3.0.4
            GIT_SHALLOW TRUE
            SYSTEM)
    FetchContent_MakeAvailable(pybind11)
endif ()

# STATIC when the Python extension absorbs it: one module, one set of
# Boost.LEAF thread-local error slots, nothing to locate at import time.
# STATIC on Windows for the same reason, and there it is not optional: PE has no
# symbol interposition, LEAF keeps its active-context slots in thread_local
# static members of a class template, and a DLL boundary would give the library
# and its consumer separate slots -- an error raised inside RMC.dll would arrive
# with its payload gone. WINDOWS_EXPORT_ALL_SYMBOLS cannot fix that (the
# consumer's own COMDAT definition wins), and a DLL with no dllexport would
# anyway produce no import library to link against. seitz forces STATIC on WIN32
# for the identical reason.
if (RMC_BUILD_PYTHON OR WIN32)
    set(RMC_LIBRARY_TYPE STATIC)
else ()
    set(RMC_LIBRARY_TYPE SHARED)
endif ()
add_library(RMC ${RMC_LIBRARY_TYPE}
        src/core/Structure.cpp
        src/core/NeighborGrid.cpp
        src/core/Parallel.cpp
        src/constraints/PairConstraintBase.cpp
        src/constraints/ConstraintCollection.cpp
        src/constraints/CoordinationConstraint.cpp
        src/constraints/ClusterCorrelationConstraint.cpp
        src/constraints/AngularDistributionConstraint.cpp
        src/callbacks/Chi2CollectorCallback.cpp
        src/callbacks/PDBSnapshotCallback.cpp
        src/callbacks/HistogramCallback.cpp
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
        Boost::nowide
        Threads::Threads
        seitz::seitz
)

option(RMC_USE_TBB "Use Intel TBB for parallel kernels inside a shared task arena" ON)
if (RMC_USE_TBB)
    # find_package first, so a system oneTBB (libtbb-dev, or the wheel prefix
    # tools/wheel_deps.sh builds) is still what Linux uses; the source archive is
    # the Windows fallback. The version matches tools/wheel_deps.sh so both paths
    # agree. It stays a shared library: tbb12.dll is bundled into the wheel and
    # installed next to RMC_run.exe.
    set(TBB_TEST OFF CACHE BOOL "" FORCE)
    set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(TBB_STRICT OFF CACHE BOOL "" FORCE)  # its -Werror/-WX vs. newer compilers
    set(TBBMALLOC_BUILD OFF CACHE BOOL "" FORCE)  # nothing here links tbbmalloc
    FetchContent_Declare(TBB
            URL https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2022.3.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP ON
            SYSTEM
            FIND_PACKAGE_ARGS)
    FetchContent_MakeAvailable(TBB)
    target_link_libraries(RMC PUBLIC TBB::tbb)
    target_compile_definitions(RMC PUBLIC RMC_USE_TBB)
    message(STATUS "TBB: enabled — parallel kernels run via parallel_for_each in a shared arena")
else ()
    message(STATUS "TBB: disabled (RMC_USE_TBB=OFF) — parallel kernels run serially")
endif ()

if (RMC_COMPILER_MSVC)
    target_compile_options(RMC PRIVATE /W4 /permissive- /Zc:__cplusplus /external:W0 /external:anglebrackets)
elseif (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG)
    target_compile_options(RMC PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()

# PUBLIC, and not negotiable: boost/parser/error_handling.hpp includes
# <Windows.h> under _MSC_VER, whose min/max macros then break every std::min and
# std::max that follows -- in the five TUs that include Boost.Parser (three in
# src/io, two in extensions/mcsqs, i.e. another target) and in any inline header
# function pulled in after them.
if (WIN32)
    target_compile_definitions(RMC PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif ()
if (RMC_COMPILER_MSVC)
    # Inherited from seitz::seitz today; restated so the tree does not depend on
    # another project's flag list. /utf-8 is a hard requirement -- RMC's own
    # sources carry narrow string literals with Å, χ² and box-drawing characters.
    # _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR guards against an older msvcp140.dll
    # loading ahead of the bundled one (the VS 17.10 std::mutex ABI trap).
    target_compile_options(RMC PUBLIC /utf-8 /bigobj /Zc:__cplusplus /wd4068)
    target_compile_definitions(RMC PUBLIC _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif ()

# Windows resolves a DLL beside the executable that needs it, and has no RPATH.
# This is what makes ctest work at all: catch_discover_tests(... POST_BUILD) runs
# RMC_tests.exe during the build, so tbb12.dll has to be there by then.
function(rmc_windows_runtime_dlls tgt)
    if (WIN32 AND TARGET ${tgt})
        add_custom_command(TARGET ${tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${tgt}> $<TARGET_FILE_DIR:${tgt}>
                COMMAND_EXPAND_LISTS)
    endif ()
endfunction()

# RMCRunner.cpp (now in the library) uses BOOST_LEAF_AUTO, which expands to a GNU
# statement-expression; silence the resulting Clang diagnostic.
if (RMC_COMPILER_CLANG)
    target_compile_options(RMC PRIVATE -Wno-gnu-statement-expression-from-macro-expansion)
endif ()

add_executable(RMC_run src/main.cpp)
target_link_libraries(RMC_run PRIVATE RMC)
rmc_windows_runtime_dlls(RMC_run)

if (RMC_COMPILER_MSVC)
    target_compile_options(RMC_run PRIVATE /W4 /permissive-)
elseif (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG)
    target_compile_options(RMC_run PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()

# Release tarballs are relocatable: bin/RMC_run finds lib/ relative to itself,
# and lib/ carries every non-glibc dependency (seitz, Boost, TBB, spdlog, fmt,
# and GCC 15's libstdc++, which stock distros do not ship yet). The RPATH is a
# DT_RPATH (--disable-new-dtags above), so it also covers indirect deps and wins
# over the system libstdc++.
# Not for the wheel, which installs COMPONENT python only.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT RMC_BUILD_PYTHON)
    include(GNUInstallDirs)
    # seitz is listed explicitly: a target built in this tree (FetchContent) is
    # never collected as a runtime dependency unless it is itself installed.
    set_target_properties(RMC seitz PROPERTIES INSTALL_RPATH "$ORIGIN")
    set_target_properties(RMC_run PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    install(TARGETS RMC RMC_run seitz COMPONENT runtime RUNTIME_DEPENDENCY_SET rmc_runtime)
    install(RUNTIME_DEPENDENCY_SET rmc_runtime COMPONENT runtime
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            PRE_EXCLUDE_REGEXES "^(ld-linux-x86-64|libc|libm|libdl|libpthread|librt)\\.so")
elseif (WIN32 AND NOT RMC_BUILD_PYTHON)
    # The zip is bin/ and nothing else: DLLs live beside the executable, because
    # Windows has no RPATH. RMC and seitz are absent on purpose -- both are static
    # archives here, so there is nothing of them left to install. DIRECTORIES is
    # needed because TBB::tbb is built in this tree (or imported from a prefix)
    # and its DLL is on no default search path. InstallRequiredSystemLibraries
    # brings msvcp140/vcruntime140: the zip has no Python interpreter to inherit a
    # redistributable from, and the System32 copies are excluded below.
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION bin)
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_COMPONENT runtime)
    set(CMAKE_INSTALL_UCRT_LIBRARIES TRUE)
    include(InstallRequiredSystemLibraries)
    # RUNTIME_DEPENDENCY_SET before the RUNTIME artifact group: once an
    # artifact group is open, later keywords are parsed as part of it.
    install(TARGETS RMC_run
            RUNTIME_DEPENDENCY_SET rmc_runtime
            RUNTIME DESTINATION bin COMPONENT runtime)
    install(RUNTIME_DEPENDENCY_SET rmc_runtime COMPONENT runtime
            DESTINATION bin
            DIRECTORIES $<TARGET_FILE_DIR:TBB::tbb>
            PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
            POST_EXCLUDE_REGEXES ".*[/\\\\][Ss]ystem32[/\\\\].*")
endif ()

option(RMC_BUILD_MCSQS "Build mcsqs_rmc SQS extension" ON)
if (RMC_BUILD_MCSQS)
    add_subdirectory(extensions/mcsqs)
endif ()

option(RMC_BUILD_EXAMPLES "Build the bundled C++ examples" ON)
if (RMC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

include(CTest)
if (BUILD_TESTING AND RMC_BUILD_TESTS)
    # Fetched only when the system has none -- apt's catch2 on Linux CI, the
    # archive on Windows, where there is no package manager in the build.
    FetchContent_Declare(Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG v3.8.1
            GIT_SHALLOW TRUE
            SYSTEM
            FIND_PACKAGE_ARGS 3)
    FetchContent_MakeAvailable(Catch2)
    if (catch2_SOURCE_DIR)  # fetched: include(Catch) comes from extras/
        list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
    endif ()
    add_executable(RMC_tests
            tests/test_boundary_conditions.cpp
            tests/test_generators.cpp
            tests/test_selectors.cpp
            tests/test_constraints.cpp
            tests/test_engine.cpp
            tests/test_examples.cpp
            tests/test_species_swap.cpp
            tests/test_sampler.cpp
            tests/test_io_readers.cpp
            tests/test_radial_distribution.cpp
            tests/test_angular_distribution.cpp
            tests/test_callbacks.cpp
            tests/bench_scaling.cpp
            tests/bench_thf.cpp
            tests/bench_adf.cpp
    )
    if (TARGET mcsqs_core) # the SQS lattice/cluster pipeline under test
        target_sources(RMC_tests PRIVATE tests/test_atat_pipeline.cpp)
        target_link_libraries(RMC_tests PRIVATE mcsqs_core)
    endif ()
    target_compile_definitions(RMC_tests PRIVATE
            THF_BENCHMARK_DATA_DIR="${CMAKE_SOURCE_DIR}/examples/benchmark/data")
    target_link_libraries(RMC_tests PRIVATE RMC Catch2::Catch2WithMain)
    if (RMC_COMPILER_MSVC)
        target_compile_options(RMC_tests PRIVATE
                /W4 /permissive- /Zc:__cplusplus /external:W0 /external:anglebrackets)
        foreach (tgt Catch2 Catch2WithMain)
            if (TARGET ${tgt})
                # Skip alias/imported targets: target_compile_options can only set
                # INTERFACE properties on those, so only the real buildable targets
                # (the non-namespaced FetchContent ones) get the warning overrides.
                get_target_property(_tgt_aliased ${tgt} ALIASED_TARGET)
                get_target_property(_tgt_imported ${tgt} IMPORTED)
                if (NOT _tgt_aliased AND NOT _tgt_imported)
                    target_compile_options(${tgt} PRIVATE /W0)
                    set_target_properties(${tgt} PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
                endif ()
            endif ()
        endforeach ()
    elseif (RMC_COMPILER_GCC OR RMC_COMPILER_CLANG)
        target_compile_options(RMC_tests PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
    endif ()
    rmc_windows_runtime_dlls(RMC_tests)
    include(Catch)
    # Catch2 exits 4 when every selected test SKIP()s (the corrdump oracle
    # without RMC_CORRDUMP); report that as skipped, not failed.
    catch_discover_tests(RMC_tests DISCOVERY_MODE POST_BUILD
            PROPERTIES SKIP_RETURN_CODE 4)
endif ()

# Last, after include(CTest), so the pytest suite registers as a ctest.
if (RMC_BUILD_PYTHON)
    if (NOT TARGET mcsqs_core)
        message(FATAL_ERROR "RMC_BUILD_PYTHON needs RMC_BUILD_MCSQS=ON (rmc.mcsqs binds mcsqs_core)")
    endif ()
    add_subdirectory(python)
endif ()

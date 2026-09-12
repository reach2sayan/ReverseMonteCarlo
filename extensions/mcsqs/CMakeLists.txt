cmake_minimum_required(VERSION 3.28)

# Dual-mode CMakeLists for mcsqs_rmc.
#
# In-tree (normal):
#   Included automatically from the main RMC project when RMC_BUILD_MCSQS=ON.
#   The RMC target is already defined; nothing extra is needed.
#
# Standalone (building against a separately built libRMC and an installed
# seitz):
#   cmake -S extensions/mcsqs -B build/mcsqs \
#         -DRMC_SOURCE_DIR=/path/to/RMC \
#         -DRMC_BUILD_DIR=/path/to/RMC/build
#   cmake --build build/mcsqs
#
if (NOT TARGET RMC)
    project(mcsqs_rmc VERSION 1.0.0 LANGUAGES CXX)

    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Mirror the Boost root logic from the main repo.
    if (NOT BOOST_ROOT AND EXISTS "/opt/boost")
        set(BOOST_ROOT "/opt/boost" CACHE PATH "Boost root directory" FORCE)
    endif ()
    if (BOOST_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")
    endif ()

    # Locate libRMC from a build tree without requiring a full install.
    if (NOT DEFINED RMC_SOURCE_DIR OR NOT DEFINED RMC_BUILD_DIR)
        message(FATAL_ERROR
                "Standalone build requires:\n"
                "  -DRMC_SOURCE_DIR=<path/to/RMC/source>\n"
                "  -DRMC_BUILD_DIR=<path/to/RMC/build>")
    endif ()

    find_library(RMC_LIB NAMES RMC PATHS "${RMC_BUILD_DIR}" NO_DEFAULT_PATH REQUIRED)

    # Fetch Eigen 5 from source (same as the top-level build — never the system
    # package). Header-only: download without add_subdirectory, expose headers.
    include(FetchContent)
    FetchContent_Declare(eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 5.0.0
            GIT_SHALLOW TRUE
            SOURCE_SUBDIR _headers_only_no_cmake)
    FetchContent_MakeAvailable(eigen)
    add_library(eigen_fetched INTERFACE)
    target_include_directories(eigen_fetched SYSTEM INTERFACE "${eigen_SOURCE_DIR}")
    add_library(Eigen3::Eigen ALIAS eigen_fetched)
    find_package(Boost 1.88 REQUIRED CONFIG COMPONENTS
            serialization container context program_options)
    find_package(spdlog REQUIRED)
    find_package(Threads REQUIRED)
    find_package(Seitz REQUIRED CONFIG)

    add_library(_rmc_imported SHARED IMPORTED)
    set_target_properties(_rmc_imported PROPERTIES
            IMPORTED_LOCATION "${RMC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${RMC_SOURCE_DIR}/include"
            INTERFACE_LINK_LIBRARIES
            "Eigen3::Eigen;spdlog::spdlog;seitz::seitz;\
Boost::serialization;\
Boost::container;Boost::context;\
Threads::Threads")

    set(_rmc_target _rmc_imported)
else ()
    # In-tree: use the already-defined RMC target directly.
    set(_rmc_target RMC)
    find_package(Boost REQUIRED CONFIG COMPONENTS program_options)
endif ()

# Lattice I/O + seitz cluster enumeration, shared by mcsqs_rmc and RMC_tests.
add_library(mcsqs_core STATIC AtatFormats.cpp SeitzClusters.cpp)
target_include_directories(mcsqs_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mcsqs_core PUBLIC ${_rmc_target})

add_executable(mcsqs_rmc mcsqs_rmc.cpp McsqsRunner.cpp)
target_link_libraries(mcsqs_rmc PRIVATE mcsqs_core Boost::program_options)

foreach (tgt mcsqs_core mcsqs_rmc)
    target_compile_features(${tgt} PRIVATE cxx_std_23)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
    endif ()
    if (MSVC)
        target_compile_options(${tgt} PRIVATE /W4 /permissive-)
        # AtatFormats.cpp and McsqsRunner.cpp both include Boost.Parser, which
        # drags in <Windows.h>. In-tree these arrive PUBLIC from RMC; the
        # standalone branch above links an imported RMC that carries no such
        # usage requirements, so set them here too. Harmless when duplicated.
        target_compile_definitions(${tgt} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        target_compile_options(${tgt} PRIVATE /utf-8 /bigobj)
    endif ()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${tgt} PRIVATE -Wno-gnu-statement-expression-from-macro-expansion)
    endif ()
endforeach ()

# In-tree only: the parent defines this. A standalone build links an imported
# RMC and has no such function, so guard rather than hard-error there.
if (COMMAND rmc_windows_runtime_dlls)
    rmc_windows_runtime_dlls(mcsqs_rmc)
endif ()

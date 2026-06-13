cmake_minimum_required(VERSION 3.28)

# Dual-mode CMakeLists for mcsqs_rmc.
#
# In-tree (normal):
#   Included automatically from the main RMC project when RMC_BUILD_MCSQS=ON.
#   The RMC target is already defined; nothing extra is needed.
#
# Standalone (building against a separately built libRMC):
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

    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
    find_package(Boost 1.87 REQUIRED CONFIG COMPONENTS
            serialization
            container context program_options
            process filesystem)
    find_package(spdlog REQUIRED)
    find_package(Threads REQUIRED)

    add_library(_rmc_imported SHARED IMPORTED)
    set_target_properties(_rmc_imported PROPERTIES
            IMPORTED_LOCATION "${RMC_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${RMC_SOURCE_DIR}/include"
            INTERFACE_LINK_LIBRARIES
            "Eigen3::Eigen;spdlog::spdlog;\
Boost::serialization;\
Boost::container;Boost::context;\
Threads::Threads")

    set(_rmc_target _rmc_imported)
else ()
    # In-tree: use the already-defined RMC target directly.
    set(_rmc_target RMC)
    find_package(Boost REQUIRED CONFIG COMPONENTS
            program_options process filesystem)
endif ()

add_executable(mcsqs_rmc
        mcsqs_rmc.cpp
        McsqsRunner.cpp
        AtatFormats.cpp
        ClusterEnumerator.cpp
        AtatRunner.cpp)
target_link_libraries(mcsqs_rmc PRIVATE ${_rmc_target}
        Boost::program_options
        Boost::process
        Boost::filesystem)
target_compile_features(mcsqs_rmc PRIVATE cxx_std_23)

# Wire in the vendored corrdump built by cmake/Atat.cmake (ExternalProject).
if (TARGET atat_ext)
    add_dependencies(mcsqs_rmc atat_ext)
endif ()
if (ATAT_CORRDUMP_PATH)
    target_compile_definitions(mcsqs_rmc PRIVATE RMC_CORRDUMP_PATH="${ATAT_CORRDUMP_PATH}")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mcsqs_rmc PRIVATE -Wall -Wextra -Wpedantic -Wfatal-errors)
endif ()
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(mcsqs_rmc PRIVATE -Wno-gnu-statement-expression-from-macro-expansion)
endif ()

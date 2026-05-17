cmake_minimum_required(VERSION 3.28)
project(fullrmc_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Dependencies ----
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost 1.83 REQUIRED COMPONENTS
    random
    log
    log_setup
    program_options
    serialization
    container
    filesystem
    system
)

# Header-only Boost components (histogram, accumulators, hana, mp11) are
# pulled in via Boost::headers
find_package(Catch2 3 REQUIRED)

# ---- Library ----
add_library(fullrmc STATIC
    src/core/Structure.cpp
    src/constraints/PairDistributionConstraint.cpp
    src/constraints/PairCorrelationConstraint.cpp
    src/constraints/StructureFactorConstraint.cpp
    src/io/PdbReader.cpp
    src/io/DataReader.cpp
    src/io/Checkpoint.cpp
    src/Engine.cpp
)

target_include_directories(fullrmc PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(fullrmc PUBLIC
    Eigen3::Eigen
    Boost::headers
    Boost::random
    Boost::log
    Boost::log_setup
    Boost::serialization
    Boost::container
    Boost::filesystem
    Boost::system
)

target_compile_options(fullrmc PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /permissive->
)

# ---- CLI executable ----
add_executable(fullrmc_run src/main.cpp)
target_link_libraries(fullrmc_run PRIVATE fullrmc Boost::program_options)

# ---- Tests ----
enable_testing()

add_executable(fullrmc_tests
    tests/test_boundary_conditions.cpp
    tests/test_generators.cpp
    tests/test_selectors.cpp
    tests/test_constraints.cpp
    tests/test_engine.cpp
)
target_link_libraries(fullrmc_tests PRIVATE fullrmc Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(fullrmc_tests)

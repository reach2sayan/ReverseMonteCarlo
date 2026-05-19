cmake_minimum_required(VERSION 3.28)

if(NOT TARGET RMC)
    project(orientation LANGUAGES CXX)
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS ON)

    if(NOT BOOST_ROOT AND EXISTS "/opt/boost")
        set(BOOST_ROOT "/opt/boost" CACHE PATH "" FORCE)
    endif()
    if(BOOST_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")
    endif()

    if(NOT DEFINED RMC_SOURCE_DIR OR NOT DEFINED RMC_BUILD_DIR)
        message(FATAL_ERROR "Standalone build requires -DRMC_SOURCE_DIR=... -DRMC_BUILD_DIR=...")
    endif()

    find_library(RMC_LIB NAMES RMC PATHS "${RMC_BUILD_DIR}" NO_DEFAULT_PATH REQUIRED)
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
    find_package(Boost 1.83 REQUIRED CONFIG COMPONENTS
        log log_setup serialization container context)
    find_package(Threads REQUIRED)

    add_library(_rmc_imported SHARED IMPORTED)
    set_target_properties(_rmc_imported PROPERTIES
        IMPORTED_LOCATION "${RMC_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${RMC_SOURCE_DIR}/include"
        INTERFACE_LINK_LIBRARIES
            "Eigen3::Eigen;Boost::log;Boost::log_setup;Boost::serialization;Boost::container;Boost::context;Threads::Threads")
    set(_rmc_target _rmc_imported)
else()
    set(_rmc_target RMC)
endif()

add_executable(orientation main.cpp)
target_link_libraries(orientation PRIVATE ${_rmc_target})
target_compile_features(orientation PRIVATE cxx_std_23)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(orientation PRIVATE -Wall -Wextra -Wpedantic)
endif()

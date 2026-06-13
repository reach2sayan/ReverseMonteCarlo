# corrdump (from ATAT) is an OPTIONAL external tool that mcsqs_rmc drives at
# runtime via boost::process. It is NEVER linked into RMC, and ATAT source is
# NOT vendored or redistributed by this project. How corrdump is provided is
# selected with RMC_ATAT_PROVIDER:
#
#   SYSTEM (default) — Use a corrdump already installed on the system (found on
#                      PATH, or overridden at runtime with `--corrdump <path>`).
#                      Nothing is downloaded or built; RMC ships no ATAT code.
#                      Install ATAT separately: https://axelvdw.github.io/atat/
#
#   FETCH            — Download an ATAT checkout at build time into the build
#                      tree (NOT committed to this repository) and build only
#                      `corrdump`. Set RMC_ATAT_GIT_REPOSITORY / RMC_ATAT_GIT_TAG
#                      to choose the source.
#
#   SOURCE           — Build `corrdump` from an existing ATAT checkout you point
#                      at with -DRMC_ATAT_SOURCE_DIR=<path>.
#
# In every mode, ATAT remains a separate executable run across a process
# boundary — it is not combined with, or distributed as part of, RMC.

set(RMC_ATAT_PROVIDER "SYSTEM" CACHE STRING
        "How corrdump is provided: SYSTEM | FETCH | SOURCE")
set_property(CACHE RMC_ATAT_PROVIDER PROPERTY STRINGS SYSTEM FETCH SOURCE)

set(RMC_ATAT_GIT_REPOSITORY "https://github.com/reach2sayan/atat.git" CACHE STRING
        "ATAT git repository for FETCH mode")
set(RMC_ATAT_GIT_TAG "main" CACHE STRING
        "ATAT git tag/branch for FETCH mode")
set(RMC_ATAT_SOURCE_DIR "" CACHE PATH
        "Path to an existing ATAT checkout (SOURCE mode)")

# CMAKE_ARGS shared by FETCH/SOURCE to build only corrdump from an ATAT tree.
set(_atat_cmake_args
        -DCMAKE_BUILD_TYPE=Release
        # Silence warnings from ATAT's sources — not our code to fix. Affects
        # only this external build, never RMC's own targets.
        "-DCMAKE_CXX_FLAGS=-w"
        -DMAKENDVIEWER=OFF
        -DUSEGSL=OFF
        -DATAT_BUILD_CVM=OFF
        -DATAT_BUILD_TESTS=OFF
        -DUSEPYTHON=OFF
        -DATAT_ALLOW_FETCHCONTENT=ON)

if (RMC_ATAT_PROVIDER STREQUAL "SYSTEM")
    find_program(ATAT_CORRDUMP_PATH corrdump)
    if (ATAT_CORRDUMP_PATH)
        message(STATUS "mcsqs: using system corrdump at ${ATAT_CORRDUMP_PATH}")
    else ()
        message(STATUS
                "mcsqs: corrdump not found on PATH. mcsqs_rmc will still build; "
                "supply one at runtime with --corrdump <path> "
                "(install ATAT: https://axelvdw.github.io/atat/).")
    endif ()

elseif (RMC_ATAT_PROVIDER STREQUAL "FETCH")
    include(ExternalProject)
    ExternalProject_Add(atat_ext
            GIT_REPOSITORY "${RMC_ATAT_GIT_REPOSITORY}"
            GIT_TAG "${RMC_ATAT_GIT_TAG}"
            GIT_SHALLOW TRUE
            PREFIX "${CMAKE_BINARY_DIR}/atat"
            CMAKE_ARGS ${_atat_cmake_args}
            # Build only corrdump (avoids ATAT's ~60 other tools).
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target corrdump
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS "<BINARY_DIR>/bin/corrdump")
    ExternalProject_Get_Property(atat_ext BINARY_DIR)
    set(ATAT_CORRDUMP_PATH "${BINARY_DIR}/bin/corrdump" CACHE FILEPATH
            "Path to the built corrdump binary" FORCE)

elseif (RMC_ATAT_PROVIDER STREQUAL "SOURCE")
    if (NOT RMC_ATAT_SOURCE_DIR OR NOT EXISTS "${RMC_ATAT_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
                "RMC_ATAT_PROVIDER=SOURCE requires -DRMC_ATAT_SOURCE_DIR=<path to an "
                "ATAT checkout containing CMakeLists.txt>.")
    endif ()
    include(ExternalProject)
    ExternalProject_Add(atat_ext
            SOURCE_DIR "${RMC_ATAT_SOURCE_DIR}"
            PREFIX "${CMAKE_BINARY_DIR}/atat"
            CMAKE_ARGS ${_atat_cmake_args}
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target corrdump
            INSTALL_COMMAND ""
            BUILD_BYPRODUCTS "<BINARY_DIR>/bin/corrdump")
    ExternalProject_Get_Property(atat_ext BINARY_DIR)
    set(ATAT_CORRDUMP_PATH "${BINARY_DIR}/bin/corrdump" CACHE FILEPATH
            "Path to the built corrdump binary" FORCE)

else ()
    message(FATAL_ERROR
            "Unknown RMC_ATAT_PROVIDER='${RMC_ATAT_PROVIDER}' "
            "(expected SYSTEM | FETCH | SOURCE).")
endif ()

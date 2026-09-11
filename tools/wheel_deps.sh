#!/usr/bin/env bash
# cibuildwheel before-all (manylinux_2_28): Boost and oneTBB, the shared
# libraries _core links, into <cache>/prefix. Built once per container, and
# skipped when release.yml restored the prefix from its cache. auditwheel then
# bundles the .so files into the wheel (pyproject.toml points it at
# <cache>/prefix/lib). spdlog needs nothing here: with no system package,
# CMakeLists.txt fetches and builds it static.
#
# Usage: CXX=<g++ 15> tools/wheel_deps.sh <cache-dir>
set -euo pipefail

cache=${1:?usage: wheel_deps.sh <cache-dir>}
prefix="${cache}/prefix"
cxx=${CXX:?set CXX to the GCC 15 compiler}
tools=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
boost_version=1.89.0
tbb_version=2022.3.0
mkdir -p "${prefix}"

if [[ ! -f "${prefix}/.boost-${boost_version}" ]]; then
  bash "${tools}/build_boost.sh" "${prefix}" gcc "${cxx}"
  touch "${prefix}/.boost-${boost_version}"
fi

if [[ ! -f "${prefix}/.tbb-${tbb_version}" ]]; then
  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
  curl -fsSL --retry 3 \
    "https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v${tbb_version}.tar.gz" \
    | tar xz -C "${work}"
  # TBB_STRICT=OFF: its default -Werror meets warnings new to GCC 15.
  cmake -S "${work}/oneTBB-${tbb_version}" -B "${work}/build" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${cxx}" \
    -DCMAKE_INSTALL_PREFIX="${prefix}" -DCMAKE_INSTALL_LIBDIR=lib \
    -DTBB_TEST=OFF -DTBB_EXAMPLES=OFF -DTBB_STRICT=OFF
  cmake --build "${work}/build" --parallel "$(nproc)"
  cmake --install "${work}/build"
  touch "${prefix}/.tbb-${tbb_version}"
fi

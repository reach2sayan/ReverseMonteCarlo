#!/usr/bin/env bash
# Build the compiled Boost libraries RMC links, plus Boost's full header tree,
# from the 1.89 source release into <prefix>. One script for ci.yml,
# release.yml and the wheel build (tools/wheel_deps.sh), so every build links
# the same Boost, matching the local /opt/boost toolchain.
#
# ubuntu-24.04's libboost-all-dev is 1.83, which lacks Boost.Parser (1.87+)
# and Boost.Process v2's compiled library (1.86+).
#
# Usage: tools/build_boost.sh <prefix> <gcc|clang> <c++ compiler>
set -euo pipefail

prefix=${1:?usage: build_boost.sh <prefix> <gcc|clang> <c++ compiler>}
toolset=${2:?usage: build_boost.sh <prefix> <gcc|clang> <c++ compiler>}
cxx=${3:?usage: build_boost.sh <prefix> <gcc|clang> <c++ compiler>}
version=1.89.0
underscored=${version//./_}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
curl -fsSL --retry 3 \
  "https://archives.boost.io/release/${version}/source/boost_${underscored}.tar.bz2" \
  -o boost.tar.bz2
tar xf boost.tar.bz2
cd "boost_${underscored}"
echo "using ${toolset} : : ${cxx} ;" > user-config.jam
./bootstrap.sh \
  --with-libraries=program_options,serialization,container,context,filesystem,process,graph,nowide
./b2 -j"$(nproc)" \
  toolset="${toolset}" cxxstd=23 variant=release link=shared \
  --user-config=user-config.jam --prefix="${prefix}" install
# `b2 install` only copies headers for the --with libraries (and their scanned
# deps), which misses header-only modules RMC uses (Boost.Parser) and the
# Process v2 umbrella header. The release tarball ships the complete header
# tree, so install it wholesale.
cp -af boost "${prefix}/include/"

#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rmcgui {

// A render-thread-ready copy of the engine state at one logged step. The worker
// fills this (downcasting coordinates to float) under a mutex; the render thread
// copies it out and uploads to the GPU only when `generation` changes. Holds
// only what the renderer needs — never the full AtomicStructure (which is
// expensive to copy).
struct RenderSnapshot {
  std::vector<float> xyz;     // 3*natoms, Cartesian (x,y,z) per atom
  std::vector<int> z;         // natoms atomic numbers (colour/radius key)
  std::array<float, 9> box{}; // column-major 3x3 cell; all-zero ⇒ infinite BC
  std::size_t natoms = 0;

  std::uint64_t step = 0;
  std::uint64_t accepted = 0;
  std::uint64_t tried = 0;
  double chi2 = 0.0;

  // Bumped on every publish; 0 means "nothing rendered yet".
  std::uint64_t generation = 0;
};

} // namespace rmcgui

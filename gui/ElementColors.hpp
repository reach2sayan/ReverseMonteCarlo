#pragma once
#include <array>

namespace rmcgui {

// CPK-ish colour + a display radius (Å, scaled down so atoms don't overlap
// visually) keyed by atomic number. Header-only and engine-independent — the
// renderer keys directly on the int Z carried in the RenderSnapshot. Unknown or
// synthetic Z falls back to a neutral pink and a default radius.

struct ElementStyle {
  std::array<float, 3> color;
  float radius;
};

namespace detail {
struct ZStyle {
  int z;
  float r, g, b;
  float radius;
};

// A small curated table covering the elements RMC users hit most (metals,
// glass-formers, common light elements). Radii are ~0.4× covalent radius.
inline constexpr ZStyle kTable[] = {
    {1, 1.00f, 1.00f, 1.00f, 0.18f},  // H
    {6, 0.30f, 0.30f, 0.30f, 0.30f},  // C
    {7, 0.20f, 0.30f, 0.95f, 0.28f},  // N
    {8, 0.95f, 0.15f, 0.15f, 0.27f},  // O
    {13, 0.75f, 0.65f, 0.55f, 0.45f}, // Al
    {14, 0.55f, 0.60f, 0.65f, 0.44f}, // Si
    {22, 0.60f, 0.65f, 0.70f, 0.52f}, // Ti
    {26, 0.88f, 0.40f, 0.20f, 0.50f}, // Fe
    {28, 0.40f, 0.55f, 0.85f, 0.50f}, // Ni
    {29, 0.80f, 0.45f, 0.20f, 0.50f}, // Cu
    {30, 0.49f, 0.50f, 0.69f, 0.50f}, // Zn
    {40, 0.40f, 0.78f, 0.78f, 0.62f}, // Zr
    {47, 0.75f, 0.75f, 0.78f, 0.58f}, // Ag
    {79, 0.90f, 0.78f, 0.20f, 0.58f}, // Au
};
} // namespace detail

[[nodiscard]] inline ElementStyle element_style(int z) {
  for (const auto &e : detail::kTable) {
    if (e.z == z) {
      return {{e.r, e.g, e.b}, e.radius};
    }
  }
  // Fallback: distinct neutral pink, mid radius. Keyed so two unknown species
  // still differ a little by a cheap hash of Z.
  const float t = static_cast<float>((z * 2654435761u) & 0xFFu) / 255.0f;
  return {{0.85f, 0.45f + 0.4f * t, 0.75f}, 0.5f};
}

} // namespace rmcgui

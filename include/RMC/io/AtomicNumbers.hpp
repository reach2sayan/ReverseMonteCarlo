#pragma once
#include <algorithm>
#include <array>
#include <string_view>

namespace RMC::io {

struct Element {
  std::string_view symbol;
  int z;
};

// Periodic table, sorted by symbol so atomic_number() can binary-search.
// Keep this ordering invariant if you edit the table.
inline constexpr std::array<Element, 118> ELEMENTS = {{
    {"Ac", 89},  {"Ag", 47},  {"Al", 13},  {"Am", 95},  {"Ar", 18},
    {"As", 33},  {"At", 85},  {"Au", 79},  {"B", 5},    {"Ba", 56},
    {"Be", 4},   {"Bh", 107}, {"Bi", 83},  {"Bk", 97},  {"Br", 35},
    {"C", 6},    {"Ca", 20},  {"Cd", 48},  {"Ce", 58},  {"Cf", 98},
    {"Cl", 17},  {"Cm", 96},  {"Cn", 112}, {"Co", 27},  {"Cr", 24},
    {"Cs", 55},  {"Cu", 29},  {"Db", 105}, {"Ds", 110}, {"Dy", 66},
    {"Er", 68},  {"Es", 99},  {"Eu", 63},  {"F", 9},    {"Fe", 26},
    {"Fl", 114}, {"Fm", 100}, {"Fr", 87},  {"Ga", 31},  {"Gd", 64},
    {"Ge", 32},  {"H", 1},    {"He", 2},   {"Hf", 72},  {"Hg", 80},
    {"Ho", 67},  {"Hs", 108}, {"I", 53},   {"In", 49},  {"Ir", 77},
    {"K", 19},   {"Kr", 36},  {"La", 57},  {"Li", 3},   {"Lr", 103},
    {"Lu", 71},  {"Lv", 116}, {"Mc", 115}, {"Md", 101}, {"Mg", 12},
    {"Mn", 25},  {"Mo", 42},  {"Mt", 109}, {"N", 7},    {"Na", 11},
    {"Nb", 41},  {"Nd", 60},  {"Ne", 10},  {"Nh", 113}, {"Ni", 28},
    {"No", 102}, {"Np", 93},  {"O", 8},    {"Og", 118}, {"Os", 76},
    {"P", 15},   {"Pa", 91},  {"Pb", 82},  {"Pd", 46},  {"Pm", 61},
    {"Po", 84},  {"Pr", 59},  {"Pt", 78},  {"Pu", 94},  {"Ra", 88},
    {"Rb", 37},  {"Re", 75},  {"Rf", 104}, {"Rg", 111}, {"Rh", 45},
    {"Rn", 86},  {"Ru", 44},  {"S", 16},   {"Sb", 51},  {"Sc", 21},
    {"Se", 34},  {"Sg", 106}, {"Si", 14},  {"Sm", 62},  {"Sn", 50},
    {"Sr", 38},  {"Ta", 73},  {"Tb", 65},  {"Tc", 43},  {"Te", 52},
    {"Th", 90},  {"Ti", 22},  {"Tl", 81},  {"Tm", 69},  {"Ts", 117},
    {"U", 92},   {"V", 23},   {"W", 74},   {"Xe", 54},  {"Y", 39},
    {"Yb", 70},  {"Zn", 30},  {"Zr", 40}}};

static_assert(std::ranges::is_sorted(ELEMENTS, {}, &Element::symbol),
              "ELEMENTS must stay sorted by symbol for binary search");

// Symbol -> atomic number; returns 0 for an unknown symbol. O(log N) over the
// symbol-sorted table.
[[nodiscard]] constexpr int atomic_number(std::string_view symbol) {
  const auto it = std::ranges::lower_bound(
      ELEMENTS, symbol, {}, &Element::symbol);
  return (it != ELEMENTS.end() && it->symbol == symbol) ? it->z : 0;
}

// Atomic number -> symbol; returns {} for an out-of-range Z. Linear scan, since
// the table is ordered by symbol rather than Z.
[[nodiscard]] constexpr std::string_view element_symbol(int z) {
  for (const auto &e : ELEMENTS)
    if (e.z == z)
      return e.symbol;
  return {};
}

} // namespace RMC::io

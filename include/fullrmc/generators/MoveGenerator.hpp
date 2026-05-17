#pragma once
#include <fullrmc/core/BoundaryConditions.hpp>
#include <fullrmc/core/Types.hpp>
#include <memory>

namespace fullrmc {

struct IMoveGenerator {
  virtual ~IMoveGenerator() = default;
  virtual void generate(coords_t &coords,
                        std::span<const std::size_t> indices) = 0;
  virtual std::shared_ptr<IMoveGenerator> clone() const = 0;
};

// CRTP helper: concrete generators inherit this to auto-implement clone()
// and the virtual interface without boilerplate.
template <typename Derived> struct MoveGeneratorBase : IMoveGenerator {
  void generate(coords_t &coords,
                std::span<const std::size_t> indices) override {
    static_cast<Derived *>(this)->generate_impl(coords, indices);
  }
  std::shared_ptr<IMoveGenerator> clone() const override {
    return std::make_shared<Derived>(static_cast<const Derived &>(*this));
  }
};

// Utility: compute centroid of the given atom indices.
inline vec3_t centroid(const coords_t &coords,
                       std::span<const std::size_t> indices) noexcept {
  Eigen::Map<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> idx(
      indices.data(), static_cast<Eigen::Index>(indices.size()));

  return coords(idx, Eigen::all).colwise().mean().transpose();
}

} // namespace fullrmc

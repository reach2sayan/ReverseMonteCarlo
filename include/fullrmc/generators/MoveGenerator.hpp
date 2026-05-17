#pragma once
#include <fullrmc/core/BoundaryConditions.hpp>
#include <fullrmc/core/Types.hpp>
#include <memory>

namespace fullrmc {

class IMoveGenerator; // forward declaration for the friend declaration below

namespace detail {
struct GeneratorToken {
private:
  constexpr GeneratorToken() = default;
  friend class ::fullrmc::IMoveGenerator;
};
} // namespace detail

// Satisfied by any type that implements the passkey-gated generate interface.
template <typename T>
concept CMoveGenerator =
    requires(T &gen, detail::GeneratorToken tok, coords_t &coords,
             std::span<const std::size_t> indices) {
      gen.generate(tok, coords, indices);
    };

class IMoveGenerator {
public:
  // Public alias so concrete generators can name the token type in their
  // signatures without knowing about the detail namespace.
  using Token = detail::GeneratorToken;

  template <CMoveGenerator T>
  IMoveGenerator(T x)
      : self_(std::make_unique<MoveGeneratorModel<T>>(std::move(x))) {}
  IMoveGenerator(const IMoveGenerator &s) : self_{s.self_->clone()} {}
  IMoveGenerator(IMoveGenerator &&s) noexcept : self_{std::move(s.self_)} {}
  IMoveGenerator &operator=(const IMoveGenerator &s) {
    self_ = s.self_->clone();
    return *this;
  }
  IMoveGenerator &operator=(IMoveGenerator &&s) noexcept {
    self_ = std::move(s.self_);
    return *this;
  }

  void generate(coords_t &coords, std::span<const std::size_t> indices) {
    self_->generate(coords, indices);
  }

private:
  static Token make_token() noexcept { return {}; }
  struct MoveGeneratorConcept {
    virtual ~MoveGeneratorConcept() = default;
    virtual void generate(coords_t &, std::span<const std::size_t>) = 0;
    virtual std::unique_ptr<MoveGeneratorConcept> clone() const = 0;
  };
  template <CMoveGenerator T>
  struct MoveGeneratorModel final : MoveGeneratorConcept {
    explicit MoveGeneratorModel(T x) : data_(std::move(x)) {}
    void generate(coords_t &coords,
                  std::span<const std::size_t> indices) override {
      data_.generate(make_token(), coords, indices);
    }
    std::unique_ptr<MoveGeneratorConcept> clone() const override {
      return std::make_unique<MoveGeneratorModel<T>>(data_);
    }
    T data_;
  };

  std::unique_ptr<MoveGeneratorConcept> self_;
};

// CRTP mixin — inherit to document that Derived satisfies CMoveGenerator.
template <typename Derived> struct MoveGeneratorBase {};

// Utility: compute centroid of the given atom indices.
inline vec3_t centroid(const coords_t &coords,
                       std::span<const std::size_t> indices) noexcept {
  Eigen::Map<const Eigen::Array<std::size_t, Eigen::Dynamic, 1>> idx(
      indices.data(), static_cast<Eigen::Index>(indices.size()));
  return coords(idx, Eigen::all).colwise().mean().transpose();
}

} // namespace fullrmc

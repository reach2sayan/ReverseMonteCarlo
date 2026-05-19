#pragma once
#include <RMC/core/BoundaryConditions.hpp>
#include <RMC/core/Types.hpp>
#include <memory>
#include <optional>

namespace RMC {

class IMoveGenerator; // forward declaration for the friend declaration below

namespace detail {
struct GeneratorToken {
private:
  constexpr GeneratorToken() = default;
  friend class ::RMC::IMoveGenerator;
};
} // namespace detail

template <typename T>
concept CMoveGenerator =
    requires(T &gen, detail::GeneratorToken tok, coords_t &coords,
             std::span<const std::size_t> indices) {
      gen.generate(tok, coords, indices);
    };

// Optional extension: generator implements its own accept/reject (e.g. HMC).
// If satisfied, IMoveGenerator exposes rejection_override() so Engine::settle()
// can defer to the generator instead of using constraints_.should_reject().
template <typename T>
concept CMoveGeneratorWithRejectionOverride =
    CMoveGenerator<T> && requires(const T &gen) {
      { gen.rejection_override() } -> std::convertible_to<std::optional<bool>>;
    };

// Optional extension: generator mutates species (elements/atomic_numbers), not
// just coordinates. Engine will save/restore species snapshots around such moves.
template <typename T>
concept CMoveGeneratorWithSpeciesModification =
    CMoveGenerator<T> && requires(const T &gen) {
      { gen.modifies_species() } -> std::convertible_to<bool>;
    };

class IMoveGenerator {
public:
  using Token = detail::GeneratorToken;

  template <CMoveGenerator T>
  constexpr IMoveGenerator(T x)
      : self_(std::make_unique<MoveGeneratorModel<T>>(std::move(x))) {}
  constexpr IMoveGenerator(const IMoveGenerator &s) : self_{s.self_->clone()} {}
  constexpr IMoveGenerator(IMoveGenerator &&s) noexcept
      : self_{std::move(s.self_)} {}
  constexpr IMoveGenerator &operator=(const IMoveGenerator &s) {
    self_ = s.self_->clone();
    return *this;
  }
  constexpr IMoveGenerator &operator=(IMoveGenerator &&s) noexcept {
    self_ = std::move(s.self_);
    return *this;
  }

  constexpr void generate(coords_t &coords,
                          std::span<const std::size_t> indices) {
    self_->generate(coords, indices);
  }

  [[nodiscard]] constexpr std::optional<bool>
  rejection_override() const noexcept {
    return self_->rejection_override();
  }

  [[nodiscard]] constexpr bool modifies_species() const noexcept {
    return self_->modifies_species();
  }

private:
  static Token make_token() noexcept { return {}; }
  struct MoveGeneratorConcept {
    virtual ~MoveGeneratorConcept() = default;
    virtual void generate(coords_t &, std::span<const std::size_t>) = 0;
    virtual std::optional<bool> rejection_override() const noexcept {
      return std::nullopt;
    }
    virtual bool modifies_species() const noexcept { return false; }
    virtual std::unique_ptr<MoveGeneratorConcept> clone() const = 0;
  };
  template <CMoveGenerator T>
  struct MoveGeneratorModel final : MoveGeneratorConcept {
    constexpr explicit MoveGeneratorModel(T x) : data_(std::move(x)) {}
    constexpr void generate(coords_t &coords,
                            std::span<const std::size_t> indices) override {
      data_.generate(make_token(), coords, indices);
    }
    constexpr std::optional<bool> rejection_override() const noexcept override {
      if constexpr (CMoveGeneratorWithRejectionOverride<T>) {
        return data_.rejection_override();
      } else {
        return std::nullopt;
      }
    }
    constexpr bool modifies_species() const noexcept override {
      if constexpr (CMoveGeneratorWithSpeciesModification<T>) {
        return data_.modifies_species();
      } else {
        return false;
      }
    }
    constexpr std::unique_ptr<MoveGeneratorConcept> clone() const override {
      return std::make_unique<MoveGeneratorModel<T>>(data_);
    }
    T data_;
  };

  std::unique_ptr<MoveGeneratorConcept> self_;
};

template <typename Derived> struct MoveGeneratorBase {};
FORCE_INLINE vec3_t centroid(const coords_t &coords,
                             std::span<const std::size_t> indices) noexcept {
  return coords(indices, Eigen::all).colwise().mean();
}

} // namespace RMC

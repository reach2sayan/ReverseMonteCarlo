#pragma once
#include <fullrmc/core/Types.hpp>
#include <memory>

namespace fullrmc {

class IGroupSelector; // forward for friend declaration

namespace detail {
struct SelectorToken {
private:
  constexpr SelectorToken() = default;
  friend class ::fullrmc::IGroupSelector;
};
} // namespace detail

// Satisfied by any type that implements the passkey-gated selector interface.
template <typename T>
concept CGroupSelector =
    requires(T &s, detail::SelectorToken tok, std::size_t n,
             std::size_t group_idx, bool accepted) {
      { s.select(tok, n) } -> std::convertible_to<std::size_t>;
      s.feedback(tok, group_idx, accepted);
    };

class IGroupSelector {
public:
  using Token = detail::SelectorToken;

  template <CGroupSelector T>
  IGroupSelector(T x) : self_(std::make_unique<SelectorModel<T>>(std::move(x))) {}
  IGroupSelector(const IGroupSelector &s) : self_{s.self_->clone()} {}
  IGroupSelector(IGroupSelector &&s) noexcept : self_{std::move(s.self_)} {}
  IGroupSelector &operator=(const IGroupSelector &s) {
    self_ = s.self_->clone();
    return *this;
  }
  IGroupSelector &operator=(IGroupSelector &&s) noexcept {
    self_ = std::move(s.self_);
    return *this;
  }

  std::size_t select(std::size_t n_groups) { return self_->select(n_groups); }
  void feedback(std::size_t group_idx, bool accepted) {
    self_->feedback(group_idx, accepted);
  }

private:
  static Token make_token() noexcept { return {}; }

  struct SelectorConcept {
    virtual ~SelectorConcept() = default;
    virtual std::size_t select(std::size_t n_groups) = 0;
    virtual void feedback(std::size_t group_idx, bool accepted) = 0;
    virtual std::unique_ptr<SelectorConcept> clone() const = 0;
  };

  template <CGroupSelector T> struct SelectorModel final : SelectorConcept {
    explicit SelectorModel(T x) : data_(std::move(x)) {}
    std::size_t select(std::size_t n) override {
      return data_.select(make_token(), n);
    }
    void feedback(std::size_t gi, bool acc) override {
      data_.feedback(make_token(), gi, acc);
    }
    std::unique_ptr<SelectorConcept> clone() const override {
      return std::make_unique<SelectorModel<T>>(data_);
    }
    T data_;
  };

  std::unique_ptr<SelectorConcept> self_;
};

// CRTP mixin — inherit to document that Derived satisfies CGroupSelector.
template <typename Derived> struct SelectorBase {};

} // namespace fullrmc

#pragma once
#include <RMC/core/TypeErasure.hpp>
#include <RMC/core/Types.hpp>
#include <memory>

namespace RMC {

class GroupSelector; // forward for friend declaration

namespace detail {
struct GroupSelectorToken {
private:
  constexpr GroupSelectorToken() = default;
  friend class ::RMC::GroupSelector;
};
} // namespace detail

// Satisfied by any type that implements the passkey-gated selector interface.
template <typename T>
concept CGroupSelector =
    requires(T &s, detail::GroupSelectorToken tok, std::size_t n,
             std::size_t group_idx, bool accepted) {
      { s.select(tok, n) } -> std::convertible_to<std::size_t>;
      s.feedback(tok, group_idx, accepted);
    };

// NOTE: the generator names the type-erased members GroupSelectorConcept /
// GroupSelectorModel, derived from the wrapper name (the hand-written versions
// were SelectorConcept / SelectorModel).
#define RMC_GROUPSELECTOR_METHODS                                              \
  ((0, std::size_t, select, (std::size_t n_groups), 1, (n_groups), , ,        \
    WITH_TOKEN))                                                              \
  ((0, void, feedback, (std::size_t group_idx, bool accepted), 2,             \
    (group_idx, accepted), , , WITH_TOKEN))
RMC_DEFINE_ERASED_TYPE(GroupSelector, RMC_GROUPSELECTOR_METHODS)
#undef RMC_GROUPSELECTOR_METHODS

template <typename Derived> struct SelectorBase {};

} // namespace RMC

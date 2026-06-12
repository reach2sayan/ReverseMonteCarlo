#pragma once
#include <RMC/selectors/GroupSelector.hpp>
#include <cstddef>

namespace RMC {

enum class RecursiveMode {
  Refine,  // retry same group while moves are ACCEPTED (exploit productive
           // region)
  Explore, // retry same group while moves are REJECTED (find a good move)
};

// Wraps any GroupSelector and retries the same group for up to `max_retries`
// additional steps after a triggering outcome.
//
//   Refine: on acceptance, lock onto that group for up to max_retries more
//           steps. First rejection resets to normal delegation.
//
//   Explore: on rejection, lock onto that group for up to max_retries more
//            attempts. First acceptance resets to normal delegation.
struct RecursiveGroupSelector : SelectorBase<RecursiveGroupSelector> {
  RecursiveMode mode{RecursiveMode::Refine};
  int max_retries{5};

  explicit RecursiveGroupSelector(GroupSelector inner,
                                  RecursiveMode m = RecursiveMode::Refine,
                                  int retries = 5)
      : mode(m), max_retries(retries), inner_(std::move(inner)) {}

  std::size_t select(GroupSelector::Token, std::size_t n_groups) {
    if (retries_left_ > 0 && last_gi_ < n_groups) {
      --retries_left_;
      return last_gi_;
    }
    last_gi_ = inner_.select(n_groups);
    retries_left_ = 0;
    return last_gi_;
  }

  void feedback(GroupSelector::Token /*tok*/, std::size_t gi, bool accepted) {
    inner_.feedback(gi, accepted);
    const bool trigger = (mode == RecursiveMode::Refine) ? accepted : !accepted;
    const bool cancel = !trigger;
    if (trigger && retries_left_ == 0) {
      retries_left_ = max_retries;
    } else if (cancel) {
      retries_left_ = 0;
    }
  }

private:
  GroupSelector inner_;
  std::size_t last_gi_{0};
  int retries_left_{0};
};

} // namespace RMC

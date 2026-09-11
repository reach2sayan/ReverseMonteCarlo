#pragma once
#include <boost/describe/enum.hpp>
#include <RMC/selectors/DirectionalOrderSelector.hpp>
#include <RMC/selectors/OrderedSelector.hpp>
#include <RMC/selectors/RandomSelector.hpp>
#include <RMC/selectors/SmartRandomSelector.hpp>

#include <cstddef>
#include <memory>
#include <variant>

namespace RMC {

struct GroupSelector;

enum class RecursiveMode {
  Refine,  // retry the same group while moves are ACCEPTED (exploit)
  Explore, // retry the same group while moves are REJECTED (find a good move)
};
BOOST_DESCRIBE_ENUM(RecursiveMode, Refine, Explore)

// Wraps another selector and, after a triggering outcome (acceptance for
// Refine, rejection for Explore), returns the same group for up to
// `max_retries` more steps; the opposite outcome cancels the lock.
class RecursiveGroupSelector {
public:
  RecursiveMode mode;
  int max_retries;

  explicit RecursiveGroupSelector(GroupSelector inner,
                                  RecursiveMode m = RecursiveMode::Refine,
                                  int retries = 5);
  std::size_t select(std::size_t n_groups);
  void feedback(std::size_t gi, bool accepted);

private:
  std::unique_ptr<GroupSelector> inner_;
  std::size_t last_gi_{0};
  int retries_left_{0};
};

// The group-selection policy: one of the selectors above, dispatched by value.
struct GroupSelector
    : std::variant<RandomSelector, WeightedRandomSelector, OrderedSelector,
                   SmartRandomSelector, DirectionalOrderSelector,
                   RecursiveGroupSelector> {
  using variant::variant;
  std::size_t select(std::size_t n_groups) {
    return std::visit([n_groups](auto &s) { return s.select(n_groups); }, *this);
  }
  // Adaptive selectors learn from the outcome; the rest ignore it.
  void feedback(std::size_t gi, bool accepted) {
    std::visit(
        [&](auto &s) {
          if constexpr (requires { s.feedback(gi, accepted); }) {
            s.feedback(gi, accepted);
          }
        },
        *this);
  }
};

inline RecursiveGroupSelector::RecursiveGroupSelector(GroupSelector inner,
                                                      RecursiveMode m,
                                                      int retries)
    : mode(m), max_retries(retries),
      inner_(std::make_unique<GroupSelector>(std::move(inner))) {}

inline std::size_t RecursiveGroupSelector::select(std::size_t n_groups) {
  if (retries_left_ > 0 && last_gi_ < n_groups) {
    --retries_left_;
    return last_gi_;
  }
  retries_left_ = 0;
  return last_gi_ = inner_->select(n_groups);
}

inline void RecursiveGroupSelector::feedback(std::size_t gi, bool accepted) {
  inner_->feedback(gi, accepted);
  const bool trigger = (mode == RecursiveMode::Refine) == accepted;
  if (!trigger) {
    retries_left_ = 0;
  } else if (retries_left_ == 0) {
    retries_left_ = max_retries;
  }
}

} // namespace RMC

#include <RMC/MultiFrameEngine.hpp>

#include <boost/assert.hpp>

#include <numeric>
#include <vector>

namespace RMC {

void MultiFrameEngine::do_initialise() {
  BOOST_ASSERT_MSG(store_.size() > 0, "MultiFrameEngine: no frames added");
  const std::size_t N = store_.size();
  constraints_.set_n_frames(N);

  const std::size_t n_atoms = store_[0].size();
  std::vector<std::size_t> all_idx(n_atoms);
  std::iota(all_idx.begin(), all_idx.end(), std::size_t{0});

  for (std::size_t k = 0; k < N; ++k) {
    constraints_.set_active_frame(k);
    constraints_.compute_before_move(store_[k].coordinates, all_idx);
  }
  // Leave active_frame at 0 (arbitrary; reset per step).
  constraints_.set_active_frame(0);
}

} // namespace RMC

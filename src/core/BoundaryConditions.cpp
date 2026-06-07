#include <RMC/core/BoundaryConditions.hpp>
#include <cmath>

namespace RMC {

void PeriodicBC::set_box(const mat3_t &box) {
  box_ = box;
  inv_box_ = box.inverse();
  volume_ = std::abs(box.determinant());
}

} // namespace RMC

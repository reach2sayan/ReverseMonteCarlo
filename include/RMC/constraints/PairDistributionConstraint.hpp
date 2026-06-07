#pragma once
#include <RMC/constraints/PairHistogram.hpp>

namespace RMC {
using PairDistributionConstraint = PairFunctionConstraint<PairNorm::PDF>;
static_assert(CConstraint<PairDistributionConstraint>,
              "PairDistributionConstraint must satisfy the CConstraint concept");
} // namespace RMC

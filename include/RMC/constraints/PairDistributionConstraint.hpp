#pragma once
#include <RMC/constraints/PairHistogram.hpp>

namespace RMC {
using PairDistributionConstraint = PairFunctionConstraint<PairNorm::PDF>;
} // namespace RMC

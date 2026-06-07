#pragma once
#include <RMC/constraints/PairHistogram.hpp>

namespace RMC {
using PairCorrelationConstraint = PairFunctionConstraint<PairNorm::PCF>;
static_assert(CConstraint<PairCorrelationConstraint>,
              "PairCorrelationConstraint must satisfy the CConstraint concept");
} // namespace RMC

#pragma once
#include <fullrmc/core/Types.hpp>
#include <fullrmc/core/Structure.hpp>
#include <fullrmc/core/BoundaryConditions.hpp>
#include <fullrmc/core/Group.hpp>
#include <fullrmc/core/AtomsCollector.hpp>
#include <fullrmc/selectors/GroupSelector.hpp>
#include <fullrmc/constraints/ConstraintCollection.hpp>
#include <fullrmc/io/Checkpoint.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/count.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/log/trivial.hpp>

#include <functional>
#include <memory>
#include <vector>
#include <filesystem>
#include <cstdint>
#include <optional>

namespace fullrmc {

// Callback invoked at the end of every `log_every` steps.
// Signature: (step, accepted_count, tried_count, total_chi2)
using StepCallback = std::function<void(std::uint64_t, std::uint64_t,
                                         std::uint64_t, real_t)>;

class Engine {
public:
    // ---- Construction / configuration ----
    explicit Engine(AtomicStructure structure, BoundaryConditions bc);

    // Add a group (takes ownership of generator via Group::generator).
    void add_group(Group g);

    // Build groups automatically: one group per atom with default TranslationGenerator.
    void build_atomic_groups(real_t min_amp = 0.0, real_t max_amp = 0.2,
                              std::uint32_t seed = 42);

    void set_selector(std::unique_ptr<IGroupSelector> s);
    void add_constraint(std::unique_ptr<IConstraint> c);

    // Optional: save a checkpoint every `every` accepted steps.
    void set_checkpoint(std::filesystem::path path, std::uint64_t every = 5000);
    void set_step_callback(StepCallback cb, std::uint64_t log_every = 1000);

    // ---- Run ----
    void run(std::uint64_t n_steps);
    // Stop when total chi² drops below target or max_steps is reached.
    void run_until(real_t target_chi2, std::uint64_t max_steps = 0);

    // ---- State accessors ----
    [[nodiscard]] const AtomicStructure&    structure()    const noexcept { return structure_; }
    [[nodiscard]] AtomicStructure&          structure()          noexcept { return structure_; }
    [[nodiscard]] const BoundaryConditions& boundary()     const noexcept { return bc_; }
    [[nodiscard]] io::EngineStats           stats()        const noexcept;
    [[nodiscard]] ConstraintCollection&     constraints()        noexcept { return constraints_; }

private:
    // Execute a single MC trial.
    void step();

    // Apply periodic wrapping to the atoms that just moved.
    void apply_pbc(std::span<const index_t> moved);

    AtomicStructure              structure_;
    BoundaryConditions           bc_;
    std::vector<Group>           groups_;
    std::unique_ptr<IGroupSelector> selector_;
    ConstraintCollection         constraints_;
    AtomsCollector               collector_;

    // Statistics
    std::uint64_t n_steps_total_   {0};
    std::uint64_t n_steps_tried_   {0};
    std::uint64_t n_steps_accepted_{0};

    // Checkpoint state
    std::optional<std::filesystem::path> checkpoint_path_;
    std::uint64_t checkpoint_every_{5000};

    // Logging callback
    StepCallback  step_cb_;
    std::uint64_t log_every_{1000};
};

} // namespace fullrmc

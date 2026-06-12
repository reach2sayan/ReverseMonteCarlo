# Design

This document describes how RMC is put together: the data that flows through a
refinement, the engine's step pipeline, and the four extension contracts
(constraints, generators, samplers, selectors). It deliberately stops at the
*contracts* — the public, stable surface each component must honour. It does not
explain how the type-erasure machinery in `core/TypeErasure.hpp` is implemented;
treat each `Constraint`, `MoveGenerator`, `Sampler` and `GroupSelector` as a
value-semantic handle that holds any type satisfying the matching concept.

For build instructions and the user-facing API, see [README.md](README.md).

## What RMC does

Reverse Monte Carlo refines an atomic configuration so that quantities computed
*from* it (a pair distribution g(r), a structure factor S(Q), an angular
distribution, cluster correlations, …) match experimental or target values. It
does this by repeatedly proposing a small random change to the structure,
scoring how the total error χ² changed, and accepting or rejecting the change
under a stochastic rule. Over many steps the configuration relaxes toward one
consistent with the data.

The same machinery drives `mcsqs_rmc`: an SQS search is just RMC where the moves
are species swaps and the target is the set of disordered cluster correlations.

## The objects

```
AtomicStructure   coordinates (N×3) + elements / names / residues / molecule_ids
BoundaryConditions  PeriodicBC(box) | InfiniteBC(volume)   — variant
Group             a set of atom indices + the MoveGenerator that perturbs them
Constraint        scores one term of χ²; caches before/after error
Sampler           the accept/reject rule given (χ²_before, χ²_after)
GroupSelector     picks which group moves next; learns from accept/reject
Engine            owns all of the above and runs the step loop
```

A **Group** is the unit of a move: an explicit list of atom indices plus the
generator that displaces them. One-atom groups give classic atomic RMC; a
molecule's atoms in one group with a `RotationGenerator` give rigid-body moves.

A **Constraint** owns one additive term of the total error. It caches the error
*before* and *after* a proposed move so the engine can form Δχ² without a full
recompute, and exposes whether it is *rigid* (a hard geometric gate that never
contributes to χ²) or *soft* (contributes to χ² and is judged by the Sampler).

## The engine pipeline

`Engine` derives from `EngineBase<Engine>` (CRTP). The base owns the step loop,
the three-tier acceptance logic, and orthogonal feature policies; `Engine`
supplies the per-engine specifics (frame storage and selection) through CRTP
customization points. Knowing the pipeline is enough to reason about every
component's contract — each contract method is called at a known stage.

One `step()` runs four stages, then settles, then logs:

```
select()                  → pick a frame + group         → TrialCtx (or skip)
snapshot_and_score_before → save state; χ²_before for every constraint
propose_move              → group.generator.generate(...) ; wrap into the box
score_after               → χ²_after for the affected constraints
settle                    → decide accept/reject; commit or roll back
```

The stages are chained monadically (`std::optional::and_then`): if `select()`
returns no group (nothing eligible), the chain short-circuits and the step is a
no-op.

```mermaid
flowchart TD
    A([step]) --> B[select<br/>pick frame + group]
    B -->|no eligible group| Z[no-op]
    B -->|TrialCtx| C[snapshot_and_score_before<br/>save state · χ²_before]
    C --> D[propose_move<br/>generator.generate · wrap into box]
    D --> E[score_after<br/>χ²_after for affected constraints]
    E --> F[settle<br/>decide accept / reject]
    F -->|accept| G[commit: after → before]
    F -->|reject| H[restore snapshot · reset constraints]
    G --> L[log every log_every steps]
    H --> L
    Z --> L
    L --> A
```

### Three-tier acceptance

`settle()` calls `decide_rejection()`, which resolves the move in priority
order (`EngineBase::decide_rejection`):

1. **Generator override.** A gradient generator (HMC leapfrog, MALA Langevin)
   that implements its own accept/reject returns a verdict via
   `rejection_override()`; the engine defers to it entirely.
2. **Rigid hard gate.** Otherwise, if *any* rigid constraint got worse, the move
   is rejected outright. Rigid constraints contribute 0 to χ², so they act
   purely as geometric gates (e.g. a minimum-separation bond bound).
3. **Sampler soft gate.** Otherwise the `Sampler` decides from the change in the
   **total soft error** (sum over non-rigid constraints), using one uniform draw
   from the engine's dedicated acceptance RNG.

```mermaid
flowchart TD
    A([decide_rejection]) --> B{generator owns<br/>accept / reject?}
    B -->|yes| BV[use rejection_override verdict]
    B -->|no| C{any rigid<br/>constraint worse?}
    C -->|yes| CR[reject — hard gate]
    C -->|no| D{Sampler.accept<br/>Δ soft error · u01}
    D -->|true| DA[accept]
    D -->|false| DR[reject]
```

On accept, every constraint's cached `after` error becomes its new `before`; on
reject, the structure snapshot is restored and constraints reset. This is why
the constraint contract has explicit `accept`/`reject` calls — they commit or
discard the cached state, they don't recompute.

### Feature policies

`EngineBase` composes orthogonal, compile-time features as empty-base policies
(`[[no_unique_address]]`, so they cost nothing when off):

| Policy | Responsibility |
|---|---|
| `WithSpecies` | save/restore element labels around species-mutating moves |
| `WithFeedback` | forward accept/reject to an adaptive selector |
| `WithCollector` | stage/commit atom removals (`RemoveGenerator`) |
| `WithBestTracking` | keep the lowest-χ² configuration seen |
| `WithCheckpoint` | periodically serialise state to disk |

A policy hooks into a specific pipeline point (e.g. species save in
`snapshot_and_score_before`, feedback at the end of `settle`). Adding a feature
means adding a policy and its hook, not editing the hot loop.

### Multi-frame

`Engine` can hold several structural frames against one dataset. Each step picks
one frame and one group; pair/angle constraints average their computed profile
across frames, so χ² is the error of the *averaged* curve — this damps
over-fitting to a single configuration. Single-frame refinement is just the
N = 1 case; the averaging lives in the constraints, not the loop. Per-step
histogram updates are incremental (O(K·N) for a K-atom group) rather than a full
O(N²) rebuild.

## The four contracts

Each extension point is a concept. A type satisfying the concept can be wrapped
in the matching value-semantic handle and handed to the engine. All interface
methods take a **passkey token** as their first argument — a private tag only the
wrapper can construct — so the methods are callable only through the wrapper, not
directly. When you write a component you accept the token but ignore it (or
inherit a base that does); you never construct one.

In the contract tables below the token argument is omitted for readability.

### Constraint — `constraints/Constraint.hpp`

A constraint owns one additive term of χ² and caches its before/after error.

| Method | Contract |
|---|---|
| `compute_before_move(coords, moved)` | Compute & cache the error for the current coords. `moved` = indices about to move; a constraint may recompute only affected terms. |
| `compute_after_move(coords, moved)` | Same, for the post-move coords → cached `after`. |
| `accept()` / `reject()` | Commit (`after`→`before`) or discard the cached after-state. `noexcept`. |
| `standard_error()` / `standard_error_before()` | Cached after / before error. **Rigid constraints return 0** for both — they never enter χ². |
| `should_reject()` | Per-constraint downhill test. Used only for the rigid hard gate. |
| `is_rigid()` | If true, any worsening is an immediate hard rejection and the term is excluded from χ². |
| `is_singular()` | If true, at most one instance of this type may be added. |
| `computation_cost()` | Relative cost hint (O(1), O(N), O(N²)); cheaper constraints are evaluated first so an early rigid rejection can skip expensive ones. |
| `name()` | Human label. **The only method called without the token.** |
| `set_boundary_conditions(bc)` / `set_collector(col)` | Engine wiring, called at registration. |
| `set_n_frames(n)` / `set_active_frame(k)` | Multi-frame support; no-ops for single-frame constraints. |
| `initialise()` | One-time setup (build target tables, prime histograms), called from `Engine::initialise()`. |

**Don't implement this by hand.** Derive from a base and implement a single
`compute_error(coords, moved) → double`:

- `ConstraintBase<Derived>` — a soft constraint. The base caches before/after,
  implements accept/reject/should_reject, and provides minimum-image helpers
  (`distance`, `distance_sq`) and `absent(i)` (true when atom `i` is staged for
  removal — skip its terms).
- `RigidConstraintBase<Derived>` — a hard gate. Same, but `standard_error*()`
  return 0 and `is_rigid()` is true.
- `SingularConstraintBase<Derived>` — soft, but limited to one instance.

### MoveGenerator — `generators/MoveGenerator.hpp`

A generator perturbs the coordinates of one group in place.

| Method | Contract |
|---|---|
| `generate(coords, indices)` | Mutate `coords` for the atoms in `indices`. The only required method. |
| `rejection_override()` *(opt)* | Return `std::optional<bool>` to own the accept/reject decision (gradient movers). Defaults to `nullopt` — defer to the engine. |
| `modifies_species()` *(opt)* | Return true if the move changes element labels, not just positions, so the engine snapshots species. Defaults to false. |

The two optional methods are opt-in: implement them and the wrapper exposes
them; omit them and the wrapper reports the safe default. The engine wraps moved
atoms back into the periodic box after `generate()`, so a generator works in
unwrapped Cartesian space.

### Sampler — `sampling/Sampler.hpp`

The soft accept/reject rule.

| Method | Contract |
|---|---|
| `accept(e_before, e_after, step, u01) → bool` | Decide from the total soft error before/after, the global step index, and one uniform draw `u01 ∈ [0,1)`. Must be deterministic given its arguments. |

Built-ins: `GreedySampler` (strict downhill, the default — RMC's historical
behaviour), `MetropolisSampler` (fixed-T), `AnnealingSampler` (geometric
cooling). Because the sampler receives `u01` rather than its own RNG, the engine
controls the stochastic stream and ensemble replicas stay reproducible.

### GroupSelector — `selectors/GroupSelector.hpp`

Chooses which group moves each step and may learn from outcomes.

| Method | Contract |
|---|---|
| `select(n_groups) → size_t` | Return an index in `[0, n_groups)`. |
| `feedback(group_idx, accepted)` | Told whether the last move on `group_idx` was accepted; adaptive selectors use it to re-weight. Stateless selectors ignore it. |

Built-ins: `RandomSelector`, `OrderedSelector`, `WeightedRandomSelector`,
`SmartRandomSelector` (biases toward groups with a higher acceptance rate),
`RecursiveGroupSelector`, `DirectionalOrderSelector`. Feedback only flows when
the `WithFeedback` policy is active.

## I/O and analysis

- `io/` — readers/writers for PDB, VASP POSCAR/CONTCAR and LAMMPS data files,
  plus two-/multi-column experimental data and checkpoint serialisation. Readers
  return a `Result<…>` (`boost::leaf::result`); the cell, when the format carries
  one, becomes the `BoundaryConditions`.
- `analysis/` — `compute_gr` (pair distribution, total + per-element-pair
  partials) and `compute_adf` (angular distribution). These are standalone
  functions over a configuration, used both by the CLI's `--gr` / `--adf-compute`
  modes and directly from the library. They share the histogram kernels the pair
  constraints use.

## The SQS extension (`extensions/mcsqs/`)

`mcsqs_rmc` reuses the engine with two pieces swapped in:

- **Generator:** `SpeciesSwapGenerator` exchanges occupancies *within a
  sublattice* (sublattices inferred from PDB residue names), keeping site
  positions fixed.
- **Constraint:** `ClusterCorrelationConstraint` scores the weighted χ²
  deviation of multi-body cluster correlations from their disordered targets.

The ATAT pipeline reads a `rndstr.in` primitive lattice, drives the vendored
`corrdump` (via `boost::process`) to enumerate cluster orbits, expands them over
the supercell, and runs the search. See the README's *SQS search* section for
the CLI.

```mermaid
flowchart TD
    L[rndstr.in lattice] --> CD[corrdump<br/>enumerate cluster orbits]
    CD --> EX[expand over supercell<br/>+ build cluster basis]
    EX --> EN[engine: SpeciesSwap moves<br/>vs ClusterCorrelation targets]
    EN --> OUT[bestsqs.pdb + str.out]
    OUT --> CK[corrdump cross-check]
```

The legacy pipeline replaces the lattice + `corrdump` stages with a fixed-site
structure and a pre-enumerated cluster-orbit file, feeding the same engine stage
(no cross-check). Both build the same cluster basis the engine consumes.

## Extending RMC

The contracts above are the extension surface. Adding a component is local —
write a type, satisfy a concept, register it — and never touches the engine loop.

### Add a constraint

1. Create `constraints/MyConstraint.hpp`. Derive from `ConstraintBase<MyConstraint>`
   (soft) or `RigidConstraintBase<MyConstraint>` (hard gate).
2. Implement `compute_error(const coords_t&, std::span<const std::size_t> moved)
   → double`. Use the base's `distance`/`distance_sq` for minimum-image
   distances and `absent(i)` to skip removed atoms. For a cheap incremental
   update, recompute only the terms touching `moved`.
3. Override `computation_cost()` if it is O(N) or worse, so the engine orders it
   after cheaper constraints.
4. For experimental-data constraints, add a setter (e.g. `set_experimental_data`)
   and do table setup in an `initialise()` override.
5. Register at runtime: `engine.add_constraint(MyConstraint{...})`. To expose it
   on the CLI, add an entry to the `experimental_targets()` table and an
   `attach_constraint<…>` call in `src/RMCRunner.cpp` — both are data-driven, so
   it's one row each.

### Add a move generator

1. Create `generators/MyGenerator.hpp` with
   `void generate(coords_t&, std::span<const std::size_t>)`.
2. To own accept/reject (e.g. a Hamiltonian/gradient move), add
   `std::optional<bool> rejection_override() const`. To mutate species, add
   `bool modifies_species() const`.
3. Attach to a group: `g.generator.emplace(MyGenerator{...})`.

### Add a sampler or selector

- **Sampler:** implement `bool accept(double before, double after, uint64_t
  step, double u01)`. Keep it a pure function of its arguments — draw no RNG of
  your own. Register with `engine.set_sampler(MySampler{...}, seed)`.
- **Selector:** implement `size_t select(size_t n_groups)` and `void
  feedback(size_t, bool)`. Register with `engine.set_selector(MySelector{...})`.

### Add an input/output format

Add a reader returning `Result<AtomicStructure>` (plus a cell if the format
carries one) under `io/`, then a case in `RMCRunner`'s input variant
(`select_input` / `load_*`) and the extension dispatch in
`write_structure_by_ext`.

### Add a feature policy

A cross-cutting behaviour that must hook into the step loop (not a single move)
becomes an `EnginePolicies.hpp` policy with a hook at the relevant pipeline
stage, exposed through a CRTP accessor on `Engine`. Use this only when a
constraint/generator can't express the feature locally — most extensions
shouldn't need it.

### A note on the passkey token

Every contract method takes a leading passkey token. You accept it in your
signature (the bases handle this for you) but never construct or inspect one —
its only purpose is to ensure components are called through their wrapper. If a
concept check fails to compile, the usual cause is a missing token parameter or a
`const`/`noexcept` qualifier that doesn't match the contract table above.

# PLAN — Reusable / extensible recursive sketches (foundation for sketch caching)

## Context

`sketched_gmres` re-sketches each window basis vector every iteration it sits in the truncation
window; we want to **cache a vector's recursive sketch and reuse/extend it across calls**. The first
attempt — memoizing each adaptive extension *batch* by its event index — is wrong: the recursive
sketch pairs sample slabs **positionally** across bonds, and the adaptive loop extends only bonds
`k+1:N` per event, so per-bond counts are non-uniform; "batch #n" then couples different global
block indices on either side of a count-step, i.e. its identity depends on residual history. A
recompute-and-compare assertion confirmed: a reused batch ≠ a fresh one (a benign-looking but real
~1e-8 perturbation; cache-off was bit-reproducible, so it was the cache, not noise).

The fix (validated this session in `/tmp/extend_proto.jl`, `/tmp/extend_proto2.jl`; design in
`memory/sketch-caching-design.md`): store each bond's columns in **global-sample order** and extend
**only deficient bonds, right-to-left, reusing the right neighbour's existing columns at the same
global positions**. Validated bit-identical to `tt_recursive_sketch` and history-independent
(one-shot == multi-shot, `maxdiff = 0.0`). Mixed block ranks (`block_rks` init vs `block_rks_inc`
ext) become **two groups** `[init: p_init·brv₁ | ext: p_ext·brv₂]` with disjoint seed namespaces and
joint normalization (validated unbiased over 401 seeds: 1.003 / 0.977 / 0.987). Raw partials stored
**un-normalized**; the `1/√count` factor applied only on finalize, so reuse needs no rescale.

Invariant: per-bond counts non-decreasing (`count[k] ≤ count[k+1]`, `k<N`; bond `N` recurses against
the tiled `ones` boundary) — the adaptive loop already satisfies this. Convention: seeds change
globally (statistical equivalence, as with `945b9ce`); exact benchmark numbers shift, sketches stay
valid. The existing kernel `contract_sketch_core_backwards!(W, A, S, V)` already supports the
incremental step via `V = W[k+1][:,:, s_range]` — no kernel change needed.

Already committed (stand regardless of this work): randomized assembly in `sketched_gmres`
(`2995afc`, ~40% of cookie wall), geometric-growth prealloc for the sum-overload extension
(`a433686`, ~9%), addressable per-block seeding (`945b9ce`/`64fbe64`).

## Stage 1 — Group-namespace seeding  (`src/sketches/blocks.jl`)
Add `group::Int` to `_block_seed` (mixed into the hash) and thread `group::Int=0` through
`generate_sketch_blocks` to its per-block `_block_seed` call. Default `0` keeps single-group callers
behaving the same (values shift once, globally).
**Verify:** package loads; `dev_tests/check_sketch_isometry.jl` ≈ isometric; `Pkg.test()` passes.

## Stage 2 — `extend_recursive_sketch!` primitive  (new `src/sketches/extend.jl`, included from `src/sketches.jl`)
- `mutable struct SketchGroup{T}` holding `W::Vector{Array{T,3}}` (raw partials,
  `W[l]::(rks[l],brv[l],counts[l])`, `W[N+1]=ones(1,1,1)`), `counts` (`counts[N+1]=1`), `brv`,
  `block_rks`, `group`; constructor computes `brv` and seeds an empty group.
- `extend_recursive_sketch!(g, A, target; seed, orthogonal, timer)`: for `k=N:-1:1`, if
  `g.counts[k] < target[k]`, generate the `add` missing blocks via `generate_sketch_blocks(seed, k,
  g.counts[k], T, g.brv[k+1], dims[k], g.brv[k], add, orthogonal; reverse=true, group=g.group)`,
  permute `(2,1,3,4)`, contract against `V = view(g.W[k+1],:,:, g.counts[k]+1:target[k])` (tiled
  `ones` for `k=N`), geometric-grow `g.W[k]`, set `g.counts[k]=target[k]`. Land a correct allocating
  version first, then port the `sketch_buffer`/`contract_buffer` + `GC.@preserve` treatment from
  `tt_recursive_sketch`'s reverse branch.
- `finalize_cols(groups, l, rks_l; weighting=:equal)` = `hcat` of the groups' reshaped columns with a
  **per-group scalar weight** `w_g` (group `g`'s columns scaled by `√w_g`, `Σ w_g = 1` keeps the
  combined sketch an unbiased isometry). Combining unequal-variance unbiased block estimators ⇒ the
  weighting is the design knob; support three, pluggable:
  - `:equal` — `w_g ∝ counts[l]_g` (→ overall `1/√Σcounts`, the prototype scheme; oblivious, simplest).
  - `:column` — `w_g ∝ counts[l]_g · brv[l]_g` (∝ total columns; embedding-dim model `Var∝1/brv`,
    up-weights richer blocks; reduces to `:equal` for uniform `block_rks`; oblivious, partition-consistent).
  - `:precision` — empirical inverse-variance (BLUE): `w_g ∝ counts_g / s²_g` from the per-group sample
    variance of the per-block estimates; min-variance, model-free, mildly data-adaptive. Applied at the
    norm/boundary level; exact form for the per-bond residual sketch TBD when measuring.
  Bring-up uses `:equal`; the production default is chosen empirically in Stage 6.
**Verify (new `test/test_sketch_extend.jl`):** single group == `tt_recursive_sketch` bit-identical;
one-shot vs two-shot bit-identical; two-group mixed `E[‖Sy‖²/‖y‖²]`≈1 over many seeds.

## Stage 3 — `CachedSketch` type + finalize  (`src/sketches/extend.jl`)
`CachedSketch{T} = Vector{SketchGroup{T}}` (1 uniform / 2 mixed). `cached_sketch(A, block_rks,
block_rks_inc)` (one group if `block_rks_inc==block_rks`, else two: init `group=0`, ext `group=1`);
`ensure_columns!(cache, A, want; seed, orthogonal)` grows the ext (or sole) group until each bond has
`≥ want[l]` total columns, init group built once and frozen; `sketch_matrix(cache, l, rks_l; weighting)`
= `finalize_cols` (threads `weighting` through).
**Verify:** grow-then-grow-more == from-scratch-to-larger, bit-identical (with `:equal`).

## Stage 4 — Rebuild adaptive extension on the primitive  (`src/tt_adaptive_rounding.jl`)
Sum overload first: per-term working `W[j][l]` becomes `sketch_matrix(cache_j, l, rks)`; "need more
columns" calls `ensure_columns!`; drop the geometric-cat renorm (normalization now in
`finalize_cols`). Add optional `caches::Union{Nothing,Vector{CachedSketch}}` (reuse caller caches, or
throwaway when `nothing`). Then the single-TT overload (`block_rks_inc≠block_rks` ⇒ two-group cache —
the mixed path the old scheme never handled addressably).
**Verify:** both overloads rel-err-vs-exact at machine precision and reproducible; with vs without
`caches` bit-identical (single-thread BLAS).

### Stage 4 status — COMPLETE (all five reverse-sweep overloads on the cache primitive)
DONE: sum overload (`a7037a6`,`7fc7fef`), single-TT two-group mixed-`block_rks` (`f05882b`), buffer
optimization (`e36a426`,`6983e04`). All cache-on==cache-off bit-identical.
- **Operator group** (sketch of `A·y`): `src/sketches/extend_operator.jl` (`373858b`),
  `test/test_sketch_extend_operator.jl`. `OperatorSketchGroup` W[l]::(y.ttv_rks[l], A.tto_rks[l],
  brv[l], samples) (4-D), per-sample 5-arg kernel (blocks already match `S=(γ,ζ,c)`, no permute),
  boundary `ones(1,1,1,1)`, 3-D `finalize_cols_operator`.
  - **operator-residual** `(Atto, y, b, ε)` rewired (`5544c55`): op cache for `A·y` + vector cache for
    `b`, shared seed/block_rks. Machine-precision rounding, cache on/off bit-identical.
  - **mixed-operator** `(α, A, y, ε)` rewired (`<this commit>`): op cache for `A·y[1]` + vector caches
    for `y[j≥2]`. Replaced dead `WAy_init/sketch_rks_init` with `caches`/`weighting`. Machine-precision,
    cache on/off bit-identical, `sketched_gmres` operator path converges end-to-end.
- **Kronecker/Hadamard group** (sketch of `y₁⊙…⊙y_M`): `src/sketches/extend_kronecker.jl` (`4ae52a9`),
  `test/test_sketch_extend_kronecker.jl`. `W[l]` is (M+2)-D `(rks₁[l],…,rks_M[l], brv[l], samples)`,
  per-sample `contract_sketch_core_kronecker_backwards!` (blocks match S, no permute), boundary
  `ones(1×(M+2))`. **Finalize is 2-D** `(∏ᵢ y[i].ttv_rks[l], total_cols)` — the flattened form the
  overload consumes. **Hadamard `NTuple{M}` overload** rewired (`2026e78`): two-group
  (init `block_rks` + ext `block_rks_inc`), `weighting=:column` matches legacy renorm. Machine-precision
  (M=2,3), cache on/off bit-identical.

NOTE: in this sandbox `julialauncher` intermittently hangs without spawning a worker; invoke the real
binary directly: `/Users/cazeaux/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia`.

## Stage 5 — Wire cache into `sketched_gmres`  (`src/tt_solvers.jl`)  — DONE
`CachedSketch` per window vector, parallel to `B_window` (push/`popfirst!`), passed into `_sg_round`
(leading non-window terms uncached); `reuse_sketches::Bool=true`.
- Summand path: `c2cedf6` (validated bit-identical on cookie).
- **Operator path** (`_sg_round(op::TToperator, …)`): now passes `caches=[nothing; win_caches]` to the
  mixed-operator overload (the `A·pv` term changes each step → `nothing`; window terms cached).
  `test/test_solvers.jl` covers both paths on a small synthetic operator: `reuse_sketches` true vs
  false bit-identical end-to-end (‖xT−xF‖/‖xF‖≈2e-16), same iterations, converges.
**Verify (single-thread):** `reuse_sketches` true vs false bit-identical end-to-end; cookie sketched
GMRES same iterations/accuracy, faster.

## Stage 6 — Choose the block-averaging default empirically  — DONE → `:column`
Measured the boundary norm estimate `‖Sy‖/‖y‖` (the quantity that sets τ) and interior partial-norm
spreads under each `weighting` on the mixed two-group path, over many seeds, on a synthetic mixed-rank
TT and the Matérn rank-621 TT (`dev_tests/weighting_variance.jl`). Findings:
- **`:column` is the production default.** Least biased (mean closest to ‖y‖) and lowest/near-lowest
  variance; biggest win over `:equal` when the ext group is pure-KRP (`block_rks_inc=1`, ×equal≈0.6–0.9
  std), shrinking to ≈1 as `block_rks_inc→block_rks` (where it provably reduces to `:equal`).
- `:precision` (provisional empirical inverse-variance) is **biased low** (means 0.35–0.77 on Matérn):
  data-dependent weights correlate with the noisy sample → systematic underestimate at small counts.
  Kept for experimentation only; would need a debiased estimator to be usable. Documented in
  `src/sketches/extend.jl::_group_weights`.
- Cookie / sketched-GMRES paths (sum, operator) are single-group (uniform `block_rks`) ⇒ `:equal`≡
  `:column` there; the mixed path Matérn represents is the only place the choice matters.
Action taken: unified the default to `:column` across all overloads (was `:equal` on operator-residual
and mixed-operator — a bit-identical no-op there since they are single-group, but correct/future-proof).
No accuracy/iteration regression (full adaptive-rounding suite green).

## Stage 7 — Forward-sketch mirror  (`reverse=false`)
Mirror the whole cache stack for the forward sweep: `SketchGroup`/`extend!` recursing **left→right**
(reuse the *left* neighbour's columns), `contract_sketch_core_forwards!` kernels, boundary at bond 1.
A `reverse::Bool` field on `SketchGroup` (and the constructor / `_heuristic_p` / `ensure_columns!`
direction) selects the sweep. Then any overload that uses forward sketches picks it up. Same
validation: forward single group == `tt_recursive_sketch(reverse=false)` bit-identical; one-shot ==
multi-shot; cache-on == off.

## Risks / notes
- Broadest-blast change (sketch core used by `ttrand_rounding`, `stta`, all adaptive overloads,
  sketched RR/GMRES). Stage-gated: each stage validated (bit-identity + isometry + `Pkg.test()`)
  before the next.
- All equivalence checks under `BLAS.set_num_threads(1)` (multithread BLAS is non-deterministic at
  ~1e-15 and amplifies through adaptive rank decisions). Benchmarks under plain OpenBLAS (`CLAUDE.md`).
- Caching's cookie speedup is small (~few %); the value is a correct general reuse primitive **and** a
  cheaper non-cached extension (today each extension rebuilds the full `k+1:N` recursion). Don't
  over-fit to cookie timing.

## Commits (one per stage)
1. Add group namespace to sketch block seeding.
2. Add `extend_recursive_sketch!` incremental sketch primitive (+ tests).
3. Add per-vector `CachedSketch` with prefix-reuse / extend.
4. Rebuild sum-overload adaptive extension on `CachedSketch`; then single-TT (mixed `block_rks`).
5. Reuse window-vector sketches across `sketched_gmres` iterations.
6. Set default sketch block-averaging weighting from measurements.

using TensorOperations
using LinearAlgebra
using Random
using TimerOutputs

# Per-bond budget knobs (`ℓ_min`, `ℓ_max`) accept either a scalar (uniform across bonds) or a
# length-(N+1) vector (per-bond targets, used by the fixed-rank `ttrand_rounding` wrappers). These
# helpers normalize them: `_bond_vec` to a length-(N+1) vector (scalar ⇒ fill, so the scalar path is
# unchanged), `_scalar_max` to a scalar for the (variance-driven) `n_samples` default.
_bond_vec(x::AbstractVector{<:Integer}, N::Int) = x
_bond_vec(x::Integer, N::Int) = fill(Int(x), N+1)
_scalar_max(x::AbstractVector{<:Integer}) = maximum(x)
_scalar_max(x::Integer) = x

# Append new orthonormal directions to Q from the residual sketch Sₖ and return
# how many were added. Two plain-QR orthogonalization sweeps:
#   1. orthonormalize Sₖ
#   2. project ⊥ Q and re-orthonormalize ("twice is enough" — drives orthogonality
#      against Q to machine precision; a single pass would drift)
#
# There is no `max_add`: the caller sizes the fed slice to at most
# max_basis - current_cols columns (the ℓ_inc_eff clamp at each call site), so the
# number of new directions can never exceed the bond's remaining room. The
# algebraic TT rank bound min(ℓ_max, prod(dims[1:k]), prod(dims[k+1:end])) is thus
# enforced upstream by sizing the residual, not downstream by capping the output.
# Because the slice is kept within the bond's remaining room it stays full-rank in
# the common case (plain qr! adds no spurious completion columns and sweep 2's
# Qn_proj is full-rank, so qr! re-orthonormalizes without needing to reveal rank);
# on the rare rank-deficient slice (Q-overlap detected), the SVD branch truncates
# to the effective rank.
"""
    expand_basis!(Q_storage, current_cols, Sₖ) -> rank_n

In-place adaptive basis expansion. The current orthonormal basis Q is held as
`view(Q_storage, :, 1:current_cols)`; new orthonormal directions (orthonormalized,
projected out of the current span, then re-orthonormalized) are written into
`Q_storage[:, current_cols+1:current_cols+rank_n]`. Returns `rank_n`, the number
of new columns added (caller is responsible for advancing `current_cols`).

The fed `Sₖ` is already sliced by the caller to at most `max_basis - current_cols`
columns, so `rank_n` cannot exceed the bond's remaining room — no `max_add` is
needed. Replaces the prior `hcat(Q, Qn)` pattern that reallocated Q every
iteration (quadratic in the final basis size): with a preallocated `Q_storage`
of max-basis width this collapses to O(rank_n) writes per call.
"""
function expand_basis!(Q_storage::AbstractMatrix{T}, current_cols::Int,
                       Sₖ::AbstractMatrix{T}) where {T<:Number}
  Q = view(Q_storage, :, 1:current_cols)

  # Sweep 1: orthonormalize the residual sketch.
  Qn, _ = qr!(Sₖ); Qn = Matrix(Qn)

  # Sweep 2: project ⊥ Q and re-orthonormalize. The squared-Frobenius comparison
  # tr_full vs tr_proj equals ‖Q'·Qn‖_F², measuring how much of Qn lives in
  # span(Q); when negligible, Qn_proj is full-rank and qr! re-orthonormalizes,
  # otherwise SVD applies an absolute rank cut at round(tr_proj) (the integer
  # effective rank of (I−QQ')Qn).
  Qn_proj = Qn - Q * (Q'*Qn)
  tr_full = size(Qn_proj, 2)
  tr_proj = sum(abs2, Qn_proj)
  local Qn_new::Matrix{T}
  local rank_n::Int
  if abs(tr_full - tr_proj) < 16 * eps() * tr_full
    Qf, _ = qr!(Qn_proj); Qn_new = Matrix(Qf); rank_n = size(Qn_new, 2)
  else
    k = clamp(round(Int, tr_proj), 0, minimum(size(Qn_proj)))
    k == 0 && return 0
    F = svd(Qn_proj)
    Qn_new = F.U[:, 1:k]
    rank_n = k
  end
  rank_n == 0 && return 0

  @views Q_storage[:, current_cols+1:current_cols+rank_n] .= Qn_new
  return rank_n
end

"""
    ttrand_rounding_adaptive(y::TTvector{T,N}, ε::Real;
                              ℓ_min=4, ℓ_inc=4, ℓ_max=maximum(y.ttv_rks),
                              init_f=0.0, n_samples=…, orthogonal=true, block_rks=N,
                              block_rks_inc=…, seed=1234, timer=TimerOutput()) -> TTvector{T,N}

Adaptive randomized TT rounding to a user-specified Frobenius-norm tolerance.

For each bond k = 1..N-1, sweeps left-to-right and incrementally grows both the residual
sketch `W` and the orthogonal factor `Q` until a probe-based estimator of the local
truncation error falls below the per-bond budget `τ_bond = ε * ‖y‖_F / √(N-1)`.

# Algorithm
At bond k with left accumulator `Y_k`:

1. Build an initial reverse recursive sketch `W` of width ~`ℓ_min` columns.
2. Repeat:
   - Form `S_new = V(Y_k) · W_new`, the columns of `W[k+1]` that have not yet been
     absorbed into `Q`.
   - Compute the residual `R = (I - Q Q') S_new`.
   - Since the TT-stack sketch is an expected isometry, the unbiased estimator of the
     true bond residual is
       `est = ‖R‖_F * √(p_total / p_unabsorbed)`,
     where `p_total` is the cumulative block count at bond k+1 and `p_unabsorbed` is the
     block count contributed by the as-yet-unabsorbed columns. The factor undoes the
     `√(p_unabsorbed/p_total)` rescaling applied when new columns are concatenated into
     the globally `1/√(p_total)`-normalised sketch.
   - If `est ≤ τ_bond` and `Q` already has at least one column, stop.
   - Otherwise extend `Q` with the new orthonormal directions: `Q := [Q | qr(R).Q]`,
     mark those columns absorbed, and extend `W` by drawing an additional fresh
     reverse recursive sketch and concatenating it with proper renormalisation.

# Arguments
- `y::TTvector{T,N}`: input TT vector
- `ε::Real`: target relative Frobenius error

# Keyword Arguments
- `ℓ_min::Int=4`: target width (columns) of the initial per-bond sketch
- `ℓ_inc::Int=4`: additive floor on the per-iteration column growth of the basis
- `n_samples::Int`: width of the residual sketch used for the error estimator
- `init_f::Real=0.0`: if >0, per-bond initial sketch width = ceil(init_f · bond cap)
- `ℓ_max::Int=maximum(y.ttv_rks)`: cap on the output rank `out_rks[k+1]`
- `orthogonal::Bool=true`: orthogonal vs Gaussian sketch blocks
- `block_rks::Int=2`: block-rank parameter (see `tt_recursive_sketch`); smaller values
  give finer rank control at slightly higher per-iteration cost
- `seed::Int=1234`: base seed; per-bond and per-iteration seeds are derived
- `timer::TimerOutput`: optional profiling timer

# Returns
- `TTvector{T,N}` with left-orthogonal cores (`ot[k]=1` for k<N) whose ranks are
  determined adaptively. With high probability, `‖y - ŷ‖_F ≲ ε · ‖y‖_F`.

# References
Al Daas, Ballard, Grigori, Martínez Aguilar, Saibaba & Verma (2025),
"Adaptive Randomized Tensor Train Rounding using Khatri-Rao Products"
(arXiv:2511.03598), Algorithms 2 and 6.
"""
function ttrand_rounding_adaptive(y::TTvector{T,N}, ε::Real;
                                   ℓ_min::Union{Int,AbstractVector{Int}}=4,
                                   ℓ_inc::Int=4,
                                   ℓ_max::Union{Int,AbstractVector{Int}}=maximum(y.ttv_rks),
                                   init_f::Real=0.0,
                                   n_samples::Int=max(N÷2, ℓ_inc, ceil(Int, 0.1 * _scalar_max(ℓ_max))),
                                   orthogonal::Bool=true,
                                   block_rks::Int=N,
                                   block_rks_inc::Int=max(1, N÷4),
                                   seed::Int=1234,
                                   cache=nothing,
                                   weighting::Symbol=:column,
                                   timer::TimerOutput=TimerOutput()) where {T,N}
  @assert n_samples >= ℓ_inc "n_samples ($n_samples) must be ≥ ℓ_inc ($ℓ_inc)"
  # ℓ_min / ℓ_max may be scalars (uniform) or per-bond length-(N+1) vectors (e.g. a fixed-rank target);
  # normalize to vectors so the loop indexes [k+1] uniformly. Scalars ⇒ fill ⇒ behavior unchanged.
  ℓ_min_vec = _bond_vec(ℓ_min, N); ℓ_max_vec = _bond_vec(ℓ_max, N)
  # MATLAB-style per-bond initial sketch width: Init_b_k = ceil(init_f · min(out_rks[k]·dims[k], y.ttv_rks[k+1])).
  # When init_f > 0, this overrides ℓ_min on a per-bond basis (still respecting ℓ_min as a floor),
  # so the algorithm starts the adaptive loop with a small basis at narrow bonds and converges to
  # the requested tolerance instead of overshooting it. ℓ_min stays as the global minimum.
  # The upfront recursive sketch is sized for the *worst* bond's Init_b_k to bound the width.
  @timeit timer "ttrand_rounding_adaptive" begin
    dims = y.ttv_dims
    vec = Vector{Array{T,3}}(undef, N)
    # Per-bond Init_b_k = max(ℓ_min, ceil(init_f · max_cols_k)), where
    # max_cols_k bounds the bond's rank capacity. Upper bound on out_rks[k]
    # is min(prod(dims[1:k-1]), max(y.ttv_rks)); for budget we use the input
    # rank y.ttv_rks[k] as a proxy (matches the deterministic rounding cap).
    function initb_k(k::Int)
        max_cols_k = min(y.ttv_rks[k] * dims[k], y.ttv_rks[k+1])
        init_f > 0 ? max(ℓ_min_vec[k+1], ceil(Int, init_f * max_cols_k)) : ℓ_min_vec[k+1]
    end
    ℓ_min_global = init_f > 0 ? maximum(initb_k(k) for k in 1:N-1) : maximum(ℓ_min_vec)
    # Two-group reusable cache: a frozen initial group (block_rks) sized to ℓ_min_global+n_samples,
    # plus a growable extension group (block_rks_inc). When the caller passes `cache`, it is reused
    # and extended in place across calls; otherwise a throwaway is built. Groups are combined by
    # `weighting` (:column matches the legacy per-column renormalization). The initial sketch must
    # cover the worst bond's Init_b_k plus n_samples.
    cache === nothing && (cache = cached_sketch(T, y, block_rks, block_rks_inc, ℓ_min_global+n_samples; seed=seed, orthogonal=orthogonal, timer=timer))
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)
    _scols(l) = sum(g.brv[l]*g.counts[l] for g in cache.groups)
    sketch_rks = [_scols(l) for l=1:N+1]
    # Only the active bond's right-neighbour W[k+1] is materialized at a time (re-derived from the
    # cache each bond and after each growth, into a reused buffer); W[1] (boundary) once for the norm.
    W = Vector{Matrix{T}}(undef, N+1)
    @timeit timer "reverse_sketch" begin
      W[1] = sketch_matrix(cache, 1, y.ttv_rks[1]; weighting=weighting)
    end

    # Estimated Frobenius norm of y via the TTStack sketch: E[‖W[1]‖²] = ‖y‖².
    y_norm = norm(W[1])
    τ = ε * y_norm / sqrt(N - 1)

    # Randomized sketching and orthogonalization. Local tensors use (L, I, R) layout.
    @timeit timer "orthogonalization" begin
      yₖ = reshape(y.ttv_vec[1], 1, dims[1], y.ttv_rks[2])
      @inbounds for k in 1:N-1
        max_basis = bond_rank_cap(dims, k, ℓ_max_vec[k+1])
        # Per-bond initial sketch width (MATLAB-style); ℓ_min remains as floor.
        ℓ_min_k = init_f > 0 ? max(ℓ_min_vec[k+1], ceil(Int, init_f * min(out_rks[k]*dims[k], y.ttv_rks[k+1]))) : ℓ_min_vec[k+1]
        # Materialize the active bond's right-neighbour from the cache (it may have grown earlier).
        @timeit timer "reverse_sketch" remat_into!(W, k+1, cache, y.ttv_rks[k+1]; weighting=weighting)
        sketch_rks[k+1] = _scols(k+1)
        # Randomized QR decomposition
        @timeit timer "Randomized adaptive QR decomposition" begin
          @timeit timer "Sketch" begin
            Zₖ = zeros(T, out_rks[k], dims[k], ℓ_min_k)
            Wₖ₊₁ = W[k+1][:,1:ℓ_min_k]
            @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁)  Zₖ[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*Wₖ₊₁[αₖ₊₁,ρₖ₊₁]
            Zₖ = reshape(Zₖ, out_rks[k]*dims[k], ℓ_min_k)
          end
          @timeit timer "QR" begin
            Q_factor, _ = qr!(Zₖ)
            Q_init = Matrix(Q_factor)
            # Preallocate Q at max_basis cols once per bond; expand_basis! writes
            # into the next slot in place, avoiding the quadratic hcat realloc.
            # The QR of an (m, ℓ_min_k) matrix gives a Q with min(m, ℓ_min_k) cols, so
            # current_cols starts at the actual width — never larger than max_basis.
            current_cols = size(Q_init, 2)
            buf_cols = max(max_basis, current_cols)
            # undef (not zeros): unused cols are written by expand_basis! before
            # they're ever read, so page-zeroing is wasted work.
            Q_storage = Matrix{T}(undef, out_rks[k]*dims[k], buf_cols)
            @views Q_storage[:, 1:current_cols] .= Q_init
            Q = view(Q_storage, :, 1:current_cols)
            ℓ = ℓ_min_k
          end

          # S_full holds only the *active window* of yₖ × W[k+1] — the n_samples
          # cols that the residual estimator will read next. As ℓ advances, we
          # shift the buffer left and contract only the ℓ_inc_eff new cols at
          # the tail (vs the old code re-contracting all n_samples cols per
          # iter). When W is extended, we renormalise S_full's valid cols in
          # place — new cols are then contracted from the already-renormalised
          # W portion, so no separate "S_extra" step is needed.
          # Allocate as 2D from the start (a 3D reshape view scopes the
          # tensoropt write) to keep S_full's type stable as Matrix{T}.
          @timeit timer "Initial residual sketch" begin
            S_full = zeros(T, out_rks[k]*dims[k], n_samples)
            let S_full_3d = reshape(S_full, out_rks[k], dims[k], n_samples),
                W_view = view(W[k+1], :, ℓ_min_k+1:ℓ_min_k+n_samples)
              @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁)  S_full_3d[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*W_view[αₖ₊₁,ρₖ₊₁]
            end
          end
          Sₖ = Matrix{T}(undef, out_rks[k]*dims[k], n_samples)
          @timeit timer "Residual sketch" begin
            copyto!(Sₖ, S_full)
            mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
          end

          @timeit timer "Adaptive basis expansion" begin
            while norm(Sₖ) > τ * sqrt(n_samples/sketch_rks[k+1]) && current_cols < max_basis
              @timeit timer "Add new orthonormal directions" begin
                # Geometric growth: absorb max(ℓ_inc, 0.2·rank(Q)) cols per iter,
                # capped at the bond's remaining room (max_basis - current_cols) so
                # the factorization is never handed more columns than the bond can
                # absorb — keeps the residual full-rank in the common case.
                ℓ_inc_eff = clamp(max(ℓ_inc, ceil(Int, 0.2 * current_cols)), 1, min(n_samples, max_basis - current_cols))
                rank_n = expand_basis!(Q_storage, current_cols, view(Sₖ, :, 1:ℓ_inc_eff))
                if rank_n == 0
                  break
                end
                current_cols += rank_n
                Q = view(Q_storage, :, 1:current_cols)
                ℓ += ℓ_inc_eff
              end
              # Shift S_full left by ℓ_inc_eff cols (drop absorbed cols).
              @timeit timer "S_full shift" begin
                # In-place memmove-style shift (copyto! on overlapping views
                # falls back to a slow safe path; linear-index unsafe_copyto!
                # routes to memmove which handles overlap natively and fast).
                m_rows = size(S_full, 1)
                n_keep = n_samples - ℓ_inc_eff
                GC.@preserve S_full Base.unsafe_copyto!(S_full, 1, S_full, m_rows*ℓ_inc_eff+1, m_rows*n_keep)
              end
              @timeit timer "Recursive sketch" begin
                if sketch_rks[k+1] < ℓ+n_samples
                  s_prev_kp1 = sketch_rks[k+1]
                  # Grow the extension group (block_rks_inc) so bonds k+1:N reach ≥ ℓ+n_samples total
                  # columns; the frozen init group is untouched. Re-materialize the active bond.
                  want = copy(sketch_rks)
                  for l = k+1:N
                    want[l] = max(want[l], ℓ+n_samples)
                  end
                  ensure_columns!(cache, y, want; seed=seed, orthogonal=orthogonal, timer=timer)
                  remat_into!(W, k+1, cache, y.ttv_rks[k+1]; weighting=weighting)
                  for l = k+1:N
                    sketch_rks[l] = _scols(l)
                  end
                  # Renormalise the kept S_full cols (still in the OLD normalization).
                  S_full[:, 1:n_samples-ℓ_inc_eff] .*= sqrt(s_prev_kp1 / sketch_rks[k+1])
                end
              end
              @timeit timer "S_full tail contraction" begin
                # Fill the last ℓ_inc_eff cols of S_full from the (renormalised) W
                # via a single mul! straight into the S_full slice — eliminates
                # the (out_rks[k]*dims[k]) × ℓ_inc_eff intermediate `tail_buf`
                # and the subsequent copyto!.
                W_tail = view(W[k+1], :, ℓ+n_samples-ℓ_inc_eff+1:ℓ+n_samples)
                S_tail = view(S_full, :, n_samples-ℓ_inc_eff+1:n_samples)
                mul!(S_tail,
                     reshape(yₖ, out_rks[k]*dims[k], y.ttv_rks[k+1]),
                     W_tail)
              end
              @timeit timer "Residual sketch" begin
                copyto!(Sₖ, S_full)
                mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
              end
            end
          end

          @timeit timer "Update core" begin
            out_rks[k+1] = current_cols
            # Copy active cols out of Q_storage so the (possibly oversized)
            # buffer can be GC'd; the core gets exactly its needed storage.
            vec[k] = reshape(Q_storage[:, 1:current_cols], out_rks[k], dims[k], out_rks[k+1])
            ot[k] = 1
          end
        end
        @timeit timer "Update yₖ" begin
          # Explicit ordering: contract (vec[k], yₖ) → T1 first, then T1 × ttv_vec[k+1].
          # The naive @tensoropt ordering would materialise a (ρₖ, iₖ, iₖ₊₁, αₖ₊₂)
          # intermediate (up to ~750 MB at the widest Matern bond); the order below
          # keeps the intermediate at (ρₖ₊₁, αₖ₊₁) — a few tens of KB.
          # T1[ρₖ₊₁, αₖ₊₁] = sum_{ρₖ, iₖ} vec[k][ρₖ, iₖ, ρₖ₊₁] * yₖ[ρₖ, iₖ, αₖ₊₁]
          T1 = reshape(vec[k], out_rks[k]*dims[k], out_rks[k+1])' *
               reshape(yₖ, out_rks[k]*dims[k], y.ttv_rks[k+1])
          # yₖ₊₁[ρₖ₊₁, iₖ₊₁, αₖ₊₂] = sum_{αₖ₊₁} T1[ρₖ₊₁, αₖ₊₁] * y.ttv_vec[k+1][αₖ₊₁, iₖ₊₁, αₖ₊₂]
          yₖ₊₁_mat = T1 * reshape(y.ttv_vec[k+1], y.ttv_rks[k+1], dims[k+1]*y.ttv_rks[k+2])
          yₖ = reshape(yₖ₊₁_mat, out_rks[k+1], dims[k+1], y.ttv_rks[k+2])
        end
      end
      vec[N] = reshape(yₖ, out_rks[N], dims[N], out_rks[N+1])
    end
    return TTvector{T,N}(N,vec,dims,out_rks,ot)
  end
end


"""
    ttrand_rounding_adaptive(α::Vector{T}, y::Vector{TTvector{T,N}}, ε::Real;
                              ℓ_min=4, ℓ_inc=4, ℓ_max=…,
                              orthogonal=true, block_rks=N,
                              seed=1234, timer=TimerOutput()) -> TTvector{T,N}

Adaptive randomized rounding of the linear combination `α[1]·y[1] + … + α[m]·y[m]`.

All `m` sketches share the same RNG seed so the random projection is linear in `α`; the
boundary sketch then directly estimates `‖∑ α[j]·y[j]‖` via `‖∑ α[j]·W[j][1]‖`. Each
adaptive extension also reuses the current `seed` across all `j` before incrementing it.
"""
function ttrand_rounding_adaptive(α::Vector{T}, y::Vector{TTvector{T,N}}, ε::Real;
                                   ℓ_min::Union{Int,AbstractVector{Int}}=4,
                                   ℓ_inc::Int=4,
                                   n_samples::Int=max(N÷2, ℓ_inc),
                                   ℓ_max::Union{Int,AbstractVector{Int}}=sum(maximum(yⱼ.ttv_rks) for yⱼ in y),
                                   orthogonal::Bool=true,
                                   block_rks::Int=N,
                                   seed::Int=1234,
                                   caches=nothing,
                                   timer::TimerOutput=TimerOutput()) where {T,N}
  @assert n_samples >= ℓ_inc "n_samples ($n_samples) must be ≥ ℓ_inc ($ℓ_inc)"
  @timeit timer "ttrand_rounding_adaptive" begin
    m = length(α)
    @assert length(y) == m
    dims = y[1].ttv_dims
    @assert all(y[j].ttv_dims == dims for j=2:m)
    # Per-bond ℓ_min/ℓ_max (scalar ⇒ uniform, unchanged); the fixed-rank wrapper passes vectors.
    ℓ_min_vec = _bond_vec(ℓ_min, N); ℓ_max_vec = _bond_vec(ℓ_max, N); ℓ_min_max = maximum(ℓ_min_vec)

    vec = Vector{Array{T,3}}(undef, N)
    # Per-term reusable sketch caches (uniform block_rks ⇒ one group each). `caches` may be nothing
    # (build all throwaway), or a length-m vector whose entries are either a persisted CachedSketch
    # (reused & extended in place across calls, e.g. a GMRES window vector) or nothing (term changes
    # every call, e.g. an operator summand — build a throwaway). The criterion below is
    # total-independent (the sketch_rks factor cancels the 1/√count column normalization), so the
    # grow pattern is free.
    _fresh(j) = cached_sketch(T, y[j], block_rks, block_rks, ℓ_min_max+n_samples; seed=seed, orthogonal=orthogonal, timer=timer)
    caches = caches === nothing ? [_fresh(j) for j=1:m] :
                                  [caches[j] === nothing ? _fresh(j) : caches[j] for j=1:m]
    # All terms are combined at a COMMON per-bond sample count `p_target` (a prefix of each cache),
    # so caches grown to different sizes across calls (a grown window vector vs a fresh summand)
    # stay consistent — otherwise their 1/√count normalizations would not match. p_target starts at
    # the initial heuristic, which every cache has at least, and grows with the adaptive loop.
    brv = caches[1].groups[1].brv
    p_target = _heuristic_p(ℓ_min_max+n_samples, brv, N)
    sketch_rks = brv .* p_target
    # Working sketch: only the active bond's right-neighbour W[j][k+1] is materialized at a time
    # (re-derived from the cache at each bond and after each growth, as the common prefix). W[j][1]
    # (the boundary) is materialized once for the norm estimate.
    W = [Vector{Matrix{T}}(undef, N+1) for j=1:m]
    @timeit timer "reverse_sketch" begin
      for j = 1:m
        W[j][1] = sketch_matrix(caches[j], 1, y[j].ttv_rks[1]; nsamp=[p_target[1]])
      end
    end

    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Boundary sketch of the linear combination (linearity via shared seed)
    y_norm = norm(sum(α[j].*W[j][1] for j=1:m))
    τ = ε * y_norm / sqrt(N - 1)

    @timeit timer "orthogonalization" begin
      Yₖ = [α[j].*reshape(y[j].ttv_vec[1], 1, dims[1], y[j].ttv_rks[2]) for j=1:m]
      @inbounds for k in 1:N-1
        ℓ_min_k = ℓ_min_vec[k+1]
        max_basis = bond_rank_cap(dims, k, ℓ_max_vec[k+1])
        # Materialize the active bond's right-neighbour from the cache at the common prefix
        # p_target[k+1] (it may have grown while earlier bonds were processed). This is the only
        # W[j] bond read at bond k; into a reused capacity buffer (consumers index within sketch_rks).
        @timeit timer "reverse_sketch" for j = 1:m
          remat_into!(W[j], k+1, caches[j], y[j].ttv_rks[k+1], [p_target[k+1]])
        end
        sketch_rks[k+1] = brv[k+1]*p_target[k+1]
        @timeit timer "Randomized adaptive QR decomposition" begin
          @timeit timer "Sketch" begin
            Zₖ = zeros(T, out_rks[k], dims[k], ℓ_min_k)
            for j = 1:m
              Wⱼₖ₊₁ = W[j][k+1][:, 1:ℓ_min_k]
              @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[ρₖ,iₖ,ρₖ₊₁] += Yₖ[j][ρₖ,iₖ,αₖ₊₁]*Wⱼₖ₊₁[αₖ₊₁,ρₖ₊₁]
            end
            Zₖ = reshape(Zₖ, out_rks[k]*dims[k], ℓ_min_k)
          end
          @timeit timer "QR" begin
            Q_factor, _ = qr!(Zₖ)
            Q_init = Matrix(Q_factor)
            # Preallocate Q at max_basis cols once per bond; expand_basis! writes
            # into the next slot in place, avoiding the quadratic hcat realloc.
            # The QR of an (m, ℓ_min) matrix gives a Q with min(m, ℓ_min) cols, so
            # current_cols starts at the actual width — never larger than max_basis.
            current_cols = size(Q_init, 2)
            buf_cols = max(max_basis, current_cols)
            # undef (not zeros): unused cols are written by expand_basis! before
            # they're ever read, so page-zeroing is wasted work.
            Q_storage = Matrix{T}(undef, out_rks[k]*dims[k], buf_cols)
            @views Q_storage[:, 1:current_cols] .= Q_init
            Q = view(Q_storage, :, 1:current_cols)
            ℓ = ℓ_min_k
          end

          # S_full holds only the active window: cols ℓ+1..ℓ+n_samples of Σⱼ Yₖ[j]·W[j][k+1].
          @timeit timer "Initial residual sketch" begin
            S_full = zeros(T, out_rks[k]*dims[k], n_samples)
            let S_full_3d = reshape(S_full, out_rks[k], dims[k], n_samples)
              for j = 1:m
                Wⱼ_view = view(W[j][k+1], :, ℓ_min_k+1:ℓ_min_k+n_samples)
                @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) S_full_3d[ρₖ,iₖ,ρₖ₊₁] += Yₖ[j][ρₖ,iₖ,αₖ₊₁]*Wⱼ_view[αₖ₊₁,ρₖ₊₁]
              end
            end
          end
          Sₖ = Matrix{T}(undef, out_rks[k]*dims[k], n_samples)
          @timeit timer "Residual sketch" begin
            copyto!(Sₖ, S_full)
            mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
          end

          @timeit timer "Adaptive basis expansion" begin
            while norm(Sₖ) > τ * sqrt(n_samples/sketch_rks[k+1]) && current_cols < max_basis
              @timeit timer "Add new orthonormal directions" begin
                # Cap the fed slice at the bond's remaining room: never hand the
                # factorization more columns than max_basis - current_cols. This
                # keeps the residual full-rank in the common case (so no junk
                # completion columns) and makes max_add unnecessary downstream.
                ℓ_inc_eff = clamp(max(ℓ_inc, ceil(Int, 0.2 * current_cols)), 1, min(n_samples, max_basis - current_cols))
                rank_n = expand_basis!(Q_storage, current_cols, view(Sₖ, :, 1:ℓ_inc_eff))
                if rank_n == 0
                  break
                end
                current_cols += rank_n
                Q = view(Q_storage, :, 1:current_cols)
                ℓ += ℓ_inc_eff
              end
              @timeit timer "S_full shift" begin
                # In-place memmove-style shift (copyto! on overlapping views
                # falls back to a slow safe path; linear-index unsafe_copyto!
                # routes to memmove which handles overlap natively and fast).
                m_rows = size(S_full, 1)
                n_keep = n_samples - ℓ_inc_eff
                GC.@preserve S_full Base.unsafe_copyto!(S_full, 1, S_full, m_rows*ℓ_inc_eff+1, m_rows*n_keep)
              end
              @timeit timer "Recursive sketch" begin
                if sketch_rks[k+1] < ℓ+n_samples
                  s_prev_kp1 = sketch_rks[k+1]
                  # Grow the COMMON sample target so bonds k+1:N have ≥ ℓ+n_samples columns, then
                  # ensure every cache reaches it (only missing samples computed; caches already
                  # larger keep their extra, used only as a prefix). The criterion is
                  # total-independent, so any sufficient growth gives the same decisions.
                  for l = k+1:N
                    p_target[l] = max(p_target[l], cld(ℓ+n_samples, brv[l]))
                  end
                  for l = 2:N
                    p_target[l] = max(p_target[l], p_target[l-1])   # keep non-decreasing
                  end
                  want = brv .* p_target
                  for j = 1:m
                    ensure_columns!(caches[j], y[j], want; seed=seed, orthogonal=orthogonal, timer=timer)
                    remat_into!(W[j], k+1, caches[j], y[j].ttv_rks[k+1], [p_target[k+1]])
                  end
                  for l = k+1:N
                    sketch_rks[l] = brv[l]*p_target[l]
                  end
                  # Rescale S_full's kept columns to the new 1/√count normalization so they are
                  # consistent with the freshly materialized (new-normalization) tail columns below.
                  S_full[:, 1:n_samples-ℓ_inc_eff] .*= sqrt(s_prev_kp1 / sketch_rks[k+1])
                end
              end
              @timeit timer "S_full tail contraction" begin
                tail_buf = zeros(T, out_rks[k], dims[k], ℓ_inc_eff)
                for j = 1:m
                  Wⱼ_tail = view(W[j][k+1], :, ℓ+n_samples-ℓ_inc_eff+1:ℓ+n_samples)
                  @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) tail_buf[ρₖ,iₖ,ρₖ₊₁] += Yₖ[j][ρₖ,iₖ,αₖ₊₁]*Wⱼ_tail[αₖ₊₁,ρₖ₊₁]
                end
                copyto!(view(S_full, :, n_samples-ℓ_inc_eff+1:n_samples),
                        reshape(tail_buf, out_rks[k]*dims[k], ℓ_inc_eff))
              end
              @timeit timer "Residual sketch" begin
                copyto!(Sₖ, S_full)
                mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
              end
            end
          end

          @timeit timer "Update core" begin
            out_rks[k+1] = size(Q, 2)
            vec[k] = reshape(Q, out_rks[k], dims[k], out_rks[k+1])
            ot[k] = 1
          end
        end
        @timeit timer "Update Yₖ" begin
          # Explicit (vec[k]·Yₖ[j])→T1, then T1·ttv_vec[k+1] ordering — avoids the large
          # (ρₖ,iₖ,iₖ₊₁,αₖ₊₂) intermediate the naive @tensoropt materialises (the single-TT-path fix).
          Yₖ₊₁ = Vector{Array{T,3}}(undef, m)
          Vk = reshape(vec[k], out_rks[k]*dims[k], out_rks[k+1])
          for j = 1:m
            T1 = Vk' * reshape(Yₖ[j], out_rks[k]*dims[k], y[j].ttv_rks[k+1])
            Yⱼ = T1 * reshape(y[j].ttv_vec[k+1], y[j].ttv_rks[k+1], dims[k+1]*y[j].ttv_rks[k+2])
            Yₖ₊₁[j] = reshape(Yⱼ, out_rks[k+1], dims[k+1], y[j].ttv_rks[k+2])
          end
          Yₖ = Yₖ₊₁
        end
      end
      vec[N] = sum(Yₖ[j] for j=1:m)
    end
    return TTvector{T,N}(N, vec, dims, out_rks, ot)
  end
end


"""
    ttrand_rounding_adaptive(α::Vector{T}, A::TToperator{T,N}, y::Vector{TTvector{T,N}}, ε::Real;
                              ℓ_min=4, ℓ_inc=4, n_samples=…, ℓ_max=…,
                              orthogonal=true, block_rks=N, seed=1234,
                              caches=nothing, weighting=:equal,
                              timer=TimerOutput()) -> TTvector{T,N}

Adaptive randomized rounding of the mixed combination `α[1]·(A·y[1]) + ∑_{j≥2} α[j]·y[j]` to a
relative Frobenius tolerance `ε`, sketching the operator product `A·y[1]` **implicitly** (never
formed). This is the tolerance-based analog of `ttrand_rounding(α, A, y, rks)`.

The operator term (`A·y[1]`) and each vector term `y[j]` are sketched with the **same seed** (so
the boundary sketch `α[1]·W_{Ay}[1] + ∑ α[j]·W[j][1]` is an unbiased estimate of the target norm),
then a single adaptive left-to-right sweep grows each bond's basis until the sketched residual
falls below the per-bond budget `τ = ε·‖target‖_F/√(N−1)`.

`caches` (optional) is a length-`m` vector of reusable per-term caches reused/extended in place across
calls: `caches[1]` an `OperatorCachedSketch` for `A·y[1]`, `caches[j≥2]` a `CachedSketch` for `y[j]`;
a `nothing` entry (or `caches=nothing`) builds a throwaway for that term. All must share `seed`/`block_rks`.
"""
function ttrand_rounding_adaptive(α::Vector{T}, A::TToperator{T,N}, y::Vector{TTvector{T,N}}, ε::Real;
                                   ℓ_min::Union{Int,AbstractVector{Int}}=4,
                                   ℓ_inc::Int=4,
                                   n_samples::Int=max(N÷2, ℓ_inc),
                                   ℓ_max::Union{Int,AbstractVector{Int}}=maximum(A.tto_rks .* y[1].ttv_rks) + sum((maximum(y[j].ttv_rks) for j=2:length(y)); init=0),
                                   orthogonal::Bool=true,
                                   block_rks::Int=N,
                                   seed::Int=1234,
                                   caches=nothing,
                                   weighting::Symbol=:column,
                                   timer::TimerOutput=TimerOutput()) where {T,N}
  @assert n_samples >= ℓ_inc "n_samples ($n_samples) must be ≥ ℓ_inc ($ℓ_inc)"
  @timeit timer "ttrand_rounding_adaptive" begin
    m = length(α)
    @assert length(y) == m
    dims = y[1].ttv_dims
    @assert all(y[j].ttv_dims == dims for j=2:m)
    ℓ_min_vec = _bond_vec(ℓ_min, N); ℓ_max_vec = _bond_vec(ℓ_max, N); ℓ_min_max = maximum(ℓ_min_vec)

    vec = Vector{Array{T,3}}(undef, N)
    # Term 1 → operator cache of A·y[1] (3-D columns); terms j≥2 → vector caches (2-D columns).
    # Uniform block_rks (single group each), shared seed → blocks coincide so the combination is
    # linear in α and per-bond column counts stay aligned. Caller caches reused, else throwaways.
    _fresh_op()    = cached_operator_sketch(T, A, y[1], block_rks, block_rks, ℓ_min_max+n_samples; seed=seed, orthogonal=orthogonal, timer=timer)
    _fresh_vec(j)  = cached_sketch(T, y[j], block_rks, block_rks, ℓ_min_max+n_samples; seed=seed, orthogonal=orthogonal, timer=timer)
    op_cache = (caches === nothing || caches[1] === nothing) ? _fresh_op() : caches[1]
    vcaches  = Vector{Any}(undef, m)
    for j = 2:m
      vcaches[j] = (caches === nothing || caches[j] === nothing) ? _fresh_vec(j) : caches[j]
    end
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)
    _scols(l) = sum(g.brv[l]*g.counts[l] for g in op_cache.groups)
    sketch_rks = [_scols(l) for l=1:N+1]
    # Only the active bond's right-neighbour per term is materialized at a time (re-derived from the
    # caches each bond and after each growth, into reused buffers); the boundary once for the norm.
    WAy = Vector{Array{T,3}}(undef, N+1)
    W = Vector{Vector{Matrix{T}}}(undef, m)
    for j = 2:m
      W[j] = Vector{Matrix{T}}(undef, N+1)
    end
    @timeit timer "reverse_sketch" begin
      WAy[1] = sketch_array(op_cache, 1, y[1].ttv_rks[1], A.tto_rks[1]; weighting=weighting)
      for j = 2:m
        W[j][1] = sketch_matrix(vcaches[j], 1, y[j].ttv_rks[1]; weighting=weighting)
      end
    end

    # Boundary sketch of the whole combination (linearity via shared blocks)
    bnd = α[1] .* Base.vec(WAy[1])
    for j = 2:m
      bnd = bnd .+ α[j] .* Base.vec(W[j][1])
    end
    y_norm = norm(bnd)
    τ = ε * y_norm / sqrt(N - 1)

    @timeit timer "orthogonalization" begin
      # Term 1 partial product: Ayₖ layout (L=1, I, R_y, R_A), scaled by α[1].
      Ayₖ = zeros(T, 1, dims[1], y[1].ttv_rks[2], A.tto_rks[2])
      let Ayₖ_3d = reshape(Ayₖ, dims[1], y[1].ttv_rks[2], A.tto_rks[2]),
          y1     = reshape(y[1].ttv_vec[1], dims[1], y[1].ttv_rks[2]),
          A1     = reshape(A.tto_vec[1], dims[1], dims[1], A.tto_rks[2])
        @tensor Ayₖ_3d[iₖ,αₖ₊₁,βₖ₊₁] = A1[iₖ,jₖ,βₖ₊₁] * y1[jₖ,αₖ₊₁]
      end
      Ayₖ .*= α[1]
      # Terms j≥2 partial products, scaled by α[j].
      Yₖ = Vector{Array{T,3}}(undef, m)
      for j = 2:m
        Yₖ[j] = α[j].*reshape(y[j].ttv_vec[1], 1, dims[1], y[j].ttv_rks[2])
      end

      @inbounds for k in 1:N-1
        ℓ_min_k = ℓ_min_vec[k+1]
        max_basis = bond_rank_cap(dims, k, ℓ_max_vec[k+1])
        # Materialize the active bond's right-neighbour per term from the caches (may have grown).
        @timeit timer "reverse_sketch" begin
          remat_into_operator!(WAy, k+1, op_cache, y[1].ttv_rks[k+1], A.tto_rks[k+1]; weighting=weighting)
          for j = 2:m
            remat_into!(W[j], k+1, vcaches[j], y[j].ttv_rks[k+1]; weighting=weighting)
          end
          sketch_rks[k+1] = _scols(k+1)
        end
        @timeit timer "Randomized adaptive QR decomposition" begin
          @timeit timer "Sketch" begin
            Zₖ = zeros(T, out_rks[k], dims[k], ℓ_min_k)
            WAyₖ₊₁ = WAy[k+1][:, :, 1:ℓ_min_k]
            @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[ρₖ,iₖ,ρₖ₊₁] += Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*WAyₖ₊₁[αₖ₊₁,βₖ₊₁,ρₖ₊₁]
            for j = 2:m
              Wⱼₖ₊₁ = W[j][k+1][:, 1:ℓ_min_k]
              @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[ρₖ,iₖ,ρₖ₊₁] += Yₖ[j][ρₖ,iₖ,αₖ₊₁]*Wⱼₖ₊₁[αₖ₊₁,ρₖ₊₁]
            end
            Zₖ = reshape(Zₖ, out_rks[k]*dims[k], ℓ_min_k)
          end
          @timeit timer "QR" begin
            Q_factor, _ = qr!(Zₖ)
            Q_init = Matrix(Q_factor)
            current_cols = size(Q_init, 2)
            buf_cols = max(max_basis, current_cols)
            Q_storage = Matrix{T}(undef, out_rks[k]*dims[k], buf_cols)
            @views Q_storage[:, 1:current_cols] .= Q_init
            Q = view(Q_storage, :, 1:current_cols)
            ℓ = ℓ_min_k
          end

          @timeit timer "Initial residual sketch" begin
            S_full = zeros(T, out_rks[k]*dims[k], n_samples)
            let S_full_3d = reshape(S_full, out_rks[k], dims[k], n_samples)
              WAy_view = view(WAy[k+1], :, :, ℓ_min_k+1:ℓ_min_k+n_samples)
              @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁) S_full_3d[ρₖ,iₖ,ρₖ₊₁] += Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*WAy_view[αₖ₊₁,βₖ₊₁,ρₖ₊₁]
              for j = 2:m
                Wⱼ_view = view(W[j][k+1], :, ℓ_min_k+1:ℓ_min_k+n_samples)
                @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) S_full_3d[ρₖ,iₖ,ρₖ₊₁] += Yₖ[j][ρₖ,iₖ,αₖ₊₁]*Wⱼ_view[αₖ₊₁,ρₖ₊₁]
              end
            end
          end
          Sₖ = Matrix{T}(undef, out_rks[k]*dims[k], n_samples)
          @timeit timer "Residual sketch" begin
            copyto!(Sₖ, S_full)
            mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
          end

          @timeit timer "Adaptive basis expansion" begin
            while norm(Sₖ) > τ * sqrt(n_samples/sketch_rks[k+1]) && current_cols < max_basis
              @timeit timer "Add new orthonormal directions" begin
                ℓ_inc_eff = clamp(max(ℓ_inc, ceil(Int, 0.2 * current_cols)), 1, min(n_samples, max_basis - current_cols))
                rank_n = expand_basis!(Q_storage, current_cols, view(Sₖ, :, 1:ℓ_inc_eff))
                if rank_n == 0
                  break
                end
                current_cols += rank_n
                Q = view(Q_storage, :, 1:current_cols)
                ℓ += ℓ_inc_eff
              end
              @timeit timer "S_full shift" begin
                m_rows = size(S_full, 1)
                n_keep = n_samples - ℓ_inc_eff
                GC.@preserve S_full Base.unsafe_copyto!(S_full, 1, S_full, m_rows*ℓ_inc_eff+1, m_rows*n_keep)
              end
              @timeit timer "Recursive sketch" begin
                if sketch_rks[k+1] < ℓ+n_samples
                  s_prev_kp1 = sketch_rks[k+1]
                  # Grow every term's cache so bonds k+1:N reach ≥ ℓ+n_samples total columns; same seed
                  # and brv keep their blocks identical (linearity) and column counts aligned.
                  want = copy(sketch_rks)
                  for l = k+1:N
                    want[l] = max(want[l], ℓ+n_samples)
                  end
                  ensure_columns!(op_cache, A, y[1], want; seed=seed, orthogonal=orthogonal, timer=timer)
                  remat_into_operator!(WAy, k+1, op_cache, y[1].ttv_rks[k+1], A.tto_rks[k+1]; weighting=weighting)
                  for j = 2:m
                    ensure_columns!(vcaches[j], y[j], want; seed=seed, orthogonal=orthogonal, timer=timer)
                    remat_into!(W[j], k+1, vcaches[j], y[j].ttv_rks[k+1]; weighting=weighting)
                  end
                  for l = k+1:N
                    sketch_rks[l] = _scols(l)
                  end
                  S_full[:, 1:n_samples-ℓ_inc_eff] .*= sqrt(s_prev_kp1 / sketch_rks[k+1])
                end
              end
              @timeit timer "S_full tail contraction" begin
                tail_buf = zeros(T, out_rks[k], dims[k], ℓ_inc_eff)
                WAy_tail = view(WAy[k+1], :, :, ℓ+n_samples-ℓ_inc_eff+1:ℓ+n_samples)
                @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁) tail_buf[ρₖ,iₖ,ρₖ₊₁] += Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*WAy_tail[αₖ₊₁,βₖ₊₁,ρₖ₊₁]
                for j = 2:m
                  Wⱼ_tail = view(W[j][k+1], :, ℓ+n_samples-ℓ_inc_eff+1:ℓ+n_samples)
                  @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) tail_buf[ρₖ,iₖ,ρₖ₊₁] += Yₖ[j][ρₖ,iₖ,αₖ₊₁]*Wⱼ_tail[αₖ₊₁,ρₖ₊₁]
                end
                copyto!(view(S_full, :, n_samples-ℓ_inc_eff+1:n_samples),
                        reshape(tail_buf, out_rks[k]*dims[k], ℓ_inc_eff))
              end
              @timeit timer "Residual sketch" begin
                copyto!(Sₖ, S_full)
                mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
              end
            end
          end

          @timeit timer "Update core" begin
            out_rks[k+1] = size(Q, 2)
            vec[k] = reshape(Q, out_rks[k], dims[k], out_rks[k+1])
            ot[k] = 1
          end
        end
        @timeit timer "Update partial products" begin
          # Term 1 operator update.
          Ayₖ₊₁ = zeros(T, out_rks[k+1], dims[k+1], y[1].ttv_rks[k+2], A.tto_rks[k+2])
          @tensoropt (αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂,ρₖ₊₁) Ayₖ₊₁[ρₖ₊₁,iₖ₊₁,αₖ₊₂,βₖ₊₂] = Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*vec[k][ρₖ,iₖ,ρₖ₊₁]*y[1].ttv_vec[k+1][αₖ₊₁,jₖ₊₁,αₖ₊₂]*A.tto_vec[k+1][βₖ₊₁,iₖ₊₁,jₖ₊₁,βₖ₊₂]
          Ayₖ = Ayₖ₊₁
          # Terms j≥2 vector updates: explicit (vec[k]·Yₖ[j])→T1, then T1·ttv_vec[k+1] ordering
          # (avoids the (ρₖ,iₖ,iₖ₊₁,αₖ₊₂) intermediate the naive @tensoropt materialises).
          Yₖ₊₁ = Vector{Array{T,3}}(undef, m)
          Vk = reshape(vec[k], out_rks[k]*dims[k], out_rks[k+1])
          for j = 2:m
            T1 = Vk' * reshape(Yₖ[j], out_rks[k]*dims[k], y[j].ttv_rks[k+1])
            Yⱼ = T1 * reshape(y[j].ttv_vec[k+1], y[j].ttv_rks[k+1], dims[k+1]*y[j].ttv_rks[k+2])
            Yₖ₊₁[j] = reshape(Yⱼ, out_rks[k+1], dims[k+1], y[j].ttv_rks[k+2])
          end
          Yₖ = Yₖ₊₁
        end
      end
      # Last core: operator term + vector terms.
      vec[N] = reshape(Ayₖ, out_rks[N], dims[N], out_rks[N+1])
      for j = 2:m
        vec[N] = vec[N] .+ reshape(Yₖ[j], out_rks[N], dims[N], out_rks[N+1])
      end
    end
    return TTvector{T,N}(N, vec, dims, out_rks, ot)
  end
end


"""
    ttrand_rounding_adaptive(Atto::TToperator{T,N}, y::TTvector{T,N}, b::TTvector{T,N}, ε::Real;
                              ℓ_min=4, ℓ_inc=4, ℓ_max=…,
                              orthogonal=true, block_rks=N,
                              seed=1234, timer=TimerOutput()) -> TTvector{T,N}

Adaptive randomized rounding of `Atto·y - b` without explicitly forming the product.

Both sketches share the seed so `WAy[1] - Wb[1]` is itself the boundary sketch of the
target `Atto·y - b`, giving an unbiased norm estimate `‖WAy[1] - Wb[1]‖ ≈ ‖Atto·y - b‖`.
"""
function ttrand_rounding_adaptive(Atto::TToperator{T,N}, y::TTvector{T,N}, b::TTvector{T,N}, ε::Real;
                                   ℓ_min::Union{Int,AbstractVector{Int}}=4,
                                   ℓ_inc::Int=4,
                                   n_samples::Int=max(N÷2, ℓ_inc),
                                   ℓ_max::Union{Int,AbstractVector{Int}}=maximum(Atto.tto_rks .* y.ttv_rks) + maximum(b.ttv_rks),
                                   orthogonal::Bool=true,
                                   block_rks::Int=N,
                                   seed::Int=1234,
                                   caches=nothing,
                                   weighting::Symbol=:column,
                                   timer::TimerOutput=TimerOutput()) where {T,N}
  @assert n_samples >= ℓ_inc "n_samples ($n_samples) must be ≥ ℓ_inc ($ℓ_inc)"
  @timeit timer "ttrand_rounding_adaptive" begin
    dims = y.ttv_dims
    ℓ_min_vec = _bond_vec(ℓ_min, N); ℓ_max_vec = _bond_vec(ℓ_max, N); ℓ_min_max = maximum(ℓ_min_vec)
    vec = Vector{Array{T,3}}(undef, N)

    # Two reusable caches sharing seed/block_rks so their blocks coincide and S(A·y) − S(b) is the
    # sketch of the residual A·y − b: an operator cache for A·y (3-D columns) and a vector cache for
    # b (2-D columns). Uniform block_rks here (single group each); `caches=(op_cache, b_cache)` reuses
    # caller caches, else throwaways are built. Columns stay aligned because both grow to the same
    # per-bond counts (identical brv).
    if caches === nothing
      op_cache = cached_operator_sketch(T, Atto, y, block_rks, block_rks, ℓ_min_max+n_samples; seed=seed, orthogonal=orthogonal, timer=timer)
      b_cache  = cached_sketch(T, b, block_rks, block_rks, ℓ_min_max+n_samples; seed=seed, orthogonal=orthogonal, timer=timer)
    else
      op_cache, b_cache = caches
    end
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)
    _scols(l) = sum(g.brv[l]*g.counts[l] for g in op_cache.groups)
    sketch_rks = [_scols(l) for l=1:N+1]
    # Only the active bond's right-neighbour is materialized at a time (re-derived from the caches
    # each bond and after each growth, into reused buffers); the boundary once for the norm.
    WAy = Vector{Array{T,3}}(undef, N+1)
    Wb  = Vector{Matrix{T}}(undef, N+1)
    @timeit timer "reverse_sketch" begin
      WAy[1] = sketch_array(op_cache, 1, y.ttv_rks[1], Atto.tto_rks[1]; weighting=weighting)
      Wb[1]  = sketch_matrix(b_cache, 1, b.ttv_rks[1]; weighting=weighting)
    end

    # Boundary sketches share blocks → linearity holds for the difference
    y_norm = norm(Base.vec(WAy[1]) .- Base.vec(Wb[1]))
    τ = ε * y_norm / sqrt(N - 1)

    @timeit timer "orthogonalization" begin
      # Boundary partial-product Ayₖ has layout (L=1, I, R_y, R_A).
      Ayₖ = zeros(T, 1, dims[1], y.ttv_rks[2], Atto.tto_rks[2])
      let Ayₖ_3d = reshape(Ayₖ, dims[1], y.ttv_rks[2], Atto.tto_rks[2]),
          y1     = reshape(y.ttv_vec[1], dims[1], y.ttv_rks[2]),
          A1     = reshape(Atto.tto_vec[1], dims[1], dims[1], Atto.tto_rks[2])
        @tensor Ayₖ_3d[iₖ,αₖ₊₁,βₖ₊₁] = A1[iₖ,jₖ,βₖ₊₁] * y1[jₖ,αₖ₊₁]
      end
      bₖ = reshape(b.ttv_vec[1], 1, dims[1], b.ttv_rks[2])

      @inbounds for k in 1:N-1
        ℓ_min_k = ℓ_min_vec[k+1]
        max_basis = bond_rank_cap(dims, k, ℓ_max_vec[k+1])
        # Materialize the active bond's right-neighbour from the caches (they may have grown earlier).
        @timeit timer "reverse_sketch" begin
          remat_into_operator!(WAy, k+1, op_cache, y.ttv_rks[k+1], Atto.tto_rks[k+1]; weighting=weighting)
          remat_into!(Wb, k+1, b_cache, b.ttv_rks[k+1]; weighting=weighting)
          sketch_rks[k+1] = _scols(k+1)
        end
        @timeit timer "Randomized adaptive QR decomposition" begin
          @timeit timer "Sketch" begin
            Zₖ = zeros(T, out_rks[k], dims[k], ℓ_min_k)
            WAyₖ₊₁ = WAy[k+1][:, :, 1:ℓ_min_k]
            Wbₖ₊₁  = Wb[k+1][:, 1:ℓ_min_k]
            @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[ρₖ,iₖ,ρₖ₊₁] = Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*WAyₖ₊₁[αₖ₊₁,βₖ₊₁,ρₖ₊₁] - bₖ[ρₖ,iₖ,αₖ₊₁]*Wbₖ₊₁[αₖ₊₁,ρₖ₊₁]
            Zₖ = reshape(Zₖ, out_rks[k]*dims[k], ℓ_min_k)
          end
          @timeit timer "QR" begin
            Q_factor, _ = qr!(Zₖ)
            Q_init = Matrix(Q_factor)
            # Preallocate Q at max_basis cols once per bond; expand_basis! writes
            # into the next slot in place, avoiding the quadratic hcat realloc.
            # The QR of an (m, ℓ_min) matrix gives a Q with min(m, ℓ_min) cols, so
            # current_cols starts at the actual width — never larger than max_basis.
            current_cols = size(Q_init, 2)
            buf_cols = max(max_basis, current_cols)
            # undef (not zeros): unused cols are written by expand_basis! before
            # they're ever read, so page-zeroing is wasted work.
            Q_storage = Matrix{T}(undef, out_rks[k]*dims[k], buf_cols)
            @views Q_storage[:, 1:current_cols] .= Q_init
            Q = view(Q_storage, :, 1:current_cols)
            ℓ = ℓ_min_k
          end

          # S_full holds only the active window: cols ℓ+1..ℓ+n_samples.
          @timeit timer "Initial residual sketch" begin
            S_full = zeros(T, out_rks[k]*dims[k], n_samples)
            let S_full_3d = reshape(S_full, out_rks[k], dims[k], n_samples),
                WAy_view = view(WAy[k+1], :, :, ℓ_min_k+1:ℓ_min_k+n_samples),
                Wb_view  = view(Wb[k+1],  :,    ℓ_min_k+1:ℓ_min_k+n_samples)
              @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁) S_full_3d[ρₖ,iₖ,ρₖ₊₁] = Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*WAy_view[αₖ₊₁,βₖ₊₁,ρₖ₊₁] - bₖ[ρₖ,iₖ,αₖ₊₁]*Wb_view[αₖ₊₁,ρₖ₊₁]
            end
          end
          Sₖ = Matrix{T}(undef, out_rks[k]*dims[k], n_samples)
          @timeit timer "Residual sketch" begin
            copyto!(Sₖ, S_full)
            mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
          end

          @timeit timer "Adaptive basis expansion" begin
            while norm(Sₖ) > τ * sqrt(n_samples/sketch_rks[k+1]) && current_cols < max_basis
              @timeit timer "Add new orthonormal directions" begin
                # Cap the fed slice at the bond's remaining room: never hand the
                # factorization more columns than max_basis - current_cols. This
                # keeps the residual full-rank in the common case (so no junk
                # completion columns) and makes max_add unnecessary downstream.
                ℓ_inc_eff = clamp(max(ℓ_inc, ceil(Int, 0.2 * current_cols)), 1, min(n_samples, max_basis - current_cols))
                rank_n = expand_basis!(Q_storage, current_cols, view(Sₖ, :, 1:ℓ_inc_eff))
                if rank_n == 0
                  break
                end
                current_cols += rank_n
                Q = view(Q_storage, :, 1:current_cols)
                ℓ += ℓ_inc_eff
              end
              @timeit timer "S_full shift" begin
                # In-place memmove-style shift (copyto! on overlapping views
                # falls back to a slow safe path; linear-index unsafe_copyto!
                # routes to memmove which handles overlap natively and fast).
                m_rows = size(S_full, 1)
                n_keep = n_samples - ℓ_inc_eff
                GC.@preserve S_full Base.unsafe_copyto!(S_full, 1, S_full, m_rows*ℓ_inc_eff+1, m_rows*n_keep)
              end
              @timeit timer "Recursive sketch" begin
                if sketch_rks[k+1] < ℓ+n_samples
                  s_prev_kp1 = sketch_rks[k+1]
                  # Grow both caches so bonds k+1:N reach ≥ ℓ+n_samples total columns; same seed and
                  # brv keep their blocks identical (so S(Ay)−S(b) stays a sketch of the residual) and
                  # their per-bond column counts aligned. Re-materialize the active bond.
                  want = copy(sketch_rks)
                  for l = k+1:N
                    want[l] = max(want[l], ℓ+n_samples)
                  end
                  ensure_columns!(op_cache, Atto, y, want; seed=seed, orthogonal=orthogonal, timer=timer)
                  ensure_columns!(b_cache, b, want; seed=seed, orthogonal=orthogonal, timer=timer)
                  remat_into_operator!(WAy, k+1, op_cache, y.ttv_rks[k+1], Atto.tto_rks[k+1]; weighting=weighting)
                  remat_into!(Wb, k+1, b_cache, b.ttv_rks[k+1]; weighting=weighting)
                  for l = k+1:N
                    sketch_rks[l] = _scols(l)
                  end
                  # Renormalise the kept S_full cols (still in the OLD normalization).
                  S_full[:, 1:n_samples-ℓ_inc_eff] .*= sqrt(s_prev_kp1 / sketch_rks[k+1])
                end
              end
              @timeit timer "S_full tail contraction" begin
                tail_buf = zeros(T, out_rks[k], dims[k], ℓ_inc_eff)
                WAy_tail = view(WAy[k+1], :, :, ℓ+n_samples-ℓ_inc_eff+1:ℓ+n_samples)
                Wb_tail  = view(Wb[k+1],  :,    ℓ+n_samples-ℓ_inc_eff+1:ℓ+n_samples)
                @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁) tail_buf[ρₖ,iₖ,ρₖ₊₁] = Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*WAy_tail[αₖ₊₁,βₖ₊₁,ρₖ₊₁] - bₖ[ρₖ,iₖ,αₖ₊₁]*Wb_tail[αₖ₊₁,ρₖ₊₁]
                copyto!(view(S_full, :, n_samples-ℓ_inc_eff+1:n_samples),
                        reshape(tail_buf, out_rks[k]*dims[k], ℓ_inc_eff))
              end
              @timeit timer "Residual sketch" begin
                copyto!(Sₖ, S_full)
                mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
              end
            end
          end

          @timeit timer "Update core" begin
            out_rks[k+1] = size(Q, 2)
            vec[k] = reshape(Q, out_rks[k], dims[k], out_rks[k+1])
            ot[k] = 1
          end
        end
        @timeit timer "Update Ayₖ, bₖ" begin
          Ayₖ₊₁ = zeros(T, out_rks[k+1], dims[k+1], y.ttv_rks[k+2], Atto.tto_rks[k+2])
          @tensoropt (αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂,ρₖ₊₁) Ayₖ₊₁[ρₖ₊₁,iₖ₊₁,αₖ₊₂,βₖ₊₂] = Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*vec[k][ρₖ,iₖ,ρₖ₊₁]*y.ttv_vec[k+1][αₖ₊₁,jₖ₊₁,αₖ₊₂]*Atto.tto_vec[k+1][βₖ₊₁,iₖ₊₁,jₖ₊₁,βₖ₊₂]
          # b is a plain vector term: explicit (vec[k]·bₖ)→T1, then T1·ttv_vec[k+1] (avoids the
          # (ρₖ,iₖ,iₖ₊₁,αₖ₊₂) intermediate the naive @tensoropt materialises).
          Vk = reshape(vec[k], out_rks[k]*dims[k], out_rks[k+1])
          T1b = Vk' * reshape(bₖ, out_rks[k]*dims[k], b.ttv_rks[k+1])
          bₖ₊₁ = reshape(T1b * reshape(b.ttv_vec[k+1], b.ttv_rks[k+1], dims[k+1]*b.ttv_rks[k+2]),
                         out_rks[k+1], dims[k+1], b.ttv_rks[k+2])
          Ayₖ = Ayₖ₊₁
          bₖ  = bₖ₊₁
        end
      end
      vec[N] = reshape(Ayₖ, out_rks[N], dims[N], out_rks[N+1]) .- reshape(bₖ, out_rks[N], dims[N], out_rks[N+1])
    end
    return TTvector{T,N}(N, vec, dims, out_rks, ot)
  end
end


"""
    ttrand_rounding_adaptive(y::NTuple{M,TTvector{T,N}}, ε::Real;
                              ℓ_min=4, ℓ_inc=4, ℓ_max=…,
                              orthogonal=true, block_rks=N, block_rks_inc=N÷4,
                              seed=1234, timer=TimerOutput()) -> TTvector{T,N}

Adaptive randomized rounding of the Hadamard (element-wise) product
`y[1] ⊙ y[2] ⊙ … ⊙ y[M]`. The sketch is a single recursive (TTStack)
sketch over the tuple; the accumulator contraction with the M factor
cores follows the pairwise scheme of the existing `ttrand_rounding(y::NTuple, …)`.

# Three independent knobs (easy to confuse)

- `ℓ_inc::Int` — additive floor on the per-iteration column growth of
  the orthogonal basis `Q`. The effective Q-growth per inner-loop step is
  `ℓ_inc_eff = max(ℓ_inc, ⌈0.2·size(Q,2)⌉)` (capped at `n_samples`), i.e.
  geometric `0.2·rk` scaling with `ℓ_inc` as a small additive floor for
  the very first iterations. This is the only knob that controls how
  many cols are appended to `Q` per inner step.
- `block_rks::Int` — block rank of the *initial* TTStack sketch built
  once at the start. Controls the variance of `y_norm = ‖W[1]‖` (per
  Al Daas et al., arXiv:2511.03598, Remark 3.1). Default `N` gives a
  low-variance, oblivious estimate; smaller values trade accuracy for
  cheaper sketch construction.
- `block_rks_inc::Int` — block rank of *extension* TTStack sketches
  built inside the inner loop when the sliding window of `n_samples`
  sketch cols is exhausted and more cols must be appended to `W`.
  Default `N÷4`. **This is not the Q-growth increment**; it only
  controls the structure (and per-block QR cost) of the *extra
  sketch cols* feeding the residual estimator. Setting
  `block_rks_inc=1` makes those extension cols rank-1 Khatri-Rao
  blocks (no QR, noisier residual estimator); large `block_rks_inc`
  matches the quality of the initial sketch but costs more per block
  to build.

All sketches use within-block QR orthogonalisation
(`orthogonal=true`) — without it, the chain product of Gaussian blocks
through `N` sites has heavy-tailed variance even at large `block_rks`
(see `dev_tests/y_norm_vs_err_distribution.jl`).
"""
function ttrand_rounding_adaptive(y::NTuple{M,TTvector{T,N}}, ε::Real;
                                   ℓ_min::Union{Int,AbstractVector{Int}}=4,
                                   ℓ_inc::Int=4,
                                   ℓ_max::Union{Int,AbstractVector{Int}}=prod(maximum(y[i].ttv_rks) for i=1:M),
                                   init_f::Real=0.0,
                                   n_samples::Int=max(N÷2, ℓ_inc, ceil(Int, 0.1 * _scalar_max(ℓ_max))),
                                   orthogonal::Bool=true,
                                   block_rks::Int=N,
                                   block_rks_inc::Int=max(1, N÷4),
                                   seed::Int=1234,
                                   cache=nothing,
                                   weighting::Symbol=:column,
                                   timer::TimerOutput=TimerOutput()) where {T,N,M}
  if M == 1
    return ttrand_rounding_adaptive(y[1], ε; ℓ_min=ℓ_min, ℓ_inc=ℓ_inc, n_samples=n_samples, ℓ_max=ℓ_max, init_f=init_f, orthogonal=orthogonal, block_rks=block_rks, seed=seed, timer=timer)
  end
  @assert n_samples >= ℓ_inc "n_samples ($n_samples) must be ≥ ℓ_inc ($ℓ_inc)"

  @timeit timer "ttrand_rounding_adaptive" begin
    dims = y[1].ttv_dims
    vec  = Vector{Array{T,3}}(undef, N)
    ℓ_min_vec = _bond_vec(ℓ_min, N); ℓ_max_vec = _bond_vec(ℓ_max, N)

    # Per-bond Init_b_k = max(ℓ_min, ceil(init_f · max_cols_k)), where
    # max_cols_k bounds the bond's rank capacity. For the Hadamard product
    # the natural bond rank cap is prod_i y[i].ttv_rks[k] (the Kronecker rank
    # at bond k, indexed as ttv_rks does — k in 1..N+1).
    rks_prod = ntuple(k -> prod(y[i].ttv_rks[k] for i=1:M), N+1)
    function initb_k(k::Int)
        max_cols_k = min(rks_prod[k] * dims[k], rks_prod[k+1])
        init_f > 0 ? max(ℓ_min_vec[k+1], ceil(Int, init_f * max_cols_k)) : ℓ_min_vec[k+1]
    end
    ℓ_min_global = init_f > 0 ? maximum(initb_k(k) for k in 1:N-1) : maximum(ℓ_min_vec)

    # Two-group reusable Kronecker cache (frozen init group block_rks + growable ext group
    # block_rks_inc); throwaway when `cache===nothing`. Groups combine by `weighting` (:column matches
    # the legacy per-column renormalization). Sized to the worst bond's Init_b_k plus n_samples.
    cache === nothing && (cache = cached_kronecker_sketch(T, y, block_rks, block_rks_inc, ℓ_min_global+n_samples; seed=seed, orthogonal=orthogonal, timer=timer))
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)
    _scols(l) = sum(g.brv[l]*g.counts[l] for g in cache.groups)
    sketch_rks = [_scols(l) for l=1:N+1]
    # Only the active bond's right-neighbour is materialized at a time (2-D (∏ᵢ rks_i, cols), into a
    # reused buffer); the boundary once for the norm.
    W = Vector{Matrix{T}}(undef, N+1)
    @timeit timer "reverse_sketch" begin
      W[1] = sketch_matrix(cache, 1, rks_prod[1]; weighting=weighting)
    end

    # Sketch-based ‖y‖ from the boundary contraction of the recursive sketch
    # (Al Daas et al., 2025, Remark 3.1 / eqn 3.7: ‖X‖ ≈ ‖V(X_1)·W_2‖_F / √r). The estimator is
    # unbiased but has variance scaling poorly with the number of modes — see Theorem 3.4. Larger
    # n_samples (default `0.1·ℓ_max`, matching f_init=0.1 from §4.2) keeps this variance manageable.
    y_norm = norm(W[1])
    τ = ε * y_norm / sqrt(N - 1)

    @timeit timer "orthogonalization sweep" begin
      yₖ = broadcast(*, (reshape(y[i].ttv_vec[1],
                                  1, dims[1], ntuple(j->( j==i ? y[i].ttv_rks[2] : 1), M)...)
                         for i=1:M)...)
      yₖ = reshape(yₖ, 1, dims[1], prod(y[i].ttv_rks[2] for i=1:M))

      @inbounds for k in 1:N-1
        max_basis = bond_rank_cap(dims, k, ℓ_max_vec[k+1])
        # Per-bond Init_b_k for the Hadamard case (MATLAB-style).
        ℓ_min_k = init_f > 0 ? max(ℓ_min_vec[k+1], ceil(Int, init_f * min(out_rks[k]*dims[k], rks_prod[k+1]))) : ℓ_min_vec[k+1]
        # Materialize the active bond's right-neighbour from the cache (it may have grown earlier).
        @timeit timer "reverse_sketch" remat_into!(W, k+1, cache, rks_prod[k+1]; weighting=weighting)
        sketch_rks[k+1] = _scols(k+1)
        @timeit timer "Randomized adaptive QR decomposition" begin
          @timeit timer "Sketch" begin
            Zₖ = zeros(T, out_rks[k], dims[k], ℓ_min_k)
            Wₖ₊₁ = W[k+1][:, 1:ℓ_min_k]
            @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*Wₖ₊₁[αₖ₊₁,ρₖ₊₁]
            Zₖ = reshape(Zₖ, out_rks[k]*dims[k], ℓ_min_k)
          end
          @timeit timer "QR" begin
            Q_factor, _ = qr!(Zₖ)
            Q_init = Matrix(Q_factor)
            # Preallocate Q at max_basis cols once per bond; expand_basis! writes
            # into the next slot in place, avoiding the quadratic hcat realloc.
            # The QR of an (m, ℓ_min_k) matrix gives a Q with min(m, ℓ_min_k) cols, so
            # current_cols starts at the actual width — never larger than max_basis.
            current_cols = size(Q_init, 2)
            buf_cols = max(max_basis, current_cols)
            # undef (not zeros): unused cols are written by expand_basis! before
            # they're ever read, so page-zeroing is wasted work.
            Q_storage = Matrix{T}(undef, out_rks[k]*dims[k], buf_cols)
            @views Q_storage[:, 1:current_cols] .= Q_init
            Q = view(Q_storage, :, 1:current_cols)
            ℓ = ℓ_min_k
          end

          # S_full holds only the active window of yₖ × W[k+1].
          @timeit timer "Initial residual sketch" begin
            S_full = zeros(T, out_rks[k]*dims[k], n_samples)
            let S_full_3d = reshape(S_full, out_rks[k], dims[k], n_samples),
                W_view   = view(W[k+1], :, ℓ_min_k+1:ℓ_min_k+n_samples)
              @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) S_full_3d[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*W_view[αₖ₊₁,ρₖ₊₁]
            end
          end
          Sₖ = Matrix{T}(undef, out_rks[k]*dims[k], n_samples)
          @timeit timer "Residual sketch" begin
            copyto!(Sₖ, S_full)
            mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
          end

          @timeit timer "Adaptive basis expansion" begin
            while norm(Sₖ) > τ * sqrt(n_samples/sketch_rks[k+1]) && current_cols < max_basis
              @timeit timer "Add new orthonormal directions" begin
                # Cap the fed slice at the bond's remaining room: never hand the
                # factorization more columns than max_basis - current_cols. This
                # keeps the residual full-rank in the common case (so no junk
                # completion columns) and makes max_add unnecessary downstream.
                ℓ_inc_eff = clamp(max(ℓ_inc, ceil(Int, 0.2 * current_cols)), 1, min(n_samples, max_basis - current_cols))
                rank_n = expand_basis!(Q_storage, current_cols, view(Sₖ, :, 1:ℓ_inc_eff))
                if rank_n == 0
                  break
                end
                current_cols += rank_n
                Q = view(Q_storage, :, 1:current_cols)
                ℓ += ℓ_inc_eff
              end
              @timeit timer "S_full shift" begin
                # In-place memmove-style shift (copyto! on overlapping views
                # falls back to a slow safe path; linear-index unsafe_copyto!
                # routes to memmove which handles overlap natively and fast).
                m_rows = size(S_full, 1)
                n_keep = n_samples - ℓ_inc_eff
                GC.@preserve S_full Base.unsafe_copyto!(S_full, 1, S_full, m_rows*ℓ_inc_eff+1, m_rows*n_keep)
              end
              @timeit timer "Recursive sketch" begin
                if sketch_rks[k+1] < ℓ+n_samples
                  s_prev_kp1 = sketch_rks[k+1]
                  # Grow the extension group (block_rks_inc) so bonds k+1:N reach ≥ ℓ+n_samples total
                  # columns; the frozen init group is untouched. Re-materialize the active bond.
                  want = copy(sketch_rks)
                  for l = k+1:N
                    want[l] = max(want[l], ℓ+n_samples)
                  end
                  ensure_columns!(cache, y, want; seed=seed, orthogonal=orthogonal, timer=timer)
                  remat_into!(W, k+1, cache, rks_prod[k+1]; weighting=weighting)
                  for l = k+1:N
                    sketch_rks[l] = _scols(l)
                  end
                  # Renormalise the kept S_full cols (still in the OLD normalization).
                  S_full[:, 1:n_samples-ℓ_inc_eff] .*= sqrt(s_prev_kp1 / sketch_rks[k+1])
                end
              end
              @timeit timer "S_full tail contraction" begin
                tail_buf = zeros(T, out_rks[k], dims[k], ℓ_inc_eff)
                W_tail = view(W[k+1], :, ℓ+n_samples-ℓ_inc_eff+1:ℓ+n_samples)
                @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) tail_buf[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*W_tail[αₖ₊₁,ρₖ₊₁]
                copyto!(view(S_full, :, n_samples-ℓ_inc_eff+1:n_samples),
                        reshape(tail_buf, out_rks[k]*dims[k], ℓ_inc_eff))
              end
              @timeit timer "Residual sketch" begin
                copyto!(Sₖ, S_full)
                mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
              end
            end
          end

          @timeit timer "Update core" begin
            out_rks[k+1] = size(Q, 2)
            vec[k] = reshape(Q, out_rks[k], dims[k], out_rks[k+1])
            ot[k] = 1
          end
        end

        @timeit timer "Update Qy" begin
          # Pairwise contraction with the M factor cores under (L, I, R) layout.
          v = ntuple(i->y[i].ttv_rks[k+1], M)
          w = ntuple(i->y[i].ttv_rks[k+2], M)
          Av = ntuple(i->y[i].ttv_vec[k+1], M)
          @tensor Qy[αₖ₊₁,ρₖ₊₁] := yₖ[ρₖ,iₖ,αₖ₊₁]*vec[k][ρₖ,iₖ,ρₖ₊₁]

          # Same Hadamard pattern as in ttrand_rounding (tt_randtools.jl): each
          # @tensor tmp_i[..., L, …, R] = Ai[l,L] * Qy[l, …, r] * Bi[r,R] expands
          # into two gemms — transpose(Ai) × Qy contraction over l, then × Bi over r.
          if M == 2
            Qy = reshape(Qy, v[1], v[2], out_rks[k+1])
            Qy = permutedims(Qy, (1,3,2))
            tmp = zeros(T, w[1], out_rks[k+1], w[2], dims[k+1])
            for i = 1:dims[k+1]
              tmp_i = view(tmp,:,:,:,i)
              Ai = Av[1][:,i,:]
              Bi = Av[2][:,i,:]
              tmp1 = transpose(Ai) * reshape(Qy, v[1], out_rks[k+1]*v[2])
              mul!(reshape(tmp_i, w[1]*out_rks[k+1], w[2]),
                   reshape(tmp1, w[1]*out_rks[k+1], v[2]),
                   Bi)
            end
            Qy = permutedims(tmp, (1,3,2,4))
          else
            αρ1 = prod(v[3:end])*out_rks[k+1]
            Qy = reshape(Qy, v[1], v[2], αρ1)
            Qy = permutedims(Qy, (1,3,2))
            tmp = zeros(T, w[1], αρ1, w[2], dims[k+1])
            for i = 1:dims[k+1]
              tmp_i = view(tmp,:,:,:,i)
              Ai = Av[1][:,i,:]
              Bi = Av[2][:,i,:]
              tmp1 = transpose(Ai) * reshape(Qy, v[1], αρ1*v[2])
              mul!(reshape(tmp_i, w[1]*αρ1, w[2]),
                   reshape(tmp1, w[1]*αρ1, v[2]),
                   Bi)
            end
            Qy = permutedims(tmp, (1,3,2,4))

            for mp = 3:2:M-1
              αρm = prod(v[mp+2:end])*out_rks[k+1]
              Qy = reshape(Qy, prod(w[1:mp-1]), v[mp], v[mp+1], αρm, dims[k+1])
              Qy = permutedims(Qy, (2,1,4,3,5))
              tmp = zeros(T, w[mp], prod(w[1:mp-1]), αρm, w[mp+1], dims[k+1])
              for i = 1:dims[k+1]
                tmp_i = view(tmp,:,:,:,:,i)
                Qy_i  = view(Qy,:,:,:,:,i)
                Ai = Av[mp][:,i,:]
                Bi = Av[mp+1][:,i,:]
                tmp1 = transpose(Ai) * reshape(Qy_i, v[mp], prod(w[1:mp-1])*αρm*v[mp+1])
                mul!(reshape(tmp_i, w[mp]*prod(w[1:mp-1])*αρm, w[mp+1]),
                     reshape(tmp1, w[mp]*prod(w[1:mp-1])*αρm, v[mp+1]),
                     Bi)
              end
              Qy = permutedims(tmp, (2,1,4,3,5))
            end

            if isodd(M)
              Qy = reshape(Qy, prod(w[1:M-1]), v[M], out_rks[k+1], dims[k+1])
              Qy = permutedims(Qy, (2,1,3,4))
              tmp = zeros(T, w[M], prod(w[1:M-1]), out_rks[k+1], dims[k+1])
              for i = 1:dims[k+1]
                tmp_i = view(tmp,:,:,:,i)
                Qy_i  = view(Qy,:,:,:,i)
                Ai = Av[M][:,i,:]
                mul!(reshape(tmp_i, w[M], prod(w[1:M-1])*out_rks[k+1]),
                     transpose(Ai),
                     reshape(Qy_i, v[M], prod(w[1:M-1])*out_rks[k+1]))
              end
              Qy = permutedims(tmp, (2,1,3,4))
            end
          end
          Qy = reshape(Qy, w..., out_rks[k+1], dims[k+1])
          # Layout (L, I, R) for yₖ: bring out_rks first, dims second, then w-product last.
          yₖ = reshape(permutedims(Qy, [M+1;M+2;1:M]), out_rks[k+1], dims[k+1], prod(w))
        end
      end
      vec[N] = reshape(yₖ, out_rks[N], dims[N], out_rks[N+1])
    end
    return TTvector{T,N}(N, vec, dims, out_rks, ot)
  end
end

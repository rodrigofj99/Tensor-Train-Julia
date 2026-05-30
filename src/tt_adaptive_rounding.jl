using TensorOperations
using LinearAlgebra
using Random
using TimerOutputs

# small helper to format floats compactly for verbose output
_fmt(x) = string(round(x, sigdigits=4))

# Diagnostic: per-bond exit residual estimator (norm(Sₖ)·√(sketch_rks/n_samples)),
# pushed at loop exit in the Hadamard NTuple variant if not nothing.
const _EXIT_RESIDUALS = Ref{Union{Nothing, Vector{Float64}}}(nothing)

# Diagnostic: sketch-estimated y_norm at start of the Hadamard variant.
const _Y_NORM_DIAG = Ref{Union{Nothing, Float64}}(nothing)


# Append new orthonormal directions to Q from the residual sketch Sₖ and return
# the expanded basis. Sₖ is consumed (qr! destroys it). `max_add` caps the number
# of columns we can add — used by callers to enforce the algebraic TT rank bound
# min(ℓ_max, prod(dims[1:k]), prod(dims[k+1:end])) − size(Q,2) at the current bond.
#
# Pivoted Householder QR (and σ-relative SVD) both suffer a 1/σ_min amplification
# on input whose smallest kept singular value sits near the noise floor — this
# blows the kept output columns into a noise-determined direction with O(eps/σ_min)
# overlap with span(Q). To detect this regime, we compare the squared Frobenius
# norms of orth(Sₖ) and its projection against Q: their difference equals exactly
# ‖Q'·orth(Sₖ)‖_F², a direct measure of how much of orth(Sₖ) lives in span(Q).
#
# * No detectable overlap AND all cols fit within max_add → RRQR (well-conditioned).
# * Otherwise → SVD with an *absolute* rank cut at min(round(Tr(proj)), max_add):
#   the integer effective rank of (I−QQ')·orth(Sₖ), capped by the algebraic
#   bond bound. This keeps only singular values well above the noise floor
#   (so the kept columns have ε-level forward error) and never inflates the TT
#   rank beyond what can carry real information.
function expand_basis(Q::AbstractMatrix{T}, Sₖ::AbstractMatrix{T};
                      max_add::Int=typemax(Int)) where {T<:Number}
  max_add <= 0 && return Q, 0
  Qn, _ = qr!(Sₖ); Qn = Matrix(Qn)
  Qn_proj = Qn - Q * (Q'*Qn)
  tr_full = size(Qn_proj, 2)
  tr_proj = sum(abs2, Qn_proj)
  if abs(tr_full - tr_proj) < 16 * eps() * tr_full && tr_full <= max_add
    Qn, _, rank_n = my_qc!(Qn_proj)
  else
    k = clamp(round(Int, tr_proj), 0, min(minimum(size(Qn_proj)), max_add))
    k == 0 && return Q, 0
    F = svd(Qn_proj)
    Qn = F.U[:, 1:k]
    rank_n = k
  end
  rank_n == 0 && return Q, 0
  return hcat(Q, Qn), rank_n
end

"""
    ttrand_rounding_adaptive(y::TTvector{T,N}, ε::Real;
                              n_probe=5, ℓ_min=4, ℓ_max=maximum(y.ttv_rks),
                              orthogonal=true, block_rks=N,
                              seed=1234, timer=TimerOutput()) -> TTvector{T,N}

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
- `n_probe::Int=5`: number of blocks added per inner-loop extension
- `ℓ_min::Int=4`: target width (columns) of the initial per-bond sketch
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
                                   ℓ_min::Int=4,
                                   ℓ_inc::Int=4,
                                   ℓ_max::Int=maximum(y.ttv_rks),
                                   n_samples::Int=max(N÷2, ℓ_inc, ceil(Int, 0.1 * ℓ_max)),
                                   orthogonal::Bool=true,
                                   block_rks::Int=N,
                                   block_rks_inc::Int=max(1, N÷4),
                                   seed::Int=1234,
                                   verbose::Bool=false,
                                   timer::TimerOutput=TimerOutput()) where {T,N}
  @assert n_samples >= ℓ_inc "n_samples ($n_samples) must be ≥ ℓ_inc ($ℓ_inc)"
  @timeit timer "ttrand_rounding_adaptive" begin
    dims = y.ttv_dims
    vec = Vector{Array{T,3}}(undef, N)
    # Initial sketch must cover the initial QR width (ℓ_min) plus the first
    # residual estimator's width (n_samples).
    @timeit timer "reverse_sketch" begin
      W, sketch_rks = tt_recursive_sketch(T, y, ℓ_min+n_samples; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
      seed = seed+1
    end
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Estimated Frobenius norm of y
    y_norm = norm(W[1])
    τ = ε * y_norm / sqrt(N - 1)
    rks_inc = ones(Int, N+1)
    rks_inc[1] = 0
    rks_inc[2:N] .= n_samples

    # Pre-compute right Gram matrices of y for the correct per-bond true-error baseline
    # G_right[k] = vec_y[k..N] · vec_y[k..N]' under (L, I, R) layout (L at position 1).
    G_right = Vector{Matrix{T}}(undef, N+1)
    if verbose
      G_right[N+1] = ones(T, 1, 1)
      for kk = N:-1:1
        Vk = y.ttv_vec[kk]
        G_prev = G_right[kk+1]
        G = zeros(T, size(Vk, 1), size(Vk, 1))
        @tensor G[a, b] = Vk[a, i, αp] * G_prev[αp, βp] * Vk[b, i, βp]
        G_right[kk] = G
      end
      println("[adaptive] y_norm(sketch)=$(_fmt(y_norm))  ‖y‖_exact=$(_fmt(norm(y)))  τ_bond=$(_fmt(τ))")
    end

    # Randomized sketching and orthogonalization. Local tensors use (L, I, R) layout.
    @timeit timer "orthogonalization" begin
      yₖ = reshape(y.ttv_vec[1], 1, dims[1], y.ttv_rks[2])
      @inbounds for k in 1:N-1
        max_basis = bond_rank_cap(dims, k, ℓ_max)
        # Randomized QR decomposition
        @timeit timer "Randomized adaptive QR decomposition" begin
          @timeit timer "Sketch" begin
            Zₖ = zeros(T, out_rks[k], dims[k], ℓ_min)
            Wₖ₊₁ = W[k+1][:,1:ℓ_min]
            @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁)  Zₖ[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*Wₖ₊₁[αₖ₊₁,ρₖ₊₁]
            Zₖ = reshape(Zₖ, out_rks[k]*dims[k], ℓ_min)
          end
          @timeit timer "QR" begin
            Q, _ = qr!(Zₖ)
            Q = Matrix(Q)
            ℓ = ℓ_min
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
                W_view = view(W[k+1], :, ℓ_min+1:ℓ_min+n_samples)
              @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁)  S_full_3d[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*W_view[αₖ₊₁,ρₖ₊₁]
            end
          end
          Sₖ = Matrix{T}(undef, out_rks[k]*dims[k], n_samples)
          @timeit timer "Residual sketch" begin
            copyto!(Sₖ, S_full)
            mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
          end

          if verbose
            V_Yk = reshape(yₖ, out_rks[k]*dims[k], y.ttv_rks[k+1])
            Gk = G_right[k+1]
            # Correct per-bond contribution to ‖y - ŷ‖²: ‖(I-QQ')·V(Y_k)·sqrt(G)‖²_F
            # = tr(V·G·V') - tr((Q'V) · G · (Q'V)')
            VGVt_trace = tr(V_Yk * Gk * V_Yk')
            QtV = Q' * V_Yk
            proj_trace = tr(QtV * Gk * QtV')
            true_contrib = sqrt(max(0.0, VGVt_trace - proj_trace))
            sqrt_VG  = sqrt(max(0.0, VGVt_trace))
            est      = norm(Sₖ) * sqrt(sketch_rks[k+1] / n_samples)
            println("[adaptive] bond ", lpad(k,2), " init   ℓ=", lpad(ℓ,3), " Qrk=", lpad(size(Q,2),3),
                    " skrk=", lpad(sketch_rks[k+1],4),
                    "  ‖Sₖ‖=", _fmt(norm(Sₖ)),
                    "  est=", _fmt(est),
                    "  true_contrib=", _fmt(true_contrib),
                    "  ‖V·sqrtG‖=", _fmt(sqrt_VG),
                    "  τ=", _fmt(τ))
          end

          @timeit timer "Adaptive basis expansion" begin
            iter = 0
            while norm(Sₖ) > τ * sqrt(n_samples/sketch_rks[k+1]) && size(Q, 2) < max_basis
              iter += 1
              @timeit timer "Add new orthonormal directions" begin
                # Geometric growth: absorb max(ℓ_inc, 0.2·rank(Q)) cols per iter,
                # capped at n_samples (the residual sketch width).
                ℓ_inc_eff = clamp(max(ℓ_inc, ceil(Int, 0.2 * size(Q, 2))), 1, n_samples)
                Q, rank_n = expand_basis(Q, view(Sₖ, :, 1:ℓ_inc_eff); max_add=max_basis - size(Q, 2))
                if rank_n == 0
                  if verbose
                    println("[adaptive] bond $k  iter $iter  rank_n=0 → breaking")
                  end
                  break
                end
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
                  W_extra, sketch_rks_extra = tt_recursive_sketch(T, y, rks_inc; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks_inc, timer=timer)
                  seed = seed+1
                  for l=k+1:N
                    s_prev = sketch_rks[l]
                    W[l] = cat(W[l], W_extra[l], dims=2)
                    sketch_rks[l] += sketch_rks_extra[l]
                    W[l][:,1:s_prev] .*= sqrt( s_prev / sketch_rks[l])
                    W[l][:,(s_prev+1):end] .*= sqrt(sketch_rks_extra[l] / sketch_rks[l])
                  end
                  # Renormalise the kept S_full cols (still in the OLD W portion).
                  S_full[:, 1:n_samples-ℓ_inc_eff] .*= sqrt(s_prev_kp1 / sketch_rks[k+1])
                end
              end
              @timeit timer "S_full tail contraction" begin
                # Fill the last ℓ_inc_eff cols of S_full from the (renormalised) W.
                W_tail = view(W[k+1], :, ℓ+n_samples-ℓ_inc_eff+1:ℓ+n_samples)
                tail_buf = zeros(T, out_rks[k], dims[k], ℓ_inc_eff)
                @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) tail_buf[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*W_tail[αₖ₊₁,ρₖ₊₁]
                copyto!(view(S_full, :, n_samples-ℓ_inc_eff+1:n_samples),
                        reshape(tail_buf, out_rks[k]*dims[k], ℓ_inc_eff))
              end
              @timeit timer "Residual sketch" begin
                copyto!(Sₖ, S_full)
                mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
              end

              if verbose
                V_Yk = reshape(yₖ, out_rks[k]*dims[k], y.ttv_rks[k+1])
                Gk = G_right[k+1]
                VGVt_trace = tr(V_Yk * Gk * V_Yk')
                QtV = Q' * V_Yk
                proj_trace = tr(QtV * Gk * QtV')
                true_contrib = sqrt(max(0.0, VGVt_trace - proj_trace))
                est       = norm(Sₖ) * sqrt(sketch_rks[k+1] / n_samples)
                println("[adaptive] bond ", lpad(k,2), " it=", lpad(iter,2),
                        "  ℓ=", lpad(ℓ,3), " Qrk=", lpad(size(Q,2),3),
                        " skrk=", lpad(sketch_rks[k+1],4),
                        "  ‖Sₖ‖=", _fmt(norm(Sₖ)),
                        "  est=", _fmt(est),
                        "  true_contrib=", _fmt(true_contrib),
                        "  τ=", _fmt(τ))
              end
            end
          end

          @timeit timer "Update core" begin
            out_rks[k+1] = size(Q,2)
            vec[k] = reshape(Q, out_rks[k], dims[k], out_rks[k+1])
            ot[k] = 1
          end
        end
        @timeit timer "Update yₖ" begin
          #update left parts
          yₖ₊₁ = zeros(T, out_rks[k+1], dims[k+1], y.ttv_rks[k+2])
          @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ₊₁)  yₖ₊₁[ρₖ₊₁,iₖ₊₁,αₖ₊₂] = yₖ[ρₖ,iₖ,αₖ₊₁]*vec[k][ρₖ,iₖ,ρₖ₊₁]*y.ttv_vec[k+1][αₖ₊₁,iₖ₊₁,αₖ₊₂]
          yₖ = yₖ₊₁
        end
        rks_inc[k+1] = 0
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
                                   ℓ_min::Int=4,
                                   ℓ_inc::Int=4,
                                   n_samples::Int=max(N÷2, ℓ_inc),
                                   ℓ_max::Int=sum(maximum(yⱼ.ttv_rks) for yⱼ in y),
                                   orthogonal::Bool=true,
                                   block_rks::Int=N,
                                   seed::Int=1234,
                                   timer::TimerOutput=TimerOutput()) where {T,N}
  @assert n_samples >= ℓ_inc "n_samples ($n_samples) must be ≥ ℓ_inc ($ℓ_inc)"
  @timeit timer "ttrand_rounding_adaptive" begin
    m = length(α)
    @assert length(y) == m
    dims = y[1].ttv_dims
    @assert all(y[j].ttv_dims == dims for j=2:m)

    vec = Vector{Array{T,3}}(undef, N)
    # Initial sketches — same seed across j for linearity
    W = Vector{Vector{Matrix{T}}}(undef, m)
    local sketch_rks
    @timeit timer "reverse_sketch" begin
      for j = 1:m
        Wⱼ, skⱼ = tt_recursive_sketch(T, y[j], ℓ_min+n_samples; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
        W[j] = Wⱼ
        if j == 1
          sketch_rks = skⱼ
        else
          @assert skⱼ == sketch_rks
        end
      end
      seed = seed + 1
    end
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Boundary sketch of the linear combination (linearity via shared seed)
    y_norm = norm(sum(α[j].*W[j][1] for j=1:m))
    τ = ε * y_norm / sqrt(N - 1)
    rks_inc = ones(Int, N+1)
    rks_inc[1] = 0
    rks_inc[2:N] .= n_samples

    @timeit timer "orthogonalization" begin
      Yₖ = [α[j].*reshape(y[j].ttv_vec[1], 1, dims[1], y[j].ttv_rks[2]) for j=1:m]
      @inbounds for k in 1:N-1
        max_basis = bond_rank_cap(dims, k, ℓ_max)
        @timeit timer "Randomized adaptive QR decomposition" begin
          @timeit timer "Sketch" begin
            Zₖ = zeros(T, out_rks[k], dims[k], ℓ_min)
            for j = 1:m
              Wⱼₖ₊₁ = W[j][k+1][:, 1:ℓ_min]
              @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[ρₖ,iₖ,ρₖ₊₁] += Yₖ[j][ρₖ,iₖ,αₖ₊₁]*Wⱼₖ₊₁[αₖ₊₁,ρₖ₊₁]
            end
            Zₖ = reshape(Zₖ, out_rks[k]*dims[k], ℓ_min)
          end
          @timeit timer "QR" begin
            Q, _ = qr!(Zₖ)
            Q = Matrix(Q)
            ℓ = ℓ_min
          end

          # S_full holds only the active window: cols ℓ+1..ℓ+n_samples of Σⱼ Yₖ[j]·W[j][k+1].
          @timeit timer "Initial residual sketch" begin
            S_full = zeros(T, out_rks[k]*dims[k], n_samples)
            let S_full_3d = reshape(S_full, out_rks[k], dims[k], n_samples)
              for j = 1:m
                Wⱼ_view = view(W[j][k+1], :, ℓ_min+1:ℓ_min+n_samples)
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
            while norm(Sₖ) > τ * sqrt(n_samples/sketch_rks[k+1]) && size(Q, 2) < max_basis
              @timeit timer "Add new orthonormal directions" begin
                ℓ_inc_eff = clamp(max(ℓ_inc, ceil(Int, 0.2 * size(Q, 2))), 1, n_samples)
                Q, rank_n = expand_basis(Q, view(Sₖ, :, 1:ℓ_inc_eff); max_add=max_basis - size(Q, 2))
                if rank_n == 0
                  break
                end
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
                  W_extra = Vector{Vector{Matrix{T}}}(undef, m)
                  local sketch_rks_extra
                  for j = 1:m
                    Wⱼ_extra, skⱼ_extra = tt_recursive_sketch(T, y[j], rks_inc; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
                    W_extra[j] = Wⱼ_extra
                    if j == 1
                      sketch_rks_extra = skⱼ_extra
                    end
                  end
                  seed = seed + 1
                  for l = k+1:N
                    s_prev = sketch_rks[l]
                    new_total = s_prev + sketch_rks_extra[l]
                    for j = 1:m
                      W[j][l] = cat(W[j][l], W_extra[j][l], dims=2)
                      W[j][l][:, 1:s_prev] .*= sqrt(s_prev / new_total)
                      W[j][l][:, (s_prev+1):end] .*= sqrt(sketch_rks_extra[l] / new_total)
                    end
                    sketch_rks[l] = new_total
                  end
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
          Yₖ₊₁ = [zeros(T, out_rks[k+1], dims[k+1], y[j].ttv_rks[k+2]) for j=1:m]
          for j = 1:m
            @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ₊₁) Yₖ₊₁[j][ρₖ₊₁,iₖ₊₁,αₖ₊₂] = Yₖ[j][ρₖ,iₖ,αₖ₊₁]*vec[k][ρₖ,iₖ,ρₖ₊₁]*y[j].ttv_vec[k+1][αₖ₊₁,iₖ₊₁,αₖ₊₂]
          end
          Yₖ = Yₖ₊₁
        end
        rks_inc[k+1] = 0
      end
      vec[N] = sum(Yₖ[j] for j=1:m)
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
                                   ℓ_min::Int=4,
                                   ℓ_inc::Int=4,
                                   n_samples::Int=max(N÷2, ℓ_inc),
                                   ℓ_max::Int=maximum(Atto.tto_rks .* y.ttv_rks) + maximum(b.ttv_rks),
                                   orthogonal::Bool=true,
                                   block_rks::Int=N,
                                   seed::Int=1234,
                                   timer::TimerOutput=TimerOutput()) where {T,N}
  @assert n_samples >= ℓ_inc "n_samples ($n_samples) must be ≥ ℓ_inc ($ℓ_inc)"
  @timeit timer "ttrand_rounding_adaptive" begin
    dims = y.ttv_dims
    vec = Vector{Array{T,3}}(undef, N)

    @timeit timer "reverse_sketch" begin
      WAy, sketch_rks = tt_recursive_sketch(T, Atto, y, ℓ_min+n_samples; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
      Wb,  sk_b       = tt_recursive_sketch(T,       b, ℓ_min+n_samples; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
      @assert sk_b == sketch_rks
      seed = seed + 1
    end
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Boundary sketches share seed → linearity holds for the difference
    y_norm = norm(Base.vec(WAy[1]) .- Base.vec(Wb[1]))
    τ = ε * y_norm / sqrt(N - 1)
    rks_inc = ones(Int, N+1)
    rks_inc[1] = 0
    rks_inc[2:N] .= n_samples

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
        max_basis = bond_rank_cap(dims, k, ℓ_max)
        @timeit timer "Randomized adaptive QR decomposition" begin
          @timeit timer "Sketch" begin
            Zₖ = zeros(T, out_rks[k], dims[k], ℓ_min)
            WAyₖ₊₁ = WAy[k+1][:, :, 1:ℓ_min]
            Wbₖ₊₁  = Wb[k+1][:, 1:ℓ_min]
            @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[ρₖ,iₖ,ρₖ₊₁] = Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*WAyₖ₊₁[αₖ₊₁,βₖ₊₁,ρₖ₊₁] - bₖ[ρₖ,iₖ,αₖ₊₁]*Wbₖ₊₁[αₖ₊₁,ρₖ₊₁]
            Zₖ = reshape(Zₖ, out_rks[k]*dims[k], ℓ_min)
          end
          @timeit timer "QR" begin
            Q, _ = qr!(Zₖ)
            Q = Matrix(Q)
            ℓ = ℓ_min
          end

          # S_full holds only the active window: cols ℓ+1..ℓ+n_samples.
          @timeit timer "Initial residual sketch" begin
            S_full = zeros(T, out_rks[k]*dims[k], n_samples)
            let S_full_3d = reshape(S_full, out_rks[k], dims[k], n_samples),
                WAy_view = view(WAy[k+1], :, :, ℓ_min+1:ℓ_min+n_samples),
                Wb_view  = view(Wb[k+1],  :,    ℓ_min+1:ℓ_min+n_samples)
              @tensoropt (αₖ₊₁,βₖ₊₁,ρₖ,ρₖ₊₁) S_full_3d[ρₖ,iₖ,ρₖ₊₁] = Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*WAy_view[αₖ₊₁,βₖ₊₁,ρₖ₊₁] - bₖ[ρₖ,iₖ,αₖ₊₁]*Wb_view[αₖ₊₁,ρₖ₊₁]
            end
          end
          Sₖ = Matrix{T}(undef, out_rks[k]*dims[k], n_samples)
          @timeit timer "Residual sketch" begin
            copyto!(Sₖ, S_full)
            mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
          end

          @timeit timer "Adaptive basis expansion" begin
            while norm(Sₖ) > τ * sqrt(n_samples/sketch_rks[k+1]) && size(Q, 2) < max_basis
              @timeit timer "Add new orthonormal directions" begin
                ℓ_inc_eff = clamp(max(ℓ_inc, ceil(Int, 0.2 * size(Q, 2))), 1, n_samples)
                Q, rank_n = expand_basis(Q, view(Sₖ, :, 1:ℓ_inc_eff); max_add=max_basis - size(Q, 2))
                if rank_n == 0
                  break
                end
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
                  WAy_extra, sk_extra   = tt_recursive_sketch(T, Atto, y, rks_inc; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
                  Wb_extra,  sk_extra_b = tt_recursive_sketch(T,       b, rks_inc; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
                  @assert sk_extra_b == sk_extra
                  seed = seed + 1
                  for l = k+1:N
                    s_prev = sketch_rks[l]
                    new_total = s_prev + sk_extra[l]
                    WAy[l] = cat(WAy[l], WAy_extra[l], dims=3)
                    Wb[l]  = cat(Wb[l],  Wb_extra[l],  dims=2)
                    WAy[l][:, :, 1:s_prev] .*= sqrt(s_prev / new_total)
                    WAy[l][:, :, (s_prev+1):end] .*= sqrt(sk_extra[l] / new_total)
                    Wb[l][:, 1:s_prev] .*= sqrt(s_prev / new_total)
                    Wb[l][:, (s_prev+1):end] .*= sqrt(sk_extra[l] / new_total)
                    sketch_rks[l] = new_total
                  end
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
          bₖ₊₁  = zeros(T, out_rks[k+1], dims[k+1], b.ttv_rks[k+2])
          @tensoropt (αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂,ρₖ₊₁) Ayₖ₊₁[ρₖ₊₁,iₖ₊₁,αₖ₊₂,βₖ₊₂] = Ayₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁]*vec[k][ρₖ,iₖ,ρₖ₊₁]*y.ttv_vec[k+1][αₖ₊₁,jₖ₊₁,αₖ₊₂]*Atto.tto_vec[k+1][βₖ₊₁,iₖ₊₁,jₖ₊₁,βₖ₊₂]
          @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ₊₁) bₖ₊₁[ρₖ₊₁,iₖ₊₁,αₖ₊₂] = bₖ[ρₖ,iₖ,αₖ₊₁]*vec[k][ρₖ,iₖ,ρₖ₊₁]*b.ttv_vec[k+1][αₖ₊₁,iₖ₊₁,αₖ₊₂]
          Ayₖ = Ayₖ₊₁
          bₖ  = bₖ₊₁
        end
        rks_inc[k+1] = 0
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
                                   ℓ_min::Int=4,
                                   ℓ_inc::Int=4,
                                   ℓ_max::Int=prod(maximum(y[i].ttv_rks) for i=1:M),
                                   n_samples::Int=max(N÷2, ℓ_inc, ceil(Int, 0.1 * ℓ_max)),
                                   orthogonal::Bool=true,
                                   block_rks::Int=N,
                                   block_rks_inc::Int=max(1, N÷4),
                                   seed::Int=1234,
                                   bond_callback::Union{Nothing,Function}=nothing,
                                   inner_callback::Union{Nothing,Function}=nothing,
                                   timer::TimerOutput=TimerOutput()) where {T,N,M}
  if M == 1
    return ttrand_rounding_adaptive(y[1], ε; ℓ_min=ℓ_min, ℓ_inc=ℓ_inc, n_samples=n_samples, ℓ_max=ℓ_max, orthogonal=orthogonal, block_rks=block_rks, seed=seed, timer=timer)
  end
  @assert n_samples >= ℓ_inc "n_samples ($n_samples) must be ≥ ℓ_inc ($ℓ_inc)"

  @timeit timer "ttrand_rounding_adaptive" begin
    dims = y[1].ttv_dims
    vec  = Vector{Array{T,3}}(undef, N)

    @timeit timer "reverse_sketch" begin
      W, sketch_rks = tt_recursive_sketch(T, y, ℓ_min+n_samples; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
      seed = seed + 1
    end
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Sketch-based ‖y‖ from the boundary contraction of the recursive sketch
    # (Al Daas et al., 2025, Remark 3.1 / eqn 3.7: ‖X‖ ≈ ‖V(X_1)·W_2‖_F / √r).
    # The 1/√r factor is already absorbed by the W[k] ./= sqrt(p[k]) normalisation
    # in tt_recursive_sketch. The estimator is unbiased but has variance scaling
    # poorly with the number of modes — see Theorem 3.4 in the same paper. Larger
    # n_samples (controlled via the default `0.1 * ℓ_max`, matching f_init=0.1
    # from the paper's §4.2 experiments) keeps this variance manageable.
    y_norm = norm(W[1])
    if _Y_NORM_DIAG[] !== nothing
      _Y_NORM_DIAG[] = y_norm
    end
    τ = ε * y_norm / sqrt(N - 1)
    rks_inc = ones(Int, N+1)
    rks_inc[1] = 0
    rks_inc[2:N] .= n_samples

    @timeit timer "orthogonalization sweep" begin
      yₖ = broadcast(*, (reshape(y[i].ttv_vec[1],
                                  1, dims[1], ntuple(j->( j==i ? y[i].ttv_rks[2] : 1), M)...)
                         for i=1:M)...)
      yₖ = reshape(yₖ, 1, dims[1], prod(y[i].ttv_rks[2] for i=1:M))

      @inbounds for k in 1:N-1
        max_basis = bond_rank_cap(dims, k, ℓ_max)
        @timeit timer "Randomized adaptive QR decomposition" begin
          @timeit timer "Sketch" begin
            Zₖ = zeros(T, out_rks[k], dims[k], ℓ_min)
            W_full = reshape(W[k+1], prod(y[i].ttv_rks[k+1] for i=1:M), sketch_rks[k+1])
            Wₖ₊₁ = W_full[:, 1:ℓ_min]
            @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) Zₖ[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*Wₖ₊₁[αₖ₊₁,ρₖ₊₁]
            Zₖ = reshape(Zₖ, out_rks[k]*dims[k], ℓ_min)
          end
          @timeit timer "QR" begin
            Q, _ = qr!(Zₖ)
            Q = Matrix(Q)
            ℓ = ℓ_min
          end

          # S_full holds only the active window of yₖ × W[k+1].
          @timeit timer "Initial residual sketch" begin
            S_full = zeros(T, out_rks[k]*dims[k], n_samples)
            let S_full_3d = reshape(S_full, out_rks[k], dims[k], n_samples),
                W_full   = reshape(W[k+1], prod(y[i].ttv_rks[k+1] for i=1:M), sketch_rks[k+1]),
                W_view   = view(W_full, :, ℓ_min+1:ℓ_min+n_samples)
              @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) S_full_3d[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*W_view[αₖ₊₁,ρₖ₊₁]
            end
          end
          Sₖ = Matrix{T}(undef, out_rks[k]*dims[k], n_samples)
          @timeit timer "Residual sketch" begin
            copyto!(Sₖ, S_full)
            mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
          end

          @timeit timer "Adaptive basis expansion" begin
            if inner_callback !== nothing
              inner_callback(k, Q, Sₖ, sketch_rks[k+1], n_samples, yₖ, out_rks, τ)
            end
            while norm(Sₖ) > τ * sqrt(n_samples/sketch_rks[k+1]) && size(Q, 2) < max_basis
              @timeit timer "Add new orthonormal directions" begin
                ℓ_inc_eff = clamp(max(ℓ_inc, ceil(Int, 0.2 * size(Q, 2))), 1, n_samples)
                Q, rank_n = expand_basis(Q, view(Sₖ, :, 1:ℓ_inc_eff); max_add=max_basis - size(Q, 2))
                if rank_n == 0
                  break
                end
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
                  W_extra, sketch_rks_extra = tt_recursive_sketch(T, y, rks_inc; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks_inc, timer=timer)
                  seed = seed + 1
                  for l = k+1:N
                    s_prev = sketch_rks[l]
                    new_total = s_prev + sketch_rks_extra[l]
                    W[l] = cat(W[l], W_extra[l], dims=M+1)  # sketch dim is the last one
                    W2 = reshape(W[l], :, new_total)        # view; in-place rescale below
                    W2[:, 1:s_prev] .*= sqrt(s_prev / new_total)
                    W2[:, (s_prev+1):end] .*= sqrt(sketch_rks_extra[l] / new_total)
                    sketch_rks[l] = new_total
                  end
                  S_full[:, 1:n_samples-ℓ_inc_eff] .*= sqrt(s_prev_kp1 / sketch_rks[k+1])
                end
              end
              @timeit timer "S_full tail contraction" begin
                tail_buf = zeros(T, out_rks[k], dims[k], ℓ_inc_eff)
                W_full = reshape(W[k+1], prod(y[i].ttv_rks[k+1] for i=1:M), sketch_rks[k+1])
                W_tail = view(W_full, :, ℓ+n_samples-ℓ_inc_eff+1:ℓ+n_samples)
                @tensoropt (αₖ₊₁,ρₖ,ρₖ₊₁) tail_buf[ρₖ,iₖ,ρₖ₊₁] = yₖ[ρₖ,iₖ,αₖ₊₁]*W_tail[αₖ₊₁,ρₖ₊₁]
                copyto!(view(S_full, :, n_samples-ℓ_inc_eff+1:n_samples),
                        reshape(tail_buf, out_rks[k]*dims[k], ℓ_inc_eff))
              end
              @timeit timer "Residual sketch" begin
                copyto!(Sₖ, S_full)
                mul!(Sₖ, Q, Q' * Sₖ, -one(T), one(T))
              end
              if inner_callback !== nothing
                inner_callback(k, Q, Sₖ, sketch_rks[k+1], n_samples, yₖ, out_rks, τ)
              end
            end
          end

          # Diagnostic: record per-bond exit-residual estimate (in ambient norm).
          if _EXIT_RESIDUALS[] !== nothing
            push!(_EXIT_RESIDUALS[], norm(Sₖ) * sqrt(sketch_rks[k+1] / n_samples))
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
        if bond_callback !== nothing
          bond_callback(k, vec, yₖ, out_rks, dims)
        end
        rks_inc[k+1] = 0
      end
      vec[N] = reshape(yₖ, out_rks[N], dims[N], out_rks[N+1])
      if bond_callback !== nothing
        bond_callback(N, vec, yₖ, out_rks, dims)
      end
    end
    return TTvector{T,N}(N, vec, dims, out_rks, ot)
  end
end

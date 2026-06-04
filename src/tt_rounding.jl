using Base.Threads
using LinearAlgebra
import LinearAlgebra.norm
import Base.floor

function r_and_d_to_rks(rks,dims;rmax=1024)
	new_rks = ones(eltype(rks),length(rks)) 
	@simd for i in eachindex(dims)
		if  prod(dims[i:end]) > 0
			if prod(dims[1:i-1]) > 0
				new_rks[i] = min(rks[i],prod(dims[1:i-1]),prod(dims[i:end]),rmax)
			else 
				new_rks[i] = min(rks[i],prod(dims[i:end]),rmax)
			end
		else 
			if prod(dims[1:i-1]) > 0
				new_rks[i] = min(rks[i],prod(dims[1:i-1]),rmax)
			else 
				new_rks[i] = min(rks[i],rmax)
			end
		end
	end
	return new_rks
end

#local ttvec rank increase function with noise ϵ_wn. Cores have layout (L, I, R).
function tt_up_rks_noise(tt_vec,tt_ot_i,rkm,rk,ϵ_wn)
	ni = size(tt_vec, 2)
	L_old, R_old = size(tt_vec, 1), size(tt_vec, 3)
	vec_out = zeros(eltype(tt_vec), rkm, ni, rk)
	vec_out[1:L_old, :, 1:R_old] = tt_vec
	if !iszero(ϵ_wn)
		if rkm == L_old && rk > R_old
			Q = rand_orthogonal(rkm*ni, rk - R_old)
			vec_out[:, :, R_old+1:rk] = ϵ_wn*reshape(Q, rkm, ni, rk - R_old)
			tt_ot_i = 0
		elseif rk == R_old && rkm > L_old
			Q = rand_orthogonal(rkm - L_old, ni*rk)
			vec_out[L_old+1:rkm, :, :] = ϵ_wn*reshape(Q, rkm - L_old, ni, rk)
			tt_ot_i = 0
		elseif rk > R_old && rkm > L_old
			Q = rand_orthogonal((rkm - L_old)*ni, rk - R_old)
			vec_out[L_old+1:rkm, :, R_old+1:rk] = ϵ_wn*reshape(Q, rkm - L_old, ni, rk - R_old)
		end
	end
	return vec_out
end

"""
returns the TTvector with ranks rks and noise ϵ_wn for the updated ranks
"""
function tt_up_rks(x_tt::TTvector{T,N},rk_max::Int;rks=vcat(1,rk_max*ones(Int,length(x_tt.ttv_dims)-1),1),ϵ_wn=0.0) where {T<:Number,N}
	d = x_tt.N
	vec_out = Array{Array{T}}(undef,d)
	out_ot = zeros(Int64,d)
	@assert(rk_max >= maximum(x_tt.ttv_rks),"New bond dimension too low")
	rks = r_and_d_to_rks(rks,x_tt.ttv_dims;rmax=rk_max)
	for i in 1:d
		vec_out[i] = tt_up_rks_noise(x_tt.ttv_vec[i],x_tt.ttv_ot[i],rks[i],rks[i+1],ϵ_wn)
	end	
	return TTvector{T,N}(d,vec_out,x_tt.ttv_dims,rks,out_ot)
end

"""
	returns the orthogonalized TTvector with root i with ranks at most max(nⱼ)rₖ
"""
function orthogonalize(x_tt::TTvector{T,N};i=1::Int) where {T<:Number,N}
	d = x_tt.N
	@assert(1≤i≤d, DimensionMismatch("Impossible orthogonalization"))
	y_rks = r_and_d_to_rks(x_tt.ttv_rks,x_tt.ttv_dims)
	# Every core is overwritten below, so skip zeros_tt's wasteful initialization
	# (would otherwise allocate + zero ~hundreds of MB only to throw it away).
	y_tt = TTvector{T,N}(d, Vector{Array{T,3}}(undef, d),
	                    x_tt.ttv_dims, copy(y_rks), zeros(Int, d))
	# Cores have layout (L, I, R). Left-orthogonalization: unfold each core
	# as (L*I, R) — a contiguous column-major reshape — and QR.
	# Two scratch arrays, reused across bonds:
	#   buf       — gemm output destination (flat, wrapped via unsafe_wrap so
	#                BLAS gemm stays on the contiguous fast path);
	#   FR_/FL_storage — R/L factor carry-over between bonds (sliced via view,
	#                still BLAS-strided for mul! source).
	# GC.@preserve keeps `buf` rooted across the unsafe_wrap window.
	# True (tight) upper bounds on the bond ranks reached during each sweep,
	# from the orthogonalization recurrence k_{j+1} = min(k_j · n_j, r_input_{j+1}).
	# The r_and_d_to_rks bound (y_rks) only constrains the final TT ranks — actual
	# y_tt.ttv_rks during the sweep can exceed it on the way out.
	left_max = ones(Int, d+1)
	for j in 1:i-1
		left_max[j+1] = min(left_max[j] * x_tt.ttv_dims[j], x_tt.ttv_rks[j+1])
	end
	right_max = ones(Int, d+1)
	for j in d:-1:i+1
		right_max[j] = min(x_tt.ttv_rks[j], x_tt.ttv_dims[j] * right_max[j+1])
	end
	max_buf_left  = i > 1 ? maximum(left_max[j]  * x_tt.ttv_dims[j] * x_tt.ttv_rks[j+1] for j in 1:i-1) : 0
	max_buf_right = i < d ? maximum(x_tt.ttv_rks[j] * x_tt.ttv_dims[j] * right_max[j+1] for j in i+1:d) : 0
	buf = Vector{T}(undef, max(max_buf_left, max_buf_right))
	max_FR_rows = i > 1 ? maximum(left_max[j+1] for j in 1:i-1) : 1
	max_FR_cols = i > 1 ? maximum(x_tt.ttv_rks[j+1] for j in 1:i-1) : 1
	FR_storage  = Matrix{T}(undef, max_FR_rows, max_FR_cols)
	max_FL_rows = i < d ? maximum(x_tt.ttv_rks[j] for j in i+1:d) : 1
	max_FL_cols = i < d ? maximum(right_max[j] for j in i+1:d) : 1
	FL_storage  = Matrix{T}(undef, max_FL_rows, max_FL_cols)
	FR = ones(T,1,1)
	GC.@preserve buf begin
		for j in 1:i-1
			y_tt.ttv_ot[j]=1
			c = x_tt.ttv_vec[j]
			rj_new = y_tt.ttv_rks[j]
			nj = x_tt.ttv_dims[j]
			rj = x_tt.ttv_rks[j+1]
			# FR (rj_new, rj_old) × core (rj_old, I*R) → contiguous (rj_new, I*R).
			M = unsafe_wrap(Array, pointer(buf), (rj_new, nj * rj))
			mul!(M, FR, reshape(c, size(c, 1), :))
			# qr! mutates M in-place: F.factors aliases buf, but Matrix(F.Q) and
			# the explicit R extraction below copy out before the next iteration's
			# mul! overwrites buf.
			F = qr!(reshape(M, rj_new*nj, rj))
			Q = Matrix(F.Q)
			y_tt.ttv_rks[j+1] = size(Q, 2)
			y_tt.ttv_vec[j] = reshape(Q, rj_new, nj, y_tt.ttv_rks[j+1])
			# Copy R (upper triangle of F.factors[1:k, 1:rj]) into FR_storage.
			k = y_tt.ttv_rks[j+1]
			FR = view(FR_storage, 1:k, 1:rj)
			copyto!(FR, view(F.factors, 1:k, 1:rj))
			triu!(FR)
		end
		# Right-orthogonalization: unfold each core as (L, I*R) and LQ.
		FL = ones(T,1,1)
		for j in d:-1:i+1
			y_tt.ttv_ot[j]=-1
			c = x_tt.ttv_vec[j]
			rj_prev = x_tt.ttv_rks[j]
			nj = x_tt.ttv_dims[j]
			rj_new = y_tt.ttv_rks[j+1]
			# core (L*I, R_old) × FL (R_old, R_new) → contiguous (L*I, R_new).
			M = unsafe_wrap(Array, pointer(buf), (rj_prev * nj, rj_new))
			mul!(M, reshape(c, rj_prev * nj, size(c, 3)), FL)
			F = lq!(reshape(M, rj_prev, nj * rj_new))
			Q = Matrix(F.Q)
			y_tt.ttv_rks[j] = size(Q, 1)
			y_tt.ttv_vec[j] = reshape(Q, y_tt.ttv_rks[j], nj, rj_new)
			# Copy L (lower triangle of F.factors[1:rj_prev, 1:k]) into FL_storage.
			k = y_tt.ttv_rks[j]
			FL = view(FL_storage, 1:rj_prev, 1:k)
			copyto!(FL, view(F.factors, 1:rj_prev, 1:k))
			tril!(FL)
		end
	end
	y_tt.ttv_ot[i]=0
	# center = FR × core × FL. Two gemms:
	#   tmp[γ, μ*β]  = core[γ, μ*δ] × FL_reshaped? Order: core×FL first since core is the big factor.
	#   tmp[γ*μ, β]  = core[γ*μ, δ] × FL[δ, β]
	#   center[α, μ*β] = FR[α, γ] × tmp[γ, μ*β]
	c = x_tt.ttv_vec[i]
	γ, μ, δ = size(c)
	tmp = reshape(c, γ*μ, δ) * FL
	y_tt.ttv_vec[i] = reshape(FR * reshape(tmp, γ, μ*size(FL,2)),
	                          size(FR,1), μ, size(FL,2))
	return y_tt
end

function cut_off_index(s::Array{T}, tol::Float64; degen_tol=1e-10) where {T<:Number}
	k = sum(s.>norm(s)*tol)
	while k<length(s) && isapprox(s[k],s[k+1];rtol=degen_tol, atol=degen_tol)
		k = k+1
	end
	return k
end

function LinearAlgebra.norm(v::TTvector)
	if length(findall(v.ttv_ot.==0))==1 #orthogonalized TTvector
		return norm(v.ttv_vec[findfirst(v.ttv_ot.==0)])
	else 
		w = orthogonalize(v;i=v.N)
		return norm(w.ttv_vec[end])
	end
end

function full_orthogonalize(x_tt::TTvector{T,N};i=1::Int) where {T,N}
	return orthogonalize(orthogonalize(x_tt,i=1),i=i)
end

"""
    tt_rounding(x_tt::TTvector; tol=1e-12, rmax=2^14, direction=:left) -> TTvector

Deterministic TT rounding: returns a new TTvector that approximates `x_tt` with
the smallest ranks such that the discarded singular values satisfy the relative
Frobenius tolerance `tol` (capped per bond by `rmax`).

`direction=:left` left-orthogonalizes then sweeps right-to-left (result is
right-orthogonal); `direction=:right` does the mirror. Both delegate to
`_tt_rounding`, whose bond truncations use `truncated_svd!` (QR/LQ pre-reduction
on skewed unfoldings, thin `gesvd`). See `tt_rounding!` for the in-place variant.
"""
function tt_rounding(x_tt::TTvector{T,N}; tol=1e-12, rmax=2^14, direction=:left) where {T<:Number,N}
	if direction == :left
		y_tt = is_leftorthogonal(x_tt) ? copy(x_tt) : orthogonalize(x_tt; i=N)
	else
		y_tt = is_rightorthogonal(x_tt) ? copy(x_tt) : orthogonalize(x_tt; i=1)
	end

	return _tt_rounding(y_tt; tol=tol, rmax=rmax, direction=direction)
end



function tt_rounding!(x_tt::TTvector{T,N}; tol=1e-12, rmax=2^14, direction=:left) where {T<:Number,N}
	if direction == :left
		y_tt = is_leftorthogonal(x_tt) ? x_tt : orthogonalize(x_tt; i=N)
	else
		y_tt = is_rightorthogonal(x_tt) ? x_tt : orthogonalize(x_tt; i=1)
	end

	_tt_rounding(y_tt; tol=tol, rmax=rmax, direction=direction)
	return 
end



"""
    truncated_svd!(M, tol_per_bond; rmax, threshold=1.5)

Truncated thin SVD of `M`, returning `(U_trunc, S_trunc, Vt_trunc, k)` where
the rank `k` is chosen as the largest index whose tail singular values fit the
per-bond tolerance budget (and is capped by `rmax`).

For skewed matrices (aspect ratio ≥ `threshold`), pre-reduces via QR (tall) or
LQ (wide) and runs the SVD on the small square R / L factor. The Householder
factors of the QR/LQ are then applied only to the truncated `k` columns of `U`
or rows of `Vt`, via `LAPACK.ormqr!` / `LAPACK.ormlq!`. This avoids both the
quadratic `Matrix(F.Q)` materialisation and the wasteful `(min(m,n) − k)`
output columns that direct `svd!` would compute and throw away — saving 30–50%
wall time at Matern-bond aspect ratios.

Mutates `M`. The returned `U_trunc` and `Vt_trunc` are fresh allocations of
sizes `(m, k)` and `(k, n)` respectively.
"""
function truncated_svd!(M::AbstractMatrix{T}, tol_per_bond::Real;
                        rmax::Int=typemax(Int), threshold::Float64=1.5) where T
    m, n = size(M)
    # Apple Accelerate's in-place dgelqf!/dgesdd! intermittently return wrong
    # results when M aliases recently-freed memory (e.g. a TT core that was
    # just materialised from a QR Q in orthogonalize). The fix is to pass a
    # fresh copy of the input to every in-place LAPACK factorisation in this
    # path. The copy is small relative to the factorisation work itself.
    if n >= threshold * m
        # Wide: LQ + SVD on L + apply Q on right to truncated rows of Vt_l.
        M_lq, tau = LAPACK.gelqf!(copy(M))
        L = M_lq[1:m, 1:m]
        tril!(L)
        u, s, vt_l = lapack_thin_svd!(L)
        _, k = floor(s, tol_per_bond)
        k = min(k, rmax, count(>(0.0), s))
        Vt_full = zeros(T, k, n)
        @views Vt_full[:, 1:m] .= vt_l[1:k, :]
        LAPACK.ormlq!('R', 'N', M_lq, tau, Vt_full)
        return (U=u[:, 1:k], S=s[1:k], Vt=Vt_full, k=k)

    elseif m >= threshold * n
        # Tall: QR + SVD on R + apply Q on left to truncated columns of U_r.
        M_qr, tau = LAPACK.geqrf!(copy(M))
        R = M_qr[1:n, 1:n]
        triu!(R)
        u_r, s, vt = lapack_thin_svd!(R)
        _, k = floor(s, tol_per_bond)
        k = min(k, rmax, count(>(0.0), s))
        U_full = zeros(T, m, k)
        @views U_full[1:n, :] .= u_r[:, 1:k]
        LAPACK.ormqr!('L', 'N', M_qr, tau, U_full)
        return (U=U_full, S=s[1:k], Vt=vt[1:k, :], k=k)

    else
        # Near-square: direct SVD.
        u, s, vt = lapack_thin_svd!(M)
        _, k = floor(s, tol_per_bond)
        k = min(k, rmax, count(>(0.0), s))
        return (U=u[:, 1:k], S=s[1:k], Vt=vt[1:k, :], k=k)
    end
end

# Thin SVD via LAPACK.gesvd! (QR-iteration driver) on a fresh copy of the
# input. Although LAPACK.gesdd! (divide-and-conquer) is 2-3× faster on
# square matrices, Apple's Accelerate dgesdd has a documented intermittent
# correctness bug after heavy preceding LAPACK calls (orthogonalize's
# qr!/Matrix(F.Q) chain triggers it). gesvd is reliable and the slowdown is
# bounded since this SVD acts on the small m×m or n×n L/R after QR/LQ
# pre-reduction, not the full unfold.
function lapack_thin_svd!(M::AbstractMatrix{T}) where T
    Mc = copy(M)
    return LAPACK.gesvd!('S', 'S', Mc)
end

"""
Internal function for rounding. It's meant to be used by a wrapper.

The per-bond truncation budget is `tol/√(N−1)` of the local unfolding's Frobenius
norm; summed over the N−1 bonds this guarantees ‖x − x̂‖_F ≤ tol·‖x‖_F.
"""
function _tt_rounding(y_tt::TTvector{T,N}; tol=1e-12, rmax=2^14, direction=:left) where {T<:Number,N}
    tol_per_bond = N > 1 ? tol / sqrt(N - 1) : tol
    if direction == :left
        # =====================================================================
        # RIGHT-TO-LEFT COMPRESSION SWEEP (Requires Left-Orthogonalization)
        # =====================================================================
        if norm(y_tt.ttv_vec[N]) < tol
            return zeros_tt(T, y_tt.ttv_dims, y_tt.ttv_rks)
        end

        for j in N:-1:2
            nj = y_tt.ttv_dims[j]
            rj_prev = y_tt.ttv_rks[j]
            rj = y_tt.ttv_rks[j+1]

            # Unfold core (L, I, R) as (L) × (I*R). Wide for typical bonds:
            # truncated_svd! takes the LQ + SVD-on-L path and applies Q only to
            # the kept k rows of Vt_l, saving the (min(m,n) − k) unused output
            # columns that direct svd! would compute.
            M = reshape(y_tt.ttv_vec[j], rj_prev, nj * rj)
            sv = truncated_svd!(M, tol_per_bond; rmax=rmax)
            k = sv.k

            # New core (j) is Vt of shape (k, n) — natural reshape to (k, I, R).
            y_tt.ttv_vec[j] = reshape(sv.Vt, k, nj, rj)

            # Absorb U*S into the left core (j-1). Scale U's k columns in place
            # to avoid an (m, k) broadcast allocation.
            U = sv.U
            @inbounds for col in 1:k
                @simd for row in 1:size(U, 1)
                    U[row, col] *= sv.S[col]
                end
            end
            nj_prev = y_tt.ttv_dims[j-1]
            rj_prev_prev = y_tt.ttv_rks[j-1]
            left_core_mat = reshape(y_tt.ttv_vec[j-1], rj_prev_prev * nj_prev, rj_prev)
            y_tt.ttv_vec[j-1] = reshape(left_core_mat * U, rj_prev_prev, nj_prev, k)

            y_tt.ttv_rks[j] = k
            y_tt.ttv_ot[j] = -1 # Right-orthogonal
        end
        y_tt.ttv_ot[1] = 0
        return y_tt

    elseif direction == :right
        # =====================================================================
        # LEFT-TO-RIGHT COMPRESSION SWEEP (Requires Right-Orthogonalization)
        # =====================================================================
        if norm(y_tt.ttv_vec[1]) < tol 
            return zeros_tt(T, y_tt.ttv_dims, y_tt.ttv_rks)
        end
        
        for j in 1:(N-1)
            nj = y_tt.ttv_dims[j]
            rj_prev = y_tt.ttv_rks[j]
            rj = y_tt.ttv_rks[j+1]

            # Unfold core (L, I, R) as (L*I) × (R). Tall for typical bonds:
            # truncated_svd! takes the QR + SVD-on-R path and applies Q only to
            # the kept k columns of U_r.
            M = reshape(y_tt.ttv_vec[j], rj_prev * nj, rj)
            sv = truncated_svd!(M, tol_per_bond; rmax=rmax)
            k = sv.k

            # New core (j) is U of shape (m, k) — natural reshape to (L, I, k).
            y_tt.ttv_vec[j] = reshape(sv.U, rj_prev, nj, k)

            # Absorb S*Vt into the right core (j+1). Scale Vt's rows in place.
            Vt = sv.Vt
            @inbounds for row in 1:k
                sval = sv.S[row]
                @simd for col in 1:size(Vt, 2)
                    Vt[row, col] *= sval
                end
            end

            nj_next = y_tt.ttv_dims[j+1]
            rj_next = y_tt.ttv_rks[j+2]
            right_core_mat = reshape(y_tt.ttv_vec[j+1], rj, nj_next * rj_next)
            y_tt.ttv_vec[j+1] = reshape(Vt * right_core_mat, k, nj_next, rj_next)

            y_tt.ttv_rks[j+1] = k
            y_tt.ttv_ot[j] = 1 # Left-orthogonal
        end
        y_tt.ttv_ot[N] = 0
        return y_tt

    else
        error("Invalid direction. Use direction=:right or direction=:left")
    end
end


"""
returns the rounding of the TT operator
"""
function tt_rounding(A_tto::TToperator{T,N};tol=1e-12,rmax=2^14) where {T,N}
	return ttv_to_tto(tt_rounding(tto_to_ttv(A_tto);tol=tol,rmax=rmax))
end

"""
returns the singular values of the reshaped tensor x[μ_1⋯μ_k;μ_{k+1}⋯μ_d] for all 1≤ k ≤ d
"""
function tt_svdvals(x_tt::TTvector{T,N};tol=1e-14) where {T<:Number,N}
	Σ = Array{Array{Float64,1},1}(undef,N-1)
	y_tt = orthogonalize(x_tt)
	y_rks = r_and_d_to_rks(y_tt.ttv_rks,y_tt.ttv_dims)
	# Unfolding (L*I, R) is a contiguous reshape under (L, I, R) layout.
	core_temp = zeros(T,maximum(y_tt.ttv_dims.*y_tt.ttv_rks[1:end-1]),maximum(y_tt.ttv_rks))
	core_temp[1:y_tt.ttv_rks[1]*y_tt.ttv_dims[1],1:y_tt.ttv_rks[2]] = reshape(y_tt.ttv_vec[1],y_tt.ttv_rks[1]*y_tt.ttv_dims[1],y_tt.ttv_rks[2])
	for j in 1:N-1
		u,s,v = svd(@view(core_temp[1:y_rks[j]*y_tt.ttv_dims[j],1:y_tt.ttv_rks[j+1]]))
		Σ[j],_ = floor(s,tol)
		core_view = reshape(view(core_temp,1:y_rks[j+1]*y_tt.ttv_dims[j+1],1:y_tt.ttv_rks[j+2]),y_rks[j+1],y_tt.ttv_dims[j+1],y_tt.ttv_rks[j+2])
		# M (α,z) × core (z, i2*β) → (α, i2*β). Single gemm into the view.
		let M = Diagonal(s) * v', c = y_tt.ttv_vec[j+1]
			mul!(reshape(core_view, y_rks[j+1], :),
			     M,
			     reshape(c, size(c,1), :))
		end
	end
	return Σ
end

function floor(s::AbstractVector{<:Real},tol;degen_tol=1e-5)
	if tol==0.0
		return s,length(s)
	else
		d = length(s)
		i=d
		weight = zero(eltype(s))
		norm2 = dot(s,s)
		while (i>0) && weight<tol^2*norm2
			weight+=s[i]^2
			i-=1
		end
		i+=1
		while (i<d) && isapprox(1.,s[i+1]/s[i];atol=degen_tol)
			i+=1
		end
		return s[1:i],i
	end
end

function left_compression(A,B;tol=1e-12)
    # Cores have layout (L, I, R). Unfold A as (L_A*I_A) × R_A; unfold B as L_B × (I_B*R_B).
    dim_A = [i for i in size(A)]
    dim_B = [i for i in size(B)]

    U = reshape(A, :, dim_A[3])
    V = reshape(B, dim_B[1], :)
    u,s,v = svd(U*V, full=false)
    s_trunc, _ = floor(s, tol)
    k = length(s_trunc)
    A_new = reshape(u[:, 1:k], dim_A[1], dim_A[2], k)
    B_new = reshape(Diagonal(s_trunc)*v[:, 1:k]', k, dim_B[2], dim_B[3])
    return A_new, B_new
end

"""
parallel compression of the TTvector
TODO refactoring
"""
function tt_compression_par(X::TTvector;tol=1e-14,Imax=2)
    Y = deepcopy(X.ttv_vec) :: Array{Array{Float64,3},1}
    rks = deepcopy(X.ttv_rks) :: Array{Int64}
	d = x_tt.N
    rks_prev = zeros(Integer,d)
    i=0
    while norm(rks-rks_prev)>0.1 && i<Imax
        i+=1
        rks_prev = deepcopy(rks) :: Array{Int64}
        if mod(i,2) == 1
            @threads for k in 1:floor(Integer,d/2)
                Y[2k-1], Y[2k] = left_compression(Y[2k-1], Y[2k], tol=tol)
                rks[2k-1] = size(Y[2k-1],3)
            end
        else
            @threads for k in 1:floor(Integer,(d-1)/2)
                Y[2k], Y[2k+1] = left_compression(Y[2k], Y[2k+1], tol=tol)
                rks[2k] = size(Y[2k],3)
            end
        end
    end
    return TTvector(d,Y,X.ttv_dims,rks,zeros(Integer,d))
end

function tt_compression_par(A::TToperator;tol=1e-14,Imax=2)
	return ttv_to_tto(tt_compression_par(tto_to_ttv(A);tol=tol,Imax=Imax))
end

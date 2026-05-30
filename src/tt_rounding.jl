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
	y_tt = zeros_tt(T,x_tt.ttv_dims,y_rks)
	# Cores have layout (L, I, R). Left-orthogonalization: unfold each core
	# as (L*I, R) — a contiguous column-major reshape — and QR.
	FR = ones(T,1,1)
	yleft_temp = zeros(T, maximum(x_tt.ttv_rks), maximum(x_tt.ttv_dims), maximum(x_tt.ttv_rks))
	for j in 1:i-1
		y_tt.ttv_ot[j]=1
		# FR (αⱼ₋₁, βⱼ₋₁) × core (βⱼ₋₁, iⱼ*αⱼ) → (αⱼ₋₁, iⱼ*αⱼ). Single gemm into the view.
		let view_lhs = view(yleft_temp, 1:y_tt.ttv_rks[j], 1:x_tt.ttv_dims[j], 1:x_tt.ttv_rks[j+1]),
			c = x_tt.ttv_vec[j]
			mul!(reshape(view_lhs, y_tt.ttv_rks[j], :),
			     FR,
			     reshape(c, size(c,1), :))
		end
		F = qr(reshape(yleft_temp[1:y_tt.ttv_rks[j],1:x_tt.ttv_dims[j],1:x_tt.ttv_rks[j+1]], y_tt.ttv_rks[j]*x_tt.ttv_dims[j], :))
		y_tt.ttv_rks[j+1] = size(Matrix(F.Q),2)
		y_tt.ttv_vec[j] = reshape(Matrix(F.Q), y_tt.ttv_rks[j], x_tt.ttv_dims[j], y_tt.ttv_rks[j+1])
		FR = F.R[1:y_tt.ttv_rks[j+1],:]
	end
	# Right-orthogonalization: unfold each core as (L, I*R) and LQ.
	FL = ones(T,1,1)
	(i<x_tt.N) && (yright_temp = zeros(T, maximum(x_tt.ttv_rks), maximum(x_tt.ttv_dims), maximum(y_tt.ttv_rks)))
	for j in d:-1:i+1
		y_tt.ttv_ot[j]=-1
		yright_temp = zeros(T, x_tt.ttv_rks[j], x_tt.ttv_dims[j], y_tt.ttv_rks[j+1])
		# core (αⱼ₋₁*iⱼ, αⱼ) × FL (αⱼ, βⱼ) → (αⱼ₋₁*iⱼ, βⱼ). Single gemm.
		let c = x_tt.ttv_vec[j]
			mul!(reshape(yright_temp, x_tt.ttv_rks[j]*x_tt.ttv_dims[j], :),
			     reshape(c, size(c,1)*size(c,2), :),
			     FL)
		end
		F = lq(reshape(yright_temp[1:x_tt.ttv_rks[j],1:x_tt.ttv_dims[j],1:y_tt.ttv_rks[j+1]], x_tt.ttv_rks[j], :))
		y_tt.ttv_rks[j] = size(Matrix(F.Q),1)
		y_tt.ttv_vec[j] = reshape(Matrix(F.Q), y_tt.ttv_rks[j], x_tt.ttv_dims[j], y_tt.ttv_rks[j+1])
		FL = F.L[:,1:y_tt.ttv_rks[j]]
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
returns a TT representation where the singular values lower than tol are discarded

function tt_rounding(x_tt::TTvector{T,N};tol=1e-12,rmax=2^14) where {T<:Number,N}
	is_leftorthogonal(x_tt) ? y_tt = copy(x_tt) :	y_tt = orthogonalize(x_tt;i=x_tt.N)
	norm(y_tt) < tol && return zeros_tt(T,x_tt.ttv_dims,x_tt.ttv_rks)
	for j in x_tt.N:-1:2
		M = reshape(permutedims(y_tt.ttv_vec[j],[2 1 3]),y_tt.ttv_rks[j],:)
		u,s,v = try
			svd(M, full=false)
		catch e
			e isa LAPACKException ? svd(M, full=false, alg=LinearAlgebra.QRIteration()) : rethrow(e)
		end
		_,k = floor(s[s.>0],tol)
		k = min(k,rmax)
		y_tt.ttv_vec[j] = permutedims(reshape(v'[1:k,:],:,x_tt.ttv_dims[j],y_tt.ttv_rks[j+1]),[2 1 3])
		y_tt.ttv_vec[j-1] = reshape(reshape(y_tt.ttv_vec[j-1],y_tt.ttv_dims[j-1]*y_tt.ttv_rks[j-1],:)*u[:,1:k]*Diagonal(s[1:k]),y_tt.ttv_dims[j-1],y_tt.ttv_rks[j-1],:)
		y_tt.ttv_rks[j] = k
		y_tt.ttv_ot[j] = 1
	end
	y_tt.ttv_ot[1] = 0
	return y_tt
end"""



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

            # Unfold core (L, I, R) as (L) × (I*R) — pure column-major reshape.
            M = reshape(y_tt.ttv_vec[j], rj_prev, nj * rj)

            u, s, v = try
                svd(M, full=false)
            catch e
                e isa LAPACKException ? svd(M, full=false, alg=LinearAlgebra.QRIteration()) : rethrow(e)
            end

            # Enforce maximum rank
            _, k = floor(s, tol_per_bond)
            k = min(k, rmax, sum(s .> 0.0))

            # Update current core (j) with V^T — V_trunc is (k, I*R), reshape to (k, I, R).
            V_trunc = adjoint(@view v[:, 1:k])
            y_tt.ttv_vec[j] = reshape(V_trunc, k, nj, rj)

            # Absorb U*S into the left core (j-1). Unfold (L', I', L) as (L'*I', L).
            US = @view(u[:, 1:k]) .* s[1:k]'
            nj_prev = y_tt.ttv_dims[j-1]
            rj_prev_prev = y_tt.ttv_rks[j-1]

            left_core_mat = reshape(y_tt.ttv_vec[j-1], rj_prev_prev * nj_prev, rj_prev)
            y_tt.ttv_vec[j-1] = reshape(left_core_mat * US, rj_prev_prev, nj_prev, k)

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

            # Unfold core (L, I, R) as (L*I) × (R) — pure column-major reshape.
            M = reshape(y_tt.ttv_vec[j], rj_prev * nj, rj)

            u, s, v = try
                svd(M, full=false)
            catch e
                e isa LAPACKException ? svd(M, full=false, alg=LinearAlgebra.QRIteration()) : rethrow(e)
            end

            _, k = floor(s, tol_per_bond)
            k = min(k, rmax, sum(s .> 0.0))

            # Update current core (j) with U — U is (L*I, k), reshape to (L, I, k).
            y_tt.ttv_vec[j] = reshape(@view(u[:, 1:k]), rj_prev, nj, k)

            # Absorb S*V^T into the right core (j+1).
            SVT = s[1:k] .* adjoint(@view v[:, 1:k])

            nj_next = y_tt.ttv_dims[j+1]
            rj_next = y_tt.ttv_rks[j+2]

            # Right core has layout (L, I, R) — unfold directly as (L) × (I*R).
            right_core_mat = reshape(y_tt.ttv_vec[j+1], rj, nj_next * rj_next)
            new_right_core_flat = SVT * right_core_mat
            y_tt.ttv_vec[j+1] = reshape(new_right_core_flat, k, nj_next, rj_next)

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

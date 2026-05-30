using LinearMaps
using TensorOperations

"""
Implementation based on the presentation in 
Holtz, Sebastian, Thorsten Rohwedder, and Reinhold Schneider. "The alternating linear scheme for tensor optimization in the tensor train format." SIAM Journal on Scientific Computing 34.2 (2012): A683-A713.
"""

function init_H(x_tt::TTvector{T},A_tto::TToperator{T}) where {T<:Number}
	d = x_tt.N
	H = Array{Array{T}}(undef, d)
	H[d] = ones(T,1,1,1)
	for i = d : -1 : 2
		H[i-1] = zeros(T,A_tto.tto_rks[i],x_tt.ttv_rks[i],x_tt.ttv_rks[i])
		x_vec = x_tt.ttv_vec[i]
		A_vec = A_tto.tto_vec[i]
		update_H!(x_vec,A_vec,H[i],H[i-1])
	end
	return H
end

function update_H!(x_vec::Array{T,3},A_vec::Array{T,4},Hi::Array{T,3},Him::Array{T,3}) where T<:Number
	# Vector core (L,I,R); operator core (L,i,j,R). Him: (R_A, L_x, L_x).
	@tensoropt((ϕ,χ), Him[a,α,β] = conj.(x_vec)[α,j,ϕ]*Hi[z,ϕ,χ]*x_vec[β,k,χ]*A_vec[a,j,k,z])
	nothing
end

function init_Hb(x_tt::TTvector{T},b_tt::TTvector{T}) where {T<:Number}
	d = x_tt.N
	H_b = Array{Array{T}}(undef, d) 
	H_b[d] = ones(T,1,1)
	for i = d : -1 : 2
		H_b[i-1] = zeros(T,x_tt.ttv_rks[i],b_tt.ttv_rks[i])
		b_vec = b_tt.ttv_vec[i]
		x_vec = x_tt.ttv_vec[i]
		update_Hb!(x_vec,b_vec,H_b[i],H_b[i-1]) #size(rbim, rim) 
	end
	return H_b
end

function update_Hb!(x_vec::Array{T,3},b_vec::Array{T,3},H_bi::Array{T,2},H_bim::Array{T,2}) where T<:Number
	# H_bim[α, β] = sum_{ϕ, χ, i} H_bi[ϕ, χ] * b_vec[β, i, χ] * conj(x_vec)[α, i, ϕ]. Two gemms.
	# Step 1: bH[β, i, ϕ] = sum_χ b_vec[β, i, χ] * H_bi[χ, ϕ]. Flat (β*i, χ) × (χ, ϕ).
	α, ni, ϕd = size(x_vec)
	β, _, χd = size(b_vec)
	bH = reshape(b_vec, β*ni, χd) * transpose(H_bi)   # (β*i, ϕ)
	# Step 2: H_bim[α, β] = sum_{i, ϕ} conj(x_vec)[α, i, ϕ] × bH[β, i, ϕ].
	#   Flatten conj(x_vec) (α, i*ϕ) × transpose(bH (β, i*ϕ)) → (α, β).
	mul!(H_bim,
	     reshape(conj.(x_vec), α, ni*ϕd),
	     transpose(reshape(bH, β, ni*ϕd)))
	nothing
end

function update_G!(x_vec::Array{T,3},A_vec::Array{T,4},Gi::AbstractArray{T,5},Gip::AbstractArray{T,5}) where T<:Number
	# Gi/Gip layout (L_x_bra, I_bra, L_x_ket, I_ket, R_A). Vector (L,I,R); operator (L,i,j,R).
	@tensor Gip[α,j,β,k,J] = (conj.(x_vec)[ϕ,l,α]*(Gi[ϕ,l,χ,m,L]*x_vec[χ,m,β]))*A_vec[L,j,k,J]
	nothing
end

function update_Gb!(x_vec::Array{T,3},b_vec::Array{T,3},G_bi::AbstractArray{T,3},G_bip::AbstractArray{T,3}) where T<:Number
	# G_b layout (L_x, I, R_b); vector (L,I,R).
	# G_bip[α, i, β] = sum_{χ, j, ϕ} conj(x_vec)[χ, j, α] * G_bi[χ, j, ϕ] * b_vec[ϕ, i, β]. Two gemms.
	χd, jd, αd = size(x_vec)   # x_vec (L_x_prev, I, L_x_curr=α)
	ϕd, _, βd  = size(b_vec)
	# Step 1: Xb[α, ϕ] = sum_{χ, j} conj(x_vec)[χ, j, α] × G_bi[χ, j, ϕ].
	#   Flatten conj(x_vec) (χ*j, α), G_bi (χ*j, ϕ); transpose(X) × G.
	Xb = transpose(reshape(conj.(x_vec), χd*jd, αd)) * reshape(G_bi, χd*jd, ϕd)  # (α, ϕ)
	# Step 2: G_bip[α, i*β] = Xb[α, ϕ] × reshape(b_vec, ϕ, i*β).
	mul!(reshape(G_bip, αd, size(b_vec,2)*βd),
	     Xb,
	     reshape(b_vec, ϕd, size(b_vec,2)*βd))
	nothing
end

#full assemble of matrix K; local core layout (L_x, I, R_x), so K_dims = (L_x, I, R_x).
function K_full(Gi::Array{T,5},Hi::Array{T,3},K_dims::NTuple{3,Int}) where T<:Number
	K = zeros(T,prod(K_dims),prod(K_dims))
	Krshp = reshape(K,(K_dims...,K_dims...))
	# Gi[L_x_bra, I_bra, L_x_ket, I_ket, R_A]; Hi[R_A, R_x_bra, R_x_ket].
	@tensor Krshp[a,b,c,d,e,f] = Gi[a,b,d,e,z]*Hi[z,c,f]
	return K
end

function Ksolve(Gi::Array{T,5},G_bi::Array{T,3},Hi::Array{T,3},H_bi::Array{T,2}) where T<:Number
	K_dims = (size(Gi,1),size(Gi,2),size(Hi,2))  # (L_x, I, R_x)
	K = K_full(Gi,Hi,K_dims)
	# Pb[α1, i, α2] = G_b[α1, i, β] * H_b[α2, β]. Reshape (α1*i, β) × (β, α2) = (α1*i, α2). One gemm.
	Pb = reshape(reshape(G_bi, size(G_bi,1)*size(G_bi,2), :) * transpose(H_bi), K_dims)
	return reshape(K\Pb[:],K_dims)
end

function K_eigmin(Gi::Array{T,5},Hi::Array{T,3},ttv_vec::Array{T,3};it_solver=false,itslv_thresh=256::Int64,maxiter=200::Int64,tol=1e-6::Float64) where T<:Number
	K_dims = (size(Gi,1),size(Gi,2),size(Hi,2))
	if it_solver && prod(K_dims) > itslv_thresh
		H = zeros(T,prod(K_dims))
		function K_matfree(V::AbstractArray{T,1};Gi=Gi::Array{T,5},Hi=Hi::Array{T,3},K_dims=K_dims,H=H::AbstractArray{T,1})
			Hrshp = reshape(H,K_dims)
			@tensoropt((b,c,e,f), Hrshp[a,b,c] = Gi[a,b,d,e,z]*reshape(V,K_dims)[d,e,f]*Hi[z,c,f])
			return H::AbstractArray{T,1}
		end
		r = lobpcg(LinearMap(K_matfree,prod(K_dims);ishermitian = true),false,ttv_vec[:],1;maxiter=maxiter,tol=tol)
		return r.λ[1]::Real, reshape(r.X[:,1],K_dims)::Array{T,3}
	else
		K = K_full(Gi,Hi,K_dims)
		F = eigen(Hermitian(K),1:1)
		return real(F.values[1])::Real,reshape(F.vectors[:,1],K_dims)::Array{T,3}
	end
end

function K_eiggenmin(Gi,Hi,Ki,Li,ttv_vec;it_solver=false,itslv_thresh=2500)
	# Local core layout (L_x, I, R_x); a/d are L_x, b/e are I, c/f are R_x.
	@tensor begin
		K[a,b,c,d,e,f] := Gi[d,e,a,b,z]*Hi[z,f,c]
		S[a,b,c,d,e,f] := Ki[d,e,a,b,z]*Li[z,f,c]
	end
	if it_solver || prod(size(K)[1:3]) > itslv_thresh
		r = lobpcg(reshape(K,prod(size(K)[1:3]),:),reshape(S,prod(size(S)[1:3]),:),false,ttv_vec[:],1;maxiter=500,tol=1e-8)
		return r.λ[1], reshape(r.X[:,1],size(K)[1:3])
	else
		F = eigen(reshape(K,prod(size(K)[1:3]),:),reshape(S,prod(size(K)[1:3]),:),)
		return real(F.values[1]),reshape(F.vectors[:,1],size(K)[1:3])
	end
end

function left_core_move(x_tt::TTvector{T},V::Array{T,3},i::Int,x_rks) where {T<:Number}
	rim,ri = x_rks[i],x_rks[i+1]
	ni = x_tt.ttv_dims[i]

	# V has layout (L=rim, I=ni, R=ri). Unfold (rim, ni*ri) and LQ for right-orthogonal.
	F = lq(reshape(V, rim, :))
	# Move 3.1: site i becomes right-orthogonal.
	x_tt.ttv_vec[i] = reshape(Matrix(F.Q), rim, ni, ri)
	x_tt.ttv_ot[i] = -1
	# Move 3.2: absorb L factor into right rank of site i-1.
	# Reshape core (a*b, z) × Lf (z, c) → (a*b, c). Single gemm.
	Lf = Matrix(F.L)
	cprev = x_tt.ttv_vec[i-1]
	x_tt.ttv_vec[i-1] = reshape(reshape(cprev, size(cprev,1)*size(cprev,2), :) * Lf,
	                            size(cprev,1), size(cprev,2), size(Lf,2))
	x_tt.ttv_ot[i-1] = 0
	return x_tt
end

function right_core_move(x_tt::TTvector{T},V::Array{T,3},i::Int,x_rks) where {T<:Number}
	rim,ri = x_rks[i],x_rks[i+1]
	ni = x_tt.ttv_dims[i]

	# V layout (L=rim, I=ni, R=ri). Unfold (rim*ni, ri) and QR for left-orthogonal.
	F = qr(reshape(V, rim*ni, :))
	# Move 3.1: site i becomes left-orthogonal.
	x_tt.ttv_vec[i] = reshape(Matrix(F.Q)[:, 1:ri], rim, ni, ri)
	x_tt.ttv_ot[i] = 1
	# Move 3.2: absorb R factor into left rank of site i+1.
	# Rf (a, z) × reshape core (z, b*c) → (a, b*c). Single gemm.
	Rf = Matrix(F.R)[1:ri, :]
	cnext = x_tt.ttv_vec[i+1]
	x_tt.ttv_vec[i+1] = reshape(Rf * reshape(cnext, size(cnext,1), :),
	                            size(Rf,1), size(cnext,2), size(cnext,3))
	x_tt.ttv_ot[i+1] = 0
	return x_tt
end


"""
Solve Ax=b using the ALS algorithm where A is given as `TToperator` and `b`, `tt_start` are `TTvector`.
The ranks of the solution is the same as `tt_start`.
`sweep_count` is the number of total sweeps in the ALS.
"""
function als_linsolv(A :: TToperator{T}, b :: TTvector{T}, tt_start :: TTvector{T} ;sweep_count=2,it_solver=false,r_itsolver=5000) where {T<:Number}
	# als finds the minimum of the operator J:1/2*<Ax,Ax> - <x,b>
	# input:
	# 	A: the tensor operator in its tensor train format
	#   b: the tensor in its tensor train format
	#	tt_start: start value in its tensor train format
	#	opt_rks: rank vector considered to be optimal enough
	# output:
	#	tt_opt: stationary point of J up to tolerated rank opt_rks
	# 			in its tensor train format

	d = A.N
	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = copy(tt_start.ttv_rks)

	# Initialize the arrays of G and G_b
	G = Array{Array{T}}(undef, d)
	G_b = Array{Array{T}}(undef, d)

	# G layout (L_x, I, L_x, I, R_A); G_b layout (L_x, I, R_b).
	for i in 1:d
		G[i] = zeros(T,rks[i],dims[i],rks[i],dims[i],A.tto_rks[i+1])
		G_b[i] = zeros(rks[i],dims[i],b.ttv_rks[i+1])
	end
	# A.tto_vec[1] has layout (1, i, j, R_A); slice the L=1 leg.
	G[1] = reshape(A.tto_vec[1][1,:,:,:], 1, dims[1], 1, dims[1], :)
	G_b[1] = reshape(b.ttv_vec[1], 1, dims[1], :)

	#Initialize H and H_b
	H = init_H(tt_opt,A)
	H_b = init_Hb(tt_opt,b)

	nsweeps = 0 #sweeps counter

	while nsweeps < sweep_count
		nsweeps+=1
		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $d")
			# Define V as solution of K*x=Pb in x
			V = Ksolve(G[i],G_b[i],H[i],H_b[i])
			tt_opt = right_core_move(tt_opt,V,i,rks)
			#update G,G_b
			update_G!(tt_opt.ttv_vec[i],A.tto_vec[i+1],G[i],G[i+1])
			update_Gb!(tt_opt.ttv_vec[i],b.ttv_vec[i+1],G_b[i],G_b[i+1])
		end

		if nsweeps == sweep_count
			return tt_opt
		else
			nsweeps+=1
			# Second half sweep
			for i = d:(-1):2
				println("Backward sweep: core optimization $i out of $d")
				# Define V as solution of K*x=Pb in x
				V = Ksolve(G[i],G_b[i],H[i],H_b[i])
				tt_opt = left_core_move(tt_opt,V,i,rks)
#				println(norm(tt_opt.ttv_vec[i-1]))
				update_H!(tt_opt.ttv_vec[i],A.tto_vec[i],H[i],H[i-1])
				update_Hb!(tt_opt.ttv_vec[i],b.ttv_vec[i],H_b[i],H_b[i-1])
			end
		end
	end
	return tt_opt
end

"""
Returns the lowest eigenvalue of A by minimizing the Rayleigh quotient in the ALS algorithm.

The ranks can be increased in the course of the ALS: if `sweep_schedule[k] ≤ i <sweep_schedule[k+1]` is the current number of sweeps then the ranks is given by `rmax_schedule[k]`.
"""
function als_eigsolv(A :: TToperator{T},
	 tt_start :: TTvector{T} ; #TT initial guess
	 sweep_schedule=[2]::Array{Int64,1}, #Number of sweeps for each bond dimension in rmax_schedule
	 rmax_schedule=[maximum(tt_start.ttv_rks)]::Array{Int64,1}, #bond dimension at each sweep
	 noise_schedule=zeros(length(rmax_schedule))::Array{Float64,1}, #noise at each bond dimension increase
	 it_solver=false::Bool, #linear solver for the microstep
	 itslv_thresh=1024::Int64, #switch from full to iterative
	 maxiter=200::Int64, #maximum of iterations for the iterative solver
	 linsolv_tol=1e-8::Float64) where {T<:Number} #tolerance of the iterative linear solver
	@assert(length(rmax_schedule)==length(sweep_schedule)==length(noise_schedule),"Sweep schedule error")	
	d = A.N
	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	E = zeros(Float64,2d*(sweep_schedule[end]+1)) #output eigenvalue
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = copy(tt_start.ttv_rks)

	# G layout (L_x, I, L_x, I, R_A).
	G = Array{Array{T}}(undef, d)
	for i in 1:d
		G[i] = zeros(T,rks[i],dims[i],rks[i],dims[i],A.tto_rks[i+1])
	end
	G[1] = reshape(A.tto_vec[1][1,:,:,:], 1, dims[1], 1, dims[1], :)

	#Initialize H and H_b
	H = init_H(tt_opt,A)

	nsweeps = 0 #sweeps counter
	i_schedule,i_μit = 1,0
	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1
		println("Macro-iteration $nsweeps; bond dimension $(rmax_schedule[i_schedule])")
		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				return E[1:i_μit]::Array{Float64,1},tt_opt::TTvector{T}
			else
				tt_opt = tt_up_rks(tt_opt,rmax_schedule[i_schedule];ϵ_wn=noise_schedule[i_schedule])
				tt_opt = orthogonalize(tt_opt)
				H = init_H(tt_opt,A)
				for i in 1:d-1
					# G layout (L_x, I, L_x, I, R_A).
					Gtemp = zeros(tt_opt.ttv_rks[i+1],dims[i+1],tt_opt.ttv_rks[i+1],dims[i+1],A.tto_rks[i+2])
					Gtemp[1:size(G[i+1],1),1:size(G[i+1],2),1:size(G[i+1],3),1:size(G[i+1],4),1:size(G[i+1],5)] = G[i+1]
					G[i+1] = Gtemp
				end
			end
		end
		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $(d-1)")
			# Define V as solution of K*x=Pb in x
			i_μit += 1
			E[i_μit],V = K_eigmin(G[i],H[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh,maxiter=maxiter,tol=linsolv_tol)
			println("Eigenvalue: $(E[i_μit])")
			tt_opt = right_core_move(tt_opt,V,i,tt_opt.ttv_rks)
			#update G
			update_G!(tt_opt.ttv_vec[i],A.tto_vec[i+1],G[i],G[i+1])
		end

		# Second half sweep
		for i = d:(-1):2
			println("Backward sweep: core optimization $(d+1-i) out of $(d-1)")
			# Define V as solution of K*x=Pb in x
			i_μit += 1
			E[i_μit],V = K_eigmin(G[i],H[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh,maxiter=maxiter,tol=linsolv_tol)
			println("Eigenvalue: $(E[i_μit])")
			tt_opt = left_core_move(tt_opt,V,i,tt_opt.ttv_rks)
			update_H!(tt_opt.ttv_vec[i],A.tto_vec[i],H[i],H[i-1])
		end
	end
	return E[1:i_μit]::Array{Float64,1},tt_opt::TTvector{T}
end

"""
returns the smallest eigenpair Ax = Sx
"""
function als_gen_eigsolv(A :: TToperator{T}, S::TToperator{T}, tt_start :: TTvector{T} ; sweep_schedule=[2],rmax_schedule=[maximum(tt_start.ttv_rks)],tol=1e-10,it_solver=false,itslv_thresh=2500) where {T<:Number}
	d = A.N
	# Initialize the to be returned tensor in its tensor train format
	tt_opt = orthogonalize(tt_start)
	dims = tt_start.ttv_dims
	E = zeros(Float64,d*sweep_schedule[end]) #output eigenvalue
	# Define the array of ranks of tt_opt [r_0=1,r_1,...,r_d]
	rks = tt_start.ttv_rks

	# Initialize the arrays of G and K
	G = Array{Array{T}}(undef, d)
	K = Array{Array{T}}(undef, d) 

	# G/K layout (L_x, I, L_x, I, R_op).
	for i in 1:d
		G[i] = zeros(rks[i],dims[i],rks[i],dims[i],A.tto_rks[i+1])
		K[i] = zeros(rks[i],dims[i],rks[i],dims[i],S.tto_rks[i+1])
	end
	G[1] = reshape(A.tto_vec[1][1,:,:,:], 1, dims[1], 1, dims[1], :)
	K[1] = reshape(S.tto_vec[1][1,:,:,:], 1, dims[1], 1, dims[1], :)

	#Initialize H and H_b
	H = init_H(tt_opt,A)
	L = init_H(tt_opt,S)

	nsweeps = 0 #sweeps counter
	i_schedule,i_μit = 1,0
	while i_schedule <= length(sweep_schedule) 
		nsweeps+=1
		if nsweeps == sweep_schedule[i_schedule]
			i_schedule+=1
			if i_schedule > length(sweep_schedule)
				return E[1:i_μit],tt_opt
			else
				tt_opt = tt_up_rks(tt_opt,rmax_schedule[i_schedule])
				for i in 1:d-1
					Htemp = zeros(tt_opt.ttv_rks[i],tt_opt.ttv_rks[i],A.tto_rks[i])
					Ltemp = zeros(tt_opt.ttv_rks[i],tt_opt.ttv_rks[i],S.tto_rks[i])
					Htemp[1:size(H[i],1),1:size(H[i],2),1:size(H[i],3)] = H[i] 
					Ltemp[1:size(L[i],1),1:size(L[i],2),1:size(L[i],3)] = L[i] 
					H[i] = Htemp
					L[i] = Ltemp
				end
			end
		end

		# First half sweep
		for i = 1:(d-1)
			println("Forward sweep: core optimization $i out of $d")

			# If i is the index of the core matrices do the optimization
			if tt_opt.ttv_ot[i] == 0
				# Define V as solution of K*x=Pb in x
				i_μit += 1
				E[i_μit],V = K_eiggenmin(G[i],H[i],K[i],L[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
				println("Eigenvalue: $(E[i_μit])")
				tt_opt = right_core_move(tt_opt,V,i,rks)
			end

			#update G and K
			update_G!(tt_opt.ttv_vec[i],A.tto_vec[i+1],G[i],G[i+1])
			update_G!(tt_opt.ttv_vec[i],S.tto_vec[i+1],K[i],K[i+1])
		end

		# Second half sweep
		for i = d:(-1):2
			println("Backward sweep: core optimization $i out of $d")
			# Define V as solution of K*x=Pb in x
			i_μit += 1
			E[i_μit],V = K_eiggenmin(G[i],H[i],K[i],L[i],tt_opt.ttv_vec[i];it_solver=it_solver,itslv_thresh=itslv_thresh)
			println("Eigenvalue: $(E[i_μit])")
			tt_opt = left_core_move(tt_opt,V,i,rks)
			update_H!(tt_opt.ttv_vec[i],A.tto_vec[i],H[i],H[i-1])
			update_H!(tt_opt.ttv_vec[i],S.tto_vec[i],L[i],L[i-1])
		end
	end
end
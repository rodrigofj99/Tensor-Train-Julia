"""
Gradient descent with fixed step and periodic TT rounding
"""

function gradient_fixed_step(A,b,α;x0=copy(b), Imax=100, tol_gd=1e-6, i_trunc = 5, eps_tt = 1e-4, r_tt = 512, rand_rounding=false, verbose=false, ℓ=0)
    i=1
    x=copy(x0)
    x_rks = x.ttv_rks
    rmax = min(r_tt,maximum(A.tto_rks .* x.ttv_rks))
    p_rks = x_rks
    rand_rounding ? p = ttrand_rounding(A,x,b;rks=x_rks,rmax=r_tt,ℓ) : p = A*x-b
    resid = zeros(Imax)
    resid[1] = norm(p)
    it_trunc = 1
    while i<Imax && resid[i] > (tol_gd+eps_tt)*resid[1]
        i+=1
        x = x - α*p
        rand_rounding ? p = ttrand_rounding(A,x,b;rks=p_rks,rmax=r_tt,ℓ) : p = A*x-b
        if verbose
            println("Iteration: "*string(i))
            println("TT rank p: "*string(maximum(p.ttv_rks)))
            println("TT rank x: "*string(maximum(x.ttv_rks))*"\n")
        end
        if (it_trunc == i_trunc) || (max(maximum(p.ttv_rks),maximum(x.ttv_rks)) > r_tt)
            if rand_rounding
                x = ttrand_rounding(x;rks=x_rks,rmax=r_tt,ℓ)
            end
            x = tt_rounding(x;tol = eps_tt,rmax=r_tt)
            p = tt_rounding(p;tol = eps_tt,rmax=r_tt)
            x_rks = x.ttv_rks
            p_rks = p.ttv_rks
            it_trunc = 1
        else 
            it_trunc +=1 
        end
        resid[i] = norm(p)
    end
    return x, resid[1:i]
 end

"""
TT version of the restarted GMRES algorithm
"""

#takes a hessenberg matrix and returns q,H resp. orthogonal and upper triangular such that q^T H_out = H_in
function qr_hessenberg(H)
    m = size(H,2)
    q = Matrix{Float64}(I,m+1,m+1)
    T = H
    for i in 1:m
        R = Matrix{Float64}(I,m+1,m+1)
        s = T[i+1,i]/sqrt(T[i,i]^2+T[i+1,i]^2)
        c = T[i,i]/sqrt(T[i,i]^2+T[i+1,i]^2)
        R[i:(i+1),i:(i+1)] = [c s; -s c]
        q = R*q
        T = R*T
    end
    return q,T
end

function tt_gmres(A::TToperator,b::TTvector,x0::TTvector;Imax=500,tol=1e-8,m=30,hist=false,γ_list=Float64[],rmax=256)
    V = Array{TTvector}(undef,m)
    W = Array{TTvector}(undef,m)
    H = zeros(m+1,m)
    r0 = tt_rounding(b-A*x0,tol=tol) 
    β = norm(r0)
    V[1] = 1/β*r0
    W[1] = tt_rounding(A*V[1],tol=tol) 
    H[1,1] = dot(W[1],V[1])
    W[1] = tt_rounding(W[1]-H[1,1]*V[1],tol=tol)
    H[2,1] = norm(W[1])
    q,r = qr_hessenberg(H[1:2,1])
    γ = abs(β*q[2,1]) #γ = \| Ax_j-b \|_2
    if hist
        γ_list = vcat(γ_list,γ)
    end
    j = 1
    if Imax <=0 || isapprox(H[j+1,j],0.,atol=tol) || isapprox(γ,0,atol=tol)
        if hist
            return x0,γ_list,H[2,1]
        else
            return x0,[γ],H[2,1]
        end
    else
        while j <= min(m-1,Imax) && !isapprox(H[j+1,j],0.,atol=tol) && !isapprox(γ,0,atol=tol)
            V[j+1] = 1/H[j+1,j]*W[j]
            j+=1 
            W[j] = A*V[j]
            for i in 1:j
                H[i,j] = dot(W[j],V[i])
                W[j] = W[j]-H[i,j]*V[i]
            end
            W[j] = tt_rounding(W[j],tol=tol)  
            H[j+1,j] = norm(W[j])
            q,r = qr_hessenberg(@view H[1:(j+1),1:j])
            γ = abs(β*q[j+1,1])
            if hist
                γ_list = vcat(γ_list,γ)
            end
        end
        z = r[1:j,1:j]\q[1:j,1]
        for i in 1:j
            x0 = x0+β*z[i]*V[i]
        end
        x0 = tt_rounding(x0,tol=tol)
        return tt_gmres(A,b,x0,tol=tol,Imax=Imax-j,m=m,hist=hist,γ_list=γ_list)
    end
end

function tt_cg(A::TToperator,b::TTvector,x0::TTvector;Imax=500,tol=1e-8)
    p = tt_compression_par(tt_add(b,mult_a_tt(-1.0,tt_compression_par(mult(A,x0)))))
    r = p
    j=1
    res= zeros(Imax)
    res[1] = sqrt(tt_dot(p,p))
    while j < Imax && res[j]>tol
        Ap = mult(A,p)
        Ap = tt_compression_par(Ap)
        a = res[j]^2/tt_dot(p,Ap)
        x0 = tt_add(x0,mult_a_tt(a,p))
        x0 = tt_compression_par(x0)
        r = tt_add(r,mult_a_tt(-a,Ap))
        r = tt_compression_par(r)
        res[j+1] = sqrt(tt_dot(r,r))
        p = tt_add(r,mult_a_tt(res[j+1]^2/res[j]^2,p))
        p = tt_compression_par(p)
        j+=1
    end
    return x0, res[1:j]
end

"""
A_k : n_k x n_k x R_{k-1} x R_k
X_k : n_k x r^X_{k-1} x r^X_k
B_k : n_k x r^B_{k-1} x r^B_k
Ql : R_{k-1} r^X_{k-1} x r^B_{k-1}
Qr : R_k r^X_k x r^B_k
"""
function init_core(A_k,dim_X,B_k,Ql,Qr)
    X_k = zeros(dim_X...)
    #B right shape
    B = reshape(B_k,:,size(B_k,3)) # n_k r^B_{k-1} x r^B_k
    B = reshape(B*(Qr'),size(B_k,1),size(B_k,2),size(A_k,4),size(X_k,3)) #n_k x r^B_{k-1} x R_k x r^X_k
    B = permutedims(B,[2,1,3,4]) #r^B_{k-1} x n_k x R_k x r^X_k
    B = Ql*reshape(B,size(B,1),:) #R_{k-1} r^X_{k-1} x n_k R_k r^X_k
    B = permutedims(reshape(B,size(A_k,3),size(X_k,2),size(A_k,1),size(A_k,4),size(X_k,3)),[3,1,4,2,5]) #n_k x R_{k-1} x R_k x r^X_{k-1} x r^X_k
    B = reshape(B,size(A_k,1)*size(A_k,3)*size(A_k,4),dim_X[2]*dim_X[3]) #n_k R_{k-1} R_k x r^X_{k-1} r^X_k
    A = reshape(permutedims(A_k,[1,3,4,2]),size(A_k,1)*size(A_k,3)*size(A_k,4),size(A_k,2))
    ua,sa,va = svd(A)
    X = va*inv(Diagonal(sa))*ua'*B
    println(norm(A*X-B))
    return reshape(X,dim_X[1],dim_X[2],dim_X[3])
end


function rand_norm(n,m) #m>=n
    A = randn(n,m)
    for i in 1:m
        A[:,i] = A[:,i]./norm(A[:,i])
    end
    return A
end

function rand_struct_orth(r_A,r_X,r_b)
    A = zeros(r_X,r_A,r_b)
    q1 = rand_norm(r_A,r_b)
    q2 = rand_orthogonal(r_b,r_X)
    for ia in 1:r_A
        for ix in 1:r_X
            for ib in 1:r_b
                A[ix,ia,ib] = q1[ia,ib]*q2[ib,ix]
            end
        end
    end
    return reshape(A,r_A*r_X,r_b)
end

function init(A::TToperator,b::TTvector,opt_rks)
    @assert(A.tto_dims == b.ttv_dims,DimensionMismatch)
    d = length(A.tto_dims)
    opt_rks = vcat([1],opt_rks)
    Q_list = Array{Array{Float64},1}(undef,d+1)
    ttvec = Array{Array{Float64},1}(undef,d)
    Q_list[1] = [1]
    Q_list[d+1] = [1]
    for k in 1:(d-1)
        Q_list[k+1] = rand_struct_orth(A.tto_rks[k],opt_rks[k+1],b.ttv_rks[k])
    end
    for k in 1:d
        ttvec[k] = init_core(A.tto_vec[k],[A.tto_dims[k],opt_rks[k],opt_rks[k+1]],b.ttv_vec[k],Q_list[k],Q_list[k+1])
    end
    return TTvector(ttvec,A.tto_dims,opt_rks[2:(d+1)],ones(Int64,d))
end

#automatically determines the initial tt ranks
function init_adapt(A::TToperator,b::TTvector)
    d = length(A.tto_dims)
    opt_rks = ones(Int64,d)
    for k in 1:(d-1)
        opt_rks[k] = lcm(A.tto_rks[k],b.ttv_rks[k])/A.tto_rks[k]
    end
    println(opt_rks)
    return init(A,b,opt_rks)
end

function arnoldi(A::TToperator{T,N},m,V;ε_tt=1e-6,rmax=256) where {T,N}
    H = UpperHessenberg(zeros(T,m+1,m+1))
    V[1] = V[1]/norm(V[1])
    for j in 1:m 
      w = dot_randrounding(A,V[j])
      for i in 1:j 
        H[i,j] = dot(V[i],w) #modified GS
        w = w-H[i,j]*V[i]
      end
      #println("TT rank: $(maximum(w.ttv_rks))")
      w = ttrand_rounding(w)
      #println("TT rank after rand_rounding: $(maximum(w.ttv_rks))")
      w = tt_rounding(w;tol=ε_tt,rmax=rmax)
      #println("TT rank after tt_rounding: $(maximum(w.ttv_rks))")
      H[j+1,j] = norm(w)
      V[j+1] = 1/H[j+1,j]*w
    end
    return H[1:m,1:m],V,H[m+1,m] 
end

function eig_arnoldi(A::TToperator,m,v::TTvector{T,N};Imax=100,ε=1e-6,ε_tt=1e-4,rmax=256,which=:LM,σ=zero(eltype(v)),history=false) where {T,N}
    i = 1
    λ = zero(eltype(v))
    V = Array{TTvector{T,N},1}(undef,m+1)
    V[1] = v
    H,V,h = arnoldi(A,m,V,rmax=2rmax)
    F = eigen(H+σ*I)
    if which==:LM
        k = argmax(abs.(F.values))
    else 
        k = argmin(abs.(F.values))
    end
    λ = F.values[k]
    v = ttrand_rounding(V[1:m]*F.vectors[:,k];rks=2*v.ttv_rks) #largest eigenvalue
    v = tt_rounding(v,tol=ε_tt,rmax=rmax)
    hist = eltype(v)[]
    while (i<Imax) && abs(h)>ε
        println("Arnoldi iteration $i")
      if eltype(v) == ComplexF64
        A = complex(A)
      end
      H,V,h = arnoldi(A,m,V;ε_tt=ε_tt,rmax=2rmax)
      F = eigen(H+σ*I)
      if which==:LM
        k = argmax(abs.(F.values))
      else 
        k = argmin(abs.(F.values))
      end
      λ = F.values[k]
      v = ttrand_rounding(V[1:m]*F.vectors[:,k];rks=2*v.ttv_rks) #largest eigenvalue
      v = tt_rounding(v,tol=ε_tt,rmax=rmax)
      if history 
        push!(hist,norm(A*v-(λ-σ)*v))
      end
      println("Current eigenvalue: $(λ-σ)")
      println("Arnoldi residual $h")
      i+=1
    end
    return λ-σ,v,hist
end

function inner_davidson(A,u,uhat,θ,V,W,H,m,r,prec;which=:LM,rmax=256,ε_tt=1e-6,σ=0.0,ε=1e-6)
    for j in 1:m-1
        V[j+1] = als_linsolv(prec-θ*id_tto(A.N),r,r) #1-site ALS
#        V[j+1] = dmrg_linsolv(prec-θ*id_tto(A.N),r,r,N=1,rmax=rmax) #1-site ALS
        for i in 1:j
            V[j+1] = V[j+1] -dot(V[i],V[j+1])*V[i] #modified GS
        end
        V[j+1] = ttrand_rounding(V[j+1])
        V[j+1] = tt_rounding(V[j+1],rmax=rmax,tol=ε_tt)
        V[j+1] = V[j+1]/norm(V[j+1])
        W[j+1] = tt_rounding(dot_randrounding(A,V[j+1]),tol=ε_tt,rmax=rmax)
        for i in 1:j+1
            H[i,j+1] = dot(V[i],W[j+1])
            H[j+1,i] = dot(V[j+1],W[i])
        end
        F = eigen(H[1:j+1,1:j+1]+σ*I)
        if which==:LM
            k = argmax(abs.(F.values))
        else #lowest magnitude
            k = argmin(abs.(F.values))
        end
        θ = F.values[k]-σ
        println("Eigenvalue: $θ")
        if eltype(F.vectors[:,k]) == ComplexF64
            V = complex(V)
            W = complex(W)
            H = complex(H)
        end
        u = tt_rounding(V[1:j+1]*F.vectors[:,k],tol=ε_tt,rmax=rmax)
        uhat = tt_rounding(W[1:j+1]*F.vectors[:,k],tol=ε_tt,rmax=rmax)
        r = tt_rounding(uhat-θ*u,tol=ε_tt,rmax=rmax)
        println("Norm residual: $(norm(r))")
        if norm(r)<ε
            break
        end
    end
    return u,uhat,θ
  end

  function davidson(A::TToperator,m,v::TTvector{T,N};Imax=100,ε=1e-6,ε_tt=1e-4,rmax=256,which=:LM,σ=0.0,prec=I) where {T,N}
    res = float(T)[]
    V = Array{TTvector{T,N},1}(undef,m)
    W = Array{TTvector{T,N},1}(undef,m)
    H = zeros(T,m,m)
    v = v/norm(v)
    vhat = tt_rounding(dot_randrounding(A,v),tol=ε_tt,rmax=rmax)
    V[1] = v
    W[1] = vhat 
    θ = dot(V[1],W[1])
    H[1,1] = θ
    r = tt_rounding(W[1]-H[1,1]*V[1],tol=ε_tt,rmax=rmax)
#    println(norm(r))
#    inner_davidson!(A,v,vhat,θ,V,W,H,m,r,prec;which=which,σ=σ,rmax=rmax,ε_tt=ε_tt,ε=ε)
    push!(res,norm(r))
    i = 1
    while (i<Imax) && res[i] > ε
      V[1] = v
      W[1] = vhat
      H[1,1] = θ
      v,vhat,θ = inner_davidson(A,v,vhat,θ,V,W,H,m,r,prec;which=which,σ=σ,rmax=rmax,ε_tt=ε_tt,ε=ε)
      push!(res,norm(dot_randrounding(A,v)-θ*v))
      i+=1
    end
    return θ, v, res
  end

# ── Sketched Rayleigh-Ritz eigensolver ────────────────────────────────────────


"""
    expand_basis!(H, B_window, W_B_window, B_sketch_window, rks; ...) -> b_new, sketch_b_new, sketch_Hb_prev

Add one vector to a sketch-orthogonal Krylov basis, reusing all precomputed sketch data.
Mutates `B_window`, `W_B_window`, and `B_sketch_window` (appends `b_new` and its sketch,
drops the oldest entry if the sliding window exceeds `k_trunc`).

The sketch of `H*B_window[end]` is computed at the start of the call.
Its boundary vector `sketch_Hb_prev = Ω H b_{j-1}` is returned for the caller to store.

# Arguments
- `H`: TT operator
- `B_window`: sliding window of the last `k_trunc` basis TTvectors
- `W_B_window`: full recursive sketches `{W_B[k]}` for each vector in `B_window`
- `B_sketch_window`: boundary sketch vectors `Ω bₗ ∈ ℝˢ` for each vector in `B_window`
- `rks`: sketch rank vector (length N+1)
- `k_trunc`: maximum window size (default: current window length)
"""
function expand_basis!(H::TToperator{T,N},
                      B_window::Vector{TTvector{T,N}},
                      W_B_window::Vector{Vector{Matrix{T}}},
                      B_sketch_window::Vector{Vector{T}},
                      rmax::Int,
                      s::Int;
                      orthogonal::Bool=true,
                      block_rks=8,
                      seed::Int=1234,
                      prec=identity,
                      k_trunc::Int=length(B_window)) where {T,N}
    # rks = ones(Int, N+1); rks[1] = max(s, rmax); rks[2:N] .= rmax

    b_prev   = B_window[end]
    dims     = b_prev.ttv_dims
    # Right preconditioning: the operator image is H·(M⁻¹ b_prev). `prec` preserves ranks,
    # so only the operator-image construction below uses `w_prev`; the Gram–Schmidt
    # subtraction keeps the stored (unpreconditioned) basis vectors in `B_window`.
    w_prev   = prec(b_prev)

    # ── Sketch of H*w_prev

    sketch_Hb_prev, WHb_raw, s_rks = tt_combined_sketch(T, H, w_prev, 2rmax, s; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)
    W_HB           = [reshape(WHb_raw[k], w_prev.ttv_rks[k]*H.tto_rks[k], s_rks[k]) for k=1:N+1]

    # rks = [s; rmax*ones(Int, N-1); 1]
    # _ = tt_recursive_sketch(T, H, b_prev, rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)

    # ── Sketched GS coefficients (O(s · k_trunc), no TT ops) ──────────────
    h    = hcat(B_sketch_window...) \ sketch_Hb_prev
    m    = 1 + length(B_window)         # terms: H b_{j-1}, then each b_l

    # ── QR sweep (left-to-right, using precomputed W matrices) ────────────
    vec_out = Vector{Array{T,3}}(undef, N)
    out_rks = ones(Int, N+1)
    ot = zeros(Int, N)

    # Initial left contractions at site 1. Cores have layout (L, I, R); L=1 at boundary.
    b1 = reshape(w_prev.ttv_vec[1], dims[1], w_prev.ttv_rks[2])
    H1 = reshape(H.tto_vec[1], dims[1], dims[1], H.tto_rks[2])
    # @tensor here would expand to 2 permutes + 1 gemm anyway — keep the macro for clarity.
    @tensor Ay₁_tmp[i₁,α₂,β₂] := H1[i₁,j₁,β₂] * b1[j₁,α₂]
    Ay₁ = reshape(Ay₁_tmp, 1, dims[1], w_prev.ttv_rks[2], H.tto_rks[2])

    Yₖ    = Vector{Array{T,3}}(undef, m)
    Yₖ[1] = reshape(Ay₁, 1, dims[1], w_prev.ttv_rks[2] * H.tto_rks[2])
    for (i,b) in enumerate(B_window)
        Yₖ[1+i] = -h[i] .* b.ttv_vec[1]
    end

    for k = 1:N-1
        # Sketch contraction: Zₖ accumulates all m terms. Layout (L, I, R).
        # Each term is reshape(Y, ρₖ*iₖ, αₖ₊₁) × W = (ρₖ*iₖ, ρₖ₊₁). One gemm per term.
        Zₖ = zeros(T, out_rks[k], dims[k], s_rks[k+1])
        Zₖ_flat = reshape(Zₖ, out_rks[k]*dims[k], s_rks[k+1])
        mul!(Zₖ_flat,
             reshape(Yₖ[1], out_rks[k]*dims[k], :),
             W_HB[k+1],
             true, true)
        for (i,W) in enumerate(W_B_window)
            mul!(Zₖ_flat,
                 reshape(Yₖ[1+i], out_rks[k]*dims[k], :),
                 W[k+1],
                 true, true)
        end

        Q, _ = qr!(reshape(Zₖ, out_rks[k]*dims[k], s_rks[k+1]))
        Q    = Matrix(Q)
        out_rks[k+1] = size(Q, 2)
        vec_out[k]   = reshape(Q, out_rks[k], dims[k], out_rks[k+1])
        ot[k] = 1

        # Propagate left contractions to site k+1
        Yₖ₊₁ = Vector{Array{T,3}}(undef, m)

        y₁ₖ    = reshape(Yₖ[1], out_rks[k], dims[k], w_prev.ttv_rks[k+1], H.tto_rks[k+1])
        Ay₁ₖ₊₁ = zeros(T, out_rks[k+1], dims[k+1], w_prev.ttv_rks[k+2], H.tto_rks[k+2])
        @tensoropt (ρₖ,ρₖ₊₁,αₖ₊₁,βₖ₊₁,αₖ₊₂,βₖ₊₂) Ay₁ₖ₊₁[ρₖ₊₁,iₖ₊₁,αₖ₊₂,βₖ₊₂] = y₁ₖ[ρₖ,iₖ,αₖ₊₁,βₖ₊₁] * vec_out[k][ρₖ,iₖ,ρₖ₊₁] * w_prev.ttv_vec[k+1][αₖ₊₁,jₖ₊₁,αₖ₊₂] * H.tto_vec[k+1][βₖ₊₁,iₖ₊₁,jₖ₊₁,βₖ₊₂]
        Yₖ₊₁[1] = reshape(Ay₁ₖ₊₁, out_rks[k+1], dims[k+1],
                           w_prev.ttv_rks[k+2] * H.tto_rks[k+2])

        for (i,b) in enumerate(B_window)
            Yₖ₊₁[1+i] = zeros(T, out_rks[k+1], dims[k+1], b.ttv_rks[k+2])
            # 3-tensor env propagation — keep @tensoropt; replacement needs 2 large gemms + permutes.
            @tensoropt (αₖ₊₁,αₖ₊₂,ρₖ,ρₖ₊₁) Yₖ₊₁[1+i][ρₖ₊₁,iₖ₊₁,αₖ₊₂] =
                Yₖ[1+i][ρₖ,iₖ,αₖ₊₁] * vec_out[k][ρₖ,iₖ,ρₖ₊₁] *
                b.ttv_vec[k+1][αₖ₊₁,iₖ₊₁,αₖ₊₂]
        end

        Yₖ = Yₖ₊₁
    end

    # Last core: direct sum of all left contractions (all shape (out_rks[N], dims[N], 1)).
    vec_out[N] = reshape(sum(Yₖ), out_rks[N], dims[N], 1)

    b_new = tt_rounding(TTvector{T,N}(N, vec_out, dims, out_rks, ot); rmax=rmax)

    # ── W_B_new: full sketch of b_new (enters the sliding window) ─────────
    sketch_b_new, W_B_new, _ = tt_combined_sketch(T, b_new, 2rmax, s; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)
    
    # _ = tt_recursive_sketch(T, b_new, rks; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)

    β = norm(sketch_b_new)
    sketch_b_new ./= β
    for i = 2:findfirst(isequal(0), b_new.ttv_ot)
        W_B_new[i] ./= β
    end
    b_new /= β

    push!(B_window,        b_new);         length(B_window)        > k_trunc && popfirst!(B_window)
    push!(B_sketch_window, sketch_b_new);  length(B_sketch_window) > k_trunc && popfirst!(B_sketch_window)
    push!(W_B_window,      W_B_new);       length(W_B_window)      > k_trunc && popfirst!(W_B_window)

    return b_new, sketch_b_new, sketch_Hb_prev
end

"""
    sketched_rr(H, B, C, D, rmax; ...) -> λs, X, res_sketch, res_sample

Stable sketched Rayleigh-Ritz (Tropp & Nakatsukasa 2022).

Given the Krylov basis `B` and sketch matrices `C = Ω B`, `D = Ω H B` (both s×d):
- `stable=false` (default): column-pivoted QR of C followed by a standard eigenproblem
  on the projected operator `C \\ D`.
- `stable=true`: thin SVD of C for basis whitening, truncating ill-conditioned
  directions (threshold `κ_max`), then a generalized eigenproblem.

Returns Ritz values `λs`, Ritz vectors `X`, the sketched residual
`‖D y₁ − λ₁ C y₁‖`, and a sampled estimate of the true residual
`‖H x₁ − λ₁ x₁‖ / ‖x₁‖`.
"""
function sketched_rr(H::TToperator{T,N}, B::Vector{TTvector{T,N}}, C::Matrix{T}, D::Matrix{T},
                     rmax::Int; orthogonal::Bool=true, block_rks=8, seed::Int=1234,
                     κ_max::Real=1/sqrt(eps(T)), stable=false) where {T,N}
    d = length(B)

    if stable
        F  = svd(C)
        σ  = F.S
        println("Sketch condition number κ(C) ≈ $(round(σ[1]/σ[end], sigdigits=4))")
        k  = something(findlast(σ ./ σ[1] .> 1 / κ_max), 1)
        k < d && @warn "Basis truncated from d=$d to k=$k (ill-conditioned directions removed)"

        U  = F.U[:, 1:k]
        Vk = F.V[:, 1:k]
        σk = σ[1:k]
        M  = U' * D * Vk
        λs, Z = eigen(M, diagm(σk))
        Y = real.(Vk * Z)
        for j = 1:k
            Y[:,j] = Y[:,j] / norm(C*Y[:,j])
        end
    else
        k = d
        F = qr(C, ColumnNorm())
        M = F \ D
        λs, Y = eigen(M)
        Y = real.(Y)
    end
    λs = real.(λs)
    res_sketch = norm(D*Y[:,1] - λs[1]*C*Y[:,1])

    # Assemble Ritz vector as TT linear combinations
    y = Y[:, 1] / norm(Y[:,1])
    X = ttrand_rounding(y, collect(B), 4rmax; orthogonal=orthogonal, seed=seed, block_rks=block_rks)
    X = tt_rounding(X; tol=res_sketch^2/2)
    X = X / norm(X)

    # Sampled estimate of the true residual ‖H x₁ − λ₁ x₁‖ / ‖x₁‖
    rks_samp  = ones(Int, N+1); rks_samp[1:N] .= N
    seed_samp = rand(Int)
    S1, = tt_sketch(T, H, X, 2N; orthogonal=orthogonal, reverse=true, seed=seed_samp, block_rks=block_rks)
    S2, = tt_sketch(T,    X, 2N; orthogonal=orthogonal, reverse=true, seed=seed_samp, block_rks=block_rks)
    res_sample = norm(S1 - λs[1]*S2) / norm(S2)

    return λs, X, res_sketch, res_sample
end

"""
    sketched_rayleigh_ritz(H, b0, d, rmax; ...) -> λ, ψ_rr, history

Build a `d`-dimensional sketch-orthogonal Krylov basis for `H` starting from `b0`,
interleaving sketched Rayleigh-Ritz (sRR) at each step to track convergence.

At each Arnoldi expansion step `j = 2, …, d`, the sketch `Ω H b_{j-1}` becomes
available and sRR is applied to the basis `B[1:j-1]`. A final sRR with all `d`
vectors is performed after the basis is complete.

# Arguments
- `H`: TT operator (Hamiltonian)
- `b0`: initial TT vector (sketch-normalized internally)
- `d`: Krylov dimension
- `rmax`: max TT rank for sketch ranks and compressed Ritz vectors
- `orthogonal`: use orthogonal sketch matrices (default: true)
- `block_rks`: sketch block size (default: 8)
- `seed`: random seed (default: 1234)
- `k_trunc`: sliding window size for sketch orthogonalization (default: `d-1`)
- `e_nuc`: nuclear repulsion energy added to all stored/returned eigenvalues (default: 0)

# Returns
- `λ`: lowest sketched Ritz value + `e_nuc`
- `ψ_rr`: ground-state Ritz vector as a TT
- `history`: `Vector` of `NamedTuple`, one entry per sRR solve, with fields:
  - `iter`: Krylov dimension `k` used in that sRR call (1 ≤ k ≤ d)
  - `e_sketch`: sketched Ritz estimate `λ₁ + e_nuc`
  - `e_true`: true Rayleigh quotient `⟨ψ|H|ψ⟩ + e_nuc`
  - `res_sketch`: sketched residual `‖D y₁ − λ₁ C y₁‖`
  - `res_sample`: sampled estimate of `‖H ψ − λ ψ‖ / ‖ψ‖`
"""
function sketched_rayleigh_ritz(H::TToperator{T,N}, b0::TTvector{T,N}, d::Int, rmax::Int;
                           orthogonal::Bool=true, block_rks=8, seed::Int=1234,
                           sketch_size = 4d, k_trunc::Int=d-1, e_nuc::Real=0.0) where {T,N}
    B         = Vector{TTvector{T,N}}(undef, d)
    sketch_b  = Vector{Vector{T}}(undef, d)
    sketch_Hb = Vector{Vector{T}}(undef, d)

    # ── b₁: sketch-normalize b0 ───────────────────────────────────────────────
    sketch_b[1], W_B_1, _ = tt_combined_sketch(T, b0, 2rmax, sketch_size; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)
    s = length(sketch_b[1])
    β = norm(sketch_b[1])
    sketch_b[1] = sketch_b[1] ./ β
    for i = 2:findfirst(isequal(0), b0.ttv_ot)
        W_B_1[i] ./= β
    end
    B[1] = b0 / β

    B_window        = TTvector{T,N}[B[1]]
    W_B_window      = [W_B_1]
    B_sketch_window = [sketch_b[1]]

    history = NamedTuple[]

    # ── Arnoldi iterations with interleaved sRR ───────────────────────────────
    for j = 2:d
        println("─── k = $(j-1) ────────────────────────────────────────────")
        b_new, sketch_b_j, sketch_Hb_prev =
            expand_basis!(H, B_window, W_B_window, B_sketch_window,
                          rmax, s; block_rks=block_rks, seed=seed, k_trunc=k_trunc)
        B[j]           = b_new
        sketch_b[j]    = sketch_b_j
        sketch_Hb[j-1] = sketch_Hb_prev

        # sRR with B[1:j-1] (sketch of H*b_{j-1} just became available)
        C = reduce(hcat, sketch_b[1:j-1])
        D = reduce(hcat, sketch_Hb[1:j-1])
        λs, ψ_rr, res_sketch, res_sample = sketched_rr(H, B[1:j-1], C, D, rmax;
                                                     block_rks=block_rks, seed=seed)
        e_sketch = λs[1] + e_nuc
        e_true   = dot_operator(ψ_rr, H, ψ_rr) + e_nuc

        push!(history, (iter=j-1, e_sketch=e_sketch, e_true=e_true,
                        res_sketch=res_sketch, res_sample=res_sample))

        println("  Sketched Ritz value:    ", e_sketch)
        println("  True Ritz value:        ", e_true)
        println("  Sketch residual:  ", res_sketch)
        println("  Sampled residual: ", res_sample)
    end

    # ── Final sRR with all d basis vectors ────────────────────────────────────

    println("─── k = $d (final) ─────────────────────────────────────────────")
    sketch_Hb[d], _ = tt_combined_sketch(T, H, B[d], 2rmax, s; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)
    C = reduce(hcat, sketch_b)
    D = reduce(hcat, sketch_Hb)
    λs, ψ_rr, res_sketch, res_sample = sketched_rr(H, B, C, D, rmax;
                                                 block_rks=block_rks, seed=seed)

    e_sketch = λs[1] + e_nuc
    e_true   = dot_operator(ψ_rr, H, ψ_rr) + e_nuc

    push!(history, (iter=d, e_sketch=e_sketch, e_true=e_true,
                    res_sketch=res_sketch, res_sample=res_sample))

    println("  Sketched Ritz:    ", e_sketch)
    println("  True Ritz:        ", e_true)
    println("  Sketch residual:  ", res_sketch)
    println("  Sampled residual: ", res_sample)

    return e_sketch, ψ_rr, history
end

"""
    sketched_rayleigh_ritz(H, b0, schedule; ...) -> λ, ψ_rr, stages

Multi-stage sketched Rayleigh-Ritz driven by a rank/dimension schedule.

`schedule` is a `Vector` of `NamedTuple`. Each entry specifies one stage.
Required fields per entry:
- `d::Int`: Krylov dimension
- `rmax::Int`: max TT rank for sketches and Ritz vectors

Optional per-stage fields (fall back to global kwargs when absent):
- `seed::Int`: random seed (default: global `seed` + stage index − 1)
- `sketch_size::Int`: sketch dimension (default: `4d`)
- `k_trunc::Int`: sliding window size (default: `d−1`)
- `label::String`: legend label (default: `"d=\$d, rmax=\$rmax"`)

At each stage transition the current best Ritz vector is truncated to the new
`rmax` and used as the starting vector; the Krylov basis is rebuilt from scratch.
When `validate_seed=true` the seed for each stage is retried (with fresh random
draws) until the sketch of the initial vector has norm within 10% of 1.

# Returns
- `λ`: lowest Ritz value + `e_nuc` from the final stage
- `ψ_rr`: corresponding Ritz vector
- `stages`: `Vector{Tuple{String,Vector{NamedTuple}}}` — one `(label, history)` pair
  per stage, directly passable to `plot_sRR_convergence`
- `seeds_used`: `Vector{Int}` — actual seed used for each stage (may differ from
  the schedule when `validate_seed=true`)
"""
function sketched_rayleigh_ritz(H::TToperator{T,N}, b0::TTvector{T,N},
                                  schedule::Vector{<:NamedTuple};
                                  orthogonal::Bool=true, block_rks::Int=8,
                                  seed::Int=1234, e_nuc::Real=0.0,
                                  validate_seed::Bool=false) where {T,N}
    stages     = Tuple{String, Vector{NamedTuple}}[]
    seeds_used = Vector{Int}(undef, length(schedule))
    ψ_rr       = b0
    λ          = 0.0

    for (i, stage) in enumerate(schedule)
        d    = stage.d
        rmax = stage.rmax
        s    = get(stage, :seed,        seed + i - 1)
        sk   = get(stage, :sketch_size, 4 * d)
        kt   = get(stage, :k_trunc,     d - 1)
        lbl  = get(stage, :label,       "d=$d, rmax=$rmax")

        b_init = i == 1 ? ψ_rr : tt_rounding(ψ_rr; rmax=rmax)

        if validate_seed
            while abs(norm(tt_combined_sketch(T, b_init, 2rmax, sk;
                           orthogonal=orthogonal, reverse=true,
                           seed=s, block_rks=block_rks)[1])^2 - 1) > 0.1
                s = rand(Int)
            end
            println("Stage $i: using seed $s")
        end

        seeds_used[i] = s
        λ, ψ_rr, hist = sketched_rayleigh_ritz(H, b_init, d, rmax;
                                                orthogonal=orthogonal,
                                                block_rks=block_rks,
                                                seed=s, k_trunc=kt,
                                                sketch_size=sk,
                                                e_nuc=e_nuc)
        push!(stages, (lbl, hist))
    end

    return λ, ψ_rr, stages, seeds_used
end

# ── Operator interface for `sketched_gmres` ─────────────────────────────────────
# `op` is either a `TToperator` (generic; product sketched via the operator-vector
# recursive sketch) or a function `x ↦ [𝒦_k·x]` returning the Kronecker/MPO summands of
# `A·x` (sum-of-Kronecker form). The summand path never forms a dense operator core and
# reuses the plain linear-combination adaptive overload — by linearity `S·(A·x)=∑ₖ S·𝒦ₖx`.
_sg_embed(op::TToperator{T,N}, pv, s, orthogonal, seed, block_rks; timer::TimerOutput=TimerOutput()) where {T,N} =
    Base.vec(tt_combined_sketch(T, op, pv, s, s; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)[1])
function _sg_embed(op, pv, s, orthogonal, seed, block_rks; timer::TimerOutput=TimerOutput())
    summ = op(pv); TT = eltype(summ[1])
    S = Base.vec(tt_combined_sketch(TT, summ[1], s, s; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)[1])
    for k = 2:length(summ)
        S = S .+ Base.vec(tt_combined_sketch(TT, summ[k], s, s; orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)[1])
    end
    return S
end
# win_caches: per-window-vector CachedSketch (aligned with `window`), reused/extended across
# iterations so a window vector is sketched once. Summands change every step (pv changes) → nothing
# caches. Only the summand path supports it (the TToperator overload ignores win_caches).
_sg_round(op::TToperator{T,N}, pv, h, window, ε, orthogonal, block_rks, seed; win_caches=nothing, timer::TimerOutput=TimerOutput()) where {T,N} =
    ttrand_rounding_adaptive(vcat(one(T), -h), op, vcat([pv], window), ε; orthogonal=orthogonal, block_rks=block_rks, seed=seed, timer=timer)
function _sg_round(op, pv, h, window, ε, orthogonal, block_rks, seed; win_caches=nothing, timer::TimerOutput=TimerOutput())
    summ = op(pv); TT = eltype(summ[1])
    caches = win_caches === nothing ? nothing : vcat(fill(nothing, length(summ)), win_caches)
    ttrand_rounding_adaptive(vcat(ones(TT, length(summ)), -h), vcat(summ, window), ε; orthogonal=orthogonal, block_rks=block_rks, seed=seed, caches=caches, timer=timer)
end
_sg_apply(op::TToperator, x) = op * x
_sg_apply(op, x) = reduce(+, op(x))
_sg_resid(op, x, b) = norm(tt_rounding(_sg_apply(op, x) - b; tol=1e-10)) / norm(b)

"""
    sketched_gmres(op, b, x0; prec=identity, m=50, tol=1e-8, rmax=256,
                   max_iters=m, sketch_size=2max_iters, k_trunc=2, ε_cap=0.1,
                   relax_factor=1/max_iters, final_trim=true, orthogonal=true,
                   block_rks=8, seed=1234, sample_res=true, verbose=true) -> (x, history)

Sketched GMRES (Nakatsukasa & Tropp, arXiv:2111.00113) for the TT linear system `A x = b`,
with optional right preconditioner `prec` (a function `v ↦ M⁻¹ v`, default identity). `op` is
either a `TToperator` `A`, or — for a sum-of-Kronecker / sum-of-MPO operator — a function
`x ↦ [𝒦_k·x]` returning the operator summands (cheap, rank-preserving; no dense core).

A fixed common embedding `S` (same `seed` throughout, dimension `sketch_size = 2·max_iters` —
an `O(m)` embedding sized from the expected iteration count) drives the cheap truncated-window
sketched Gram–Schmidt and the small least-squares. Each new Krylov vector `H·M⁻¹·v_j − ∑ hᵢ vᵢ`
is formed by the mixed **adaptive** rounding
`ttrand_rounding_adaptive([1; −h], H, [M⁻¹v_j; window], ε_basis,j)` — the operator product is
sketched implicitly and `∑ hᵢ vᵢ` is never formed.

The per-vector tolerance **relaxes like an inexact Krylov solver**:
`ε_basis,j = clamp(relax_factor · tol / relres_{j−1}, tol, ε_cap)` — tight while the residual is
large, loose as it shrinks — so the basis stays at its bounded natural rank instead of inflating
by chasing components the solver does not need. `relax_factor=1` is the (optimistic)
Bouras–Frayssé heuristic; the default `relax_factor=1/max_iters` is the van-den-Eshof/Simoncini
safety-factor tightening, which makes the residual actually reach `tol` rather than stagnating
just above it. `final_trim` adds a deterministic trim. The LS column is `D[j] = S·(A·M⁻¹·v_j)`;
the solution is `x = x₀ + M⁻¹(∑ yᵢ vᵢ)`.

`history` is a `Vector{NamedTuple}` with `(iter, res_sketch)` per step; the final entry also
carries `res_sample`, a sampled estimate of `‖b − H x‖/‖b‖`.
"""
function sketched_gmres(op, b::TTvector{T,N}, x0::TTvector{T,N};
                        prec=identity, m::Int=50, tol::Real=1e-8, rmax::Int=256,
                        max_iters::Int=m, sketch_size::Int=2max_iters, k_trunc::Int=2,
                        ε_cap::Real=0.1, relax_factor::Real=1/max_iters, final_trim::Bool=true, orthogonal::Bool=true,
                        block_rks=8, seed::Int=1234, sample_res::Bool=true,
                        track_cond::Bool=false, track_exact::Bool=false, verbose::Bool=true,
                        reuse_sketches::Bool=true,
                        timer::TimerOutput=TimerOutput(), show_timer::Bool=false) where {T,N}
    s = sketch_size                         # common embedding dimension (= 2·max_iters)
    # Per-window-vector reusable sketch cache (Stage-5): each window vector is sketched once and
    # reused/extended while it sits in the truncation window. init_rank must match the sum overload's
    # ℓ_min+n_samples defaults (ℓ_min=4, n_samples=max(N÷2,4)) so the initial prefix lines up.
    _cache_init_rank = 4 + max(N ÷ 2, 4)
    _mk_cache(vec) = cached_sketch(T, vec, block_rks, block_rks, _cache_init_rank; seed=seed, orthogonal=orthogonal)
    # Initial residual r0 = b − A x0 (x0 typically the zero TT).
    x0_zero = all(c -> all(iszero, c), x0.ttv_vec)
    r0 = x0_zero ? tt_rounding(b; tol=tol, rmax=rmax) :
                   tt_rounding(b - _sg_apply(op, x0); tol=tol, rmax=rmax)

    # ── Sketch-normalize r0 → v1 (base sketch boundary = s) ────────────────────
    sketch_r0, = tt_combined_sketch(T, r0, s, s;
                                    orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks)
    rhs = copy(sketch_r0)                   # S r0  — least-squares right-hand side
    β   = norm(sketch_r0)
    v1  = r0 / β

    B_window        = TTvector{T,N}[v1]     # truncated GS window
    B_sketch_window = [sketch_r0 ./ β]      # S·v_i for window vectors
    cache_window    = reuse_sketches ? Any[_mk_cache(v1)] : nothing   # parallel reusable sketch caches
    V = TTvector{T,N}[v1]                   # full basis (kept for the assembly)
    D = Vector{Vector{T}}()                 # full sketched images S·(H M⁻¹ v_j)
    history = NamedTuple[]
    local y
    k = 0
    relres_prev = one(T)                    # relative residual of the previous step

    # Basis-conditioning diagnostics: κ(C) of the sketched basis [S·v_i] (cheap) and, if
    # track_cond, the exact κ(V)=√κ(Gram) via TT inner products (the "actual" basis condition).
    Vsk  = Vector{Vector{T}}([sketch_r0 ./ β])
    Gram = zeros(T, m + 1, m + 1)
    track_cond && (Gram[1, 1] = dot(v1, v1))

    for j = 1:m
        # Inexact-Krylov relaxation: tighter early, looser as the residual shrinks.
        # relax_factor=1 is Bouras–Frayssé; relax_factor=1/max_iters is the safe
        # van-den-Eshof/Simoncini-style tightening (a-priori, no tuning).
        ε_basis = clamp(relax_factor * tol / relres_prev, tol, ε_cap)

        pv = prec(B_window[end])            # M⁻¹ v_j
        # Embedding of A·M⁻¹·v_j (drives the sketched GS and is the LS column D[j]).
        S_Hv = @timeit timer "embed" _sg_embed(op, pv, s, orthogonal, seed, block_rks; timer=timer)
        h = reduce(hcat, B_sketch_window) \ S_Hv      # sketched GS against the window
        push!(D, S_Hv)

        # New Krylov vector = A·M⁻¹·v_j − ∑ hᵢ vᵢ, rounded adaptively (∑ never formed).
        w = @timeit timer "round" _sg_round(op, pv, h, B_window, ε_basis, orthogonal, block_rks, seed; win_caches=cache_window, timer=timer)
        final_trim && (w = @timeit timer "final_trim" tt_rounding(w; tol=ε_basis))

        S_w, = @timeit timer "sketch_w" tt_combined_sketch(T, w, s, s;
                                  orthogonal=orthogonal, reverse=true, seed=seed, block_rks=block_rks, timer=timer)
        βw = norm(S_w)
        w = w / βw
        push!(V, w)
        push!(B_window, w);              length(B_window) > k_trunc && popfirst!(B_window)
        push!(B_sketch_window, S_w ./ βw); length(B_sketch_window) > k_trunc && popfirst!(B_sketch_window)
        if reuse_sketches
            push!(cache_window, _mk_cache(w)); length(cache_window) > k_trunc && popfirst!(cache_window)
        end

        # ── Basis conditioning ────────────────────────────────────────────────
        push!(Vsk, S_w ./ βw)
        κ_V = T(NaN)
        if track_cond
            jj = length(V)                  # = j+1; w is V[jj]
            for i = 1:jj
                Gram[i, jj] = dot(V[i], w);  Gram[jj, i] = Gram[i, jj]
            end
            κ_V = sqrt(cond(Symmetric(Gram[1:jj, 1:jj])))
        end

        κ_C = @timeit timer "cond" cond(reduce(hcat, Vsk))  # sketched basis condition (moved here for timing)
        Dmat = reduce(hcat, D)              # s × j
        y    = @timeit timer "lsq" qr(Dmat, ColumnNorm()) \ rhs
        res_sketch = norm(rhs - Dmat * y) / β

        # Exact residual of the current iterate xⱼ = M⁻¹(∑_{i≤j} yᵢ vᵢ): ‖H xⱼ − b‖/‖b‖.
        res_exact = T(NaN)
        if track_exact
            uj = y[1] * V[1]
            for i = 2:j
                uj = uj + y[i] * V[i]
            end
            uj = tt_rounding(uj; tol=tol, rmax=rmax)
            xj = x0_zero ? prec(uj) : x0 + prec(uj)
            res_exact = _sg_resid(op, xj, b)
        end

        push!(history, (iter=j, res_sketch=res_sketch, res_exact=res_exact, ε_basis=ε_basis, κ_sketch=κ_C, κ_basis=κ_V))
        verbose && (println("  sGMRES it $(lpad(j,2))  sketched = $(round(res_sketch, sigdigits=4))  EXACT = $(round(res_exact, sigdigits=4))  ε_basis = $(round(ε_basis, sigdigits=2))  rank = $(maximum(w.ttv_rks))  κ(V) = $(round(κ_V, sigdigits=3))"); flush(stdout))
        relres_prev = max(res_sketch, tol)
        k = j
        res_sketch < tol && break
    end

    # ── Assemble x = x0 + M⁻¹ (∑_i y_i v_i) ───────────────────────────────────
    # Round the weighted sum with the adaptive randomized linear-combination overload: a single
    # sketched sweep that never forms ∑ᵢ yᵢ·Vᵢ. Forming it explicitly inflated the running TT to
    # ∑ᵢ rank(Vᵢ) (~thousands of ranks, 15.6 GiB) and dominated the solve (~40% of wall). This is
    # the same machinery the basis vectors are built with; accuracy is preserved (the achieved
    # residual ~1e-6 sits well above tol). u is the *preconditioned* variable, so its rounding
    # perturbs the residual by ≈‖H·M⁻¹‖·tol≈tol — harmless (see the no-round note below).
    @timeit timer "assemble" begin
    u = k == 1 ? y[1] * V[1] :
        ttrand_rounding_adaptive(y[1:k], V[1:k], tol; orthogonal=orthogonal, block_rks=block_rks, seed=seed)
    # Cheap deterministic trim: the randomized sweep leaves u at a modest rank (already ~rmax,
    # not the ∑ᵢ rank(Vᵢ) of the raw sum), so this final SVD pass is inexpensive and recovers the
    # minimal rank for the same tol (randomized rounding tends to retain a little extra rank).
    k > 1 && (u = tt_rounding(u; tol=tol, rmax=rmax))
    x = x0_zero ? prec(u) : x0 + prec(u)
    end
    # NOTE: do NOT round x here. Rounding the un-preconditioned solution at tolerance δ
    # perturbs the residual by ‖H·δ‖ ≈ ‖H‖·δ; for the stiff cookie operator (‖H‖~1e3–1e4)
    # a tol=1e-8 round inflates ‖Hx−b‖ to ~1e-5. The preconditioned variable u is already
    # rounded above (harmless, since ‖H·M⁻¹‖≈1). If a final trim is needed, use tol/‖H‖.

    if sample_res
        res_final = @timeit timer "sample_res" _sg_resid(op, x, b)   # exact ‖A x − b‖/‖b‖
        history[end] = merge(history[end], (res_sample=res_final,))
        verbose && println("  sGMRES true relres = $(round(res_final, sigdigits=4))")
    end
    show_timer && (show(timer); println())

    return x, history
end
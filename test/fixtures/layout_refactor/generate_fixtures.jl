#=
Baseline fixture generator for the (I,L,R) → (L,I,R) core layout refactor.

Run on the pre-refactor branch to capture canonical outputs of
layout-sensitive operations. Stored fixtures are layout-invariant
(scalars and dense tensors), so the post-refactor `test_layout_refactor.jl`
can rebuild the same TT inputs and verify numerics match within tolerance.

Run with:
    julia --project=. test/fixtures/layout_refactor/generate_fixtures.jl
=#

using TensorTrains
using LinearAlgebra
using Random
using Serialization

const FIXTURE_DIR = @__DIR__
const D = 6
const N_PHYS = 2
const DIMS = ntuple(_ -> N_PHYS, D)

function save_jls(name::String, data)
    path = joinpath(FIXTURE_DIR, name * ".jls")
    open(path, "w") do io
        serialize(io, data)
    end
    println("  wrote ", name * ".jls")
end

function safely(name, f)
    try
        return f()
    catch e
        println("    [WARN] $name failed: ", sprint(showerror, e))
        return missing
    end
end

# -----------------------------------------------------------------
# 1. Canonical dense inputs
# -----------------------------------------------------------------
println("Generating inputs...")
Random.seed!(42)
x_dense = randn(DIMS...)
Random.seed!(43)
y_dense = randn(DIMS...)
save_jls("inputs_dense", Dict(:x_dense => x_dense, :y_dense => y_dense, :dims => DIMS))

x_tt = ttv_decomp(x_dense)
y_tt = ttv_decomp(y_dense)

results = Dict{Symbol,Any}()

# -----------------------------------------------------------------
# 2. Invariants and reconstruction
# -----------------------------------------------------------------
println("Computing scalar invariants...")
results[:norm_x] = norm(x_tt)
results[:norm_y] = norm(y_tt)
results[:dot_xy] = dot(x_tt, y_tt)
results[:dot_xx] = dot(x_tt, x_tt)
results[:dense_dot_xy_ref] = vec(x_dense) ⋅ vec(y_dense)
results[:tensor_x_reconstructed] = ttv_to_tensor(x_tt)
results[:tensor_y_reconstructed] = ttv_to_tensor(y_tt)

# -----------------------------------------------------------------
# 3. tt_rounding
# -----------------------------------------------------------------
println("Running tt_rounding...")
x_round_l = tt_rounding(x_tt; tol=1e-6, direction=:left)
x_round_r = tt_rounding(x_tt; tol=1e-6, direction=:right)
results[:tensor_x_round_left]  = ttv_to_tensor(x_round_l)
results[:tensor_x_round_right] = ttv_to_tensor(x_round_r)
results[:norm_x_round_left]    = norm(x_round_l)
results[:norm_x_round_right]   = norm(x_round_r)

# orthogonalize at different roots
results[:tensor_x_orth_1] = ttv_to_tensor(orthogonalize(x_tt; i=1))
results[:tensor_x_orth_3] = ttv_to_tensor(orthogonalize(x_tt; i=3))
results[:tensor_x_orth_d] = ttv_to_tensor(orthogonalize(x_tt; i=D))

# -----------------------------------------------------------------
# 4. Arithmetic
# -----------------------------------------------------------------
println("Running arithmetic...")
sum_xy = x_tt + y_tt
results[:tensor_sum_xy] = ttv_to_tensor(sum_xy)
results[:norm_sum_xy] = norm(sum_xy)

scaled_x = 2.5 * x_tt
results[:tensor_scaled_x] = ttv_to_tensor(scaled_x)

# -----------------------------------------------------------------
# 5. Operator construction and A*v
# -----------------------------------------------------------------
println("Constructing operator and A*v...")
Random.seed!(44)
H = rand_tto(DIMS, 3)
results[:tensor_H_dense] = tto_to_tensor(H)

Av = H * x_tt
results[:tensor_Av]  = ttv_to_tensor(Av)
results[:norm_Av]    = norm(Av)
results[:dot_x_Av]   = dot(x_tt, Av)

# -----------------------------------------------------------------
# 6. outer_product
# -----------------------------------------------------------------
println("Running outer_product...")
xy_op = outer_product(x_tt, y_tt)
results[:tensor_xy_op] = tto_to_tensor(xy_op)

# -----------------------------------------------------------------
# 7. Randomized rounding (fixed seed)
# -----------------------------------------------------------------
println("Running ttrand_rounding...")
big_sum = x_tt + y_tt + (1.5 * x_tt) + (0.5 * y_tt)
target_rks = [1, 2, 3, 3, 3, 2, 1]

Random.seed!(101)
big_round_rand = ttrand_rounding(big_sum, target_rks)
results[:tensor_big_round_rand] = ttv_to_tensor(big_round_rand)
results[:norm_big_round_rand]   = norm(big_round_rand)

# Deterministic reference at high precision
big_round_det = tt_rounding(big_sum; tol=1e-12)
results[:tensor_big_round_det] = ttv_to_tensor(big_round_det)
results[:norm_big_sum_dense]   = norm(big_round_det)

# -----------------------------------------------------------------
# 8. STTA
# -----------------------------------------------------------------
println("Running stta...")
Random.seed!(102)
stta_out = safely("stta", () -> stta(big_sum, target_rks))
if !ismissing(stta_out)
    results[:tensor_stta] = ttv_to_tensor(stta_out)
    results[:norm_stta] = norm(stta_out)
end

# -----------------------------------------------------------------
# 9. Eigensolvers — build Hermitian H
# -----------------------------------------------------------------
println("Constructing Hermitian H_herm...")
H_dense_full = tto_to_tensor(H)
# permute the i and j multi-indices to take transpose, then symmetrize
perm_t = (collect(D+1:2D)..., collect(1:D)...)
H_dense_herm = 0.5 .* (H_dense_full .+ permutedims(H_dense_full, perm_t))
H_herm_tt = tto_decomp(H_dense_herm)

H_herm_mat = reshape(H_dense_herm, prod(DIMS), prod(DIMS))
F = eigen(Hermitian(H_herm_mat))
results[:H_herm_eigenvalues] = F.values
results[:H_herm_min_eigval]  = F.values[1]
results[:tensor_H_herm_tt] = tto_to_tensor(H_herm_tt)

Random.seed!(200)
x0 = rand_tt(DIMS, [1, 2, 4, 4, 4, 2, 1])

println("  als_eigsolv...")
als_out = safely("als_eigsolv", () ->
    als_eigsolv(H_herm_tt, x0; sweep_schedule=[3], rmax_schedule=[4]))
if !ismissing(als_out)
    E_als, x_als = als_out
    results[:als_eigvals_history] = E_als
    results[:als_eigval]          = E_als[end]
    results[:tensor_als_eigvec]   = ttv_to_tensor(x_als)
    results[:norm_als_eigvec]     = norm(x_als)
end

println("  mals_eigsolv...")
Random.seed!(201)
x0_mals = rand_tt(DIMS, [1, 2, 4, 4, 4, 2, 1])
mals_out = safely("mals_eigsolv", () ->
    mals_eigsolv(H_herm_tt, x0_mals; sweep_schedule=[2], rmax_schedule=[4]))
if !ismissing(mals_out)
    E_mals, x_mals = mals_out
    results[:mals_eigvals_history] = E_mals
    results[:mals_eigval]          = E_mals[end]
    results[:tensor_mals_eigvec]   = ttv_to_tensor(x_mals)
    results[:norm_mals_eigvec]     = norm(x_mals)
end

println("  dmrg_eigsolv...")
Random.seed!(202)
x0_dmrg = rand_tt(DIMS, [1, 2, 4, 4, 4, 2, 1])
dmrg_out = safely("dmrg_eigsolv", () -> begin
    sched = dmrg_schedule_default(; rmax=4, nsweeps=2)
    dmrg_eigsolv(H_herm_tt, x0_dmrg; schedule=sched, verbose=false)
end)
if !ismissing(dmrg_out)
    if length(dmrg_out) >= 2
        E_dmrg = dmrg_out[1]
        x_dmrg = dmrg_out[2]
        results[:dmrg_eigvals_history] = E_dmrg
        if length(E_dmrg) >= 1
            results[:dmrg_eigval] = E_dmrg[end]
        end
        results[:tensor_dmrg_eigvec] = ttv_to_tensor(x_dmrg)
        results[:norm_dmrg_eigvec]   = norm(x_dmrg)
    end
end

# -----------------------------------------------------------------
# Save
# -----------------------------------------------------------------
save_jls("results", results)
println("\nDone. Keys captured:")
for k in sort!(collect(keys(results)); by=string)
    println("  ", k)
end

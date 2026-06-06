# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

TensorTrains is a Julia package implementing tensor train (TT) decomposition methods for high-dimensional tensors. The package focuses on quantum chemistry and condensed matter physics applications, particularly solving eigenvalue problems and linear systems for Hamiltonians in tensor train format.

## Development Commands

### Running Tests
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

To run specific test files:
```bash
julia --project=. test/test_tt_tools.jl
julia --project=. test/test_als.jl
julia --project=. test/test_mals.jl
```

### Activating the Project
```bash
julia --project=.
```

Then in the Julia REPL:
```julia
using TensorTrains
```

### Running Examples
Examples are in the `examples/` directory and can be run directly:
```bash
julia --project=. examples/hubbard_1d.jl
julia --project=. examples/laplacian.jl
```

## Architecture

### Core Data Structures

- **TTvector**: Represents tensors in tensor train format as `C[μ₁,...,μ_d] = A₁[μ₁] * ... * A_d[μ_d]`
  - `ttv_vec`: Array of 3-order tensor cores `A_k[α_{k-1}, μ_k, α_k]` — axes ordered `(left rank, physical index, right rank)`
  - `ttv_dims`: Dimensions along each mode
  - `ttv_rks`: TT ranks `(r₀,...,r_d)` where `r₀=r_d=1`
  - `ttv_ot`: Orthogonality information (-1: right-orth, 0: unknown, 1: left-orth)

- **TToperator**: Represents matrices in TT format `M[i₁,..,i_d; j₁,..,j_d]`
  - `tto_vec`: Array of 4-order tensor cores `A_k[α_{k-1}, i_k, j_k, α_k]` — axes ordered `(left rank, row physical, column physical, right rank)`
  - Similar rank and orthogonality structure as TTvector

- **TT_vidal**: Vidal/MPS representation with explicit singular values
  - `core`: Orthogonal tensor cores (same `(left, physical, right)` axis order as TTvector)
  - `Σ`: Singular value arrays

The `(L, I, R)` layout matches Julia's column-major memory order: the canonical
left-unfolding `reshape(core, L*I, R)` and right-unfolding `reshape(core, L, I*R)`
are pure reshapes with no `permutedims`, which makes SVD, QR, and gemm-based
operations cache-contiguous. Previously the codebase used `(I, L, R)`; the
refactor under `refactor-core-layout` switched conventions globally.

### Module Organization

The codebase is organized into focused modules in `src/`:

- **tt_tools.jl**: Core TT data structures, constructors, conversions (TTvector, TToperator, zeros_tt, rand_tt, ones_tt, json I/O)
- **tt_operations.jl**: Arithmetic operations (+, -, *, /, dot, outer_product)
- **tt_rounding.jl**: Compression and orthogonalization (tt_rounding, orthogonalize, tt_svdvals, norm)
- **tt_randtools.jl**: Randomized algorithms (ttrand_rounding, stta, tt_hmt for sketching)
- **tt_adaptive_rounding.jl**: Adaptive randomized TT rounding to a target Frobenius tolerance (`ttrand_rounding_adaptive`); recursive-sketch helpers live in `src/sketches/`
- **als.jl**: Alternating Linear Scheme (one-site DMRG) for linear systems and eigenvalue problems
- **mals.jl**: Modified ALS (two-site DMRG)
- **dmrg.jl**: DMRG with scheduling (DMRGScheduler, sweep schedules, adaptive rank control)
- **tt_solvers.jl**: Iterative solvers (tt_cg, tt_gmres, gradient_fixed_step, eig_arnoldi, davidson)
- **models.jl**: Physics Hamiltonians (hubbard_1D, hubbard_2D, hV_to_mpo for second quantization)
- **ordering_schemes.jl**: Orbital ordering optimization (fiedler_order, bwpo_order, entropy calculations, reduced density matrices)
- **qtt.jl**: Quantized TT (QTT) for function approximation
- **FCIDUMP.jl**: Read quantum chemistry integral files

### Key Algorithms

**DMRG/ALS Solvers**: The package implements both one-site (ALS) and two-site (MALS/DMRG) algorithms for:
- Eigenvalue problems: `als_eigsolv`, `mals_eigsolv`, `dmrg_eigsolv`
- Linear systems: `als_linsolv`, `mals_linsolv`, `dmrg_linsolv`

DMRG uses a scheduler system (`DMRGScheduler`) with:
- `sweep_schedule`: Number of sweeps per bond dimension
- `rmax_schedule`: Maximum rank schedule
- Adaptive truncation based on SVD with tolerance parameter
- Switch between direct and iterative solvers based on subproblem size

**Orthogonalization**: TT cores can be left- or right-orthogonalized. The `ttv_ot` array tracks orthogonality:
- Left sweep: cores become right-orthogonal
- Right sweep: cores become left-orthogonal
- Critical for numerical stability in DMRG

**Randomized Methods**: `tt_randtools.jl` implements sketching-based compression:
- `ttrand_rounding`: Randomized TT rounding (fixed target ranks)
- `stta`: Sketched TT approximation
- `tt_hmt`: TT-HMT for randomized compression

`tt_adaptive_rounding.jl` implements `ttrand_rounding_adaptive` (Al Daas et al.,
arXiv:2511.03598): a single left-to-right sweep that grows each bond's basis from a
recursive (TTStack) sketch until a sketch-based residual estimate falls below the
per-bond budget `τ = ε·‖y‖_F/√(N-1)`. Overloads cover a single TT, a linear
combination `Σ αⱼ yⱼ`, an operator residual `A·y − b`, and a Hadamard product
`y₁ ⊙ … ⊙ y_M` — each sketching the target implicitly (never forming it). Key knobs:
- `init_f`: per-bond *initial* basis width as a fraction of the bond's rank cap
  (localizes the `ℓ_min` floor).
- `ℓ_inc`: small absolute floor on the per-iteration increment; the increment is
  otherwise local (`0.2·current_cols`), so it scales with the bond rank, **not**
  `ℓ_max`. Do not tie `ℓ_inc` to `ℓ_max` — on Kronecker/Hadamard products `ℓ_max`
  (the Kronecker rank) far exceeds typical bond ranks and over-inflates mid-bond
  ranks and runtime.
- `block_rks` / `block_rks_inc`: sketch block ranks for the initial / extension
  sketches (TTStack vs pure-KRP behaviour).

**Oblivious-embedding facts (do not re-derive these the hard way).** TTStack is an
*oblivious* (data-independent) subspace embedding: its distortion depends on the
embedding dimension and the dimension of the subspace being embedded, **not on the
internal TT rank of the vectors**. A high-rank and a low-rank TT are sketched with the
same fidelity at a given embedding size — so the recursive/embedding rank never needs to
exceed (or even track) the vectors' TT ranks. Conversely, the pure Khatri–Rao product
sketch (`block_rks=1`, `orthogonal=false`) is **worst for rank-1 TTs**: its variance is
highest there and improves as the input rank grows, which is the opposite of the
intuition that low-rank inputs are "easier" to sketch.

The basis-expansion helper `expand_basis!` uses two QR sweeps (orthonormalize,
then project ⊥ Q and re-orthonormalize) and a rank-revealing fallback; the residual
slice fed to it is sized to `max_basis - current_cols`, so it never over-extracts.

### Physics Models

Second quantization Hamiltonians of the form:
```
H = Σ_{i,j} h_{ij} (a_i† a_j + c.c.) + Σ_{i,j,k,l} V_{ijkl} (a_i† a_j† a_l a_k + c.c.)
```

- `hV_to_mpo`: Converts one-body (h) and two-body (V) integrals to TT operator
- `hubbard_1D`, `hubbard_2D`: Construct Hubbard model Hamiltonians
- `site_switch`: Change orbital orderings in TT operators
- Orbital ordering significantly affects TT rank; use `ordering_schemes.jl` to optimize

### Testing

Tests are organized by module functionality:
- `test_tt_tools.jl`: Basic TT operations and conversions
- `test_als.jl`, `test_mals.jl`, `test_dmrg.jl`: Solver tests
- `test_models.jl`: Physics Hamiltonian construction
- `test_ordering_schemes.jl`: Orbital ordering methods

## Important Conventions

- **Indexing**: Julia uses 1-based indexing
- **Ranks**: TT ranks are stored as `(r₀, r₁, ..., r_d)` with `r₀ = r_d = 1`
- **Tensor contractions**: Use `@tensor` and `@tensoropt` macros from TensorOperations.jl
- **Orthogonality tracking**: Always update `ttv_ot` or `tto_ot` when modifying cores
- **Type parameters**: TTvector and TToperator are parameterized by element type `T` and number of dimensions `M`
- **Rounding a solution vs the residual it produces**: rounding a TT solution `x` at tolerance
  `δ` perturbs the residual of a linear system by `‖A·δx‖ ≈ ‖A‖·δ·‖x‖`. For a **stiff**
  operator (`‖A‖ ≫ 1`, e.g. an unpreconditioned FEM stiffness operator with `‖A‖~1e3–1e4`),
  a seemingly tight `tol=1e-8` round inflates `‖Ax−b‖` to `~1e-5`. **Do not round the
  un-preconditioned solution at a loose tolerance**: round the *preconditioned* variable `u`
  (where `x = M⁻¹u` and `‖A·M⁻¹‖≈1`, so rounding is harmless), or round `x` at `tol/‖A‖`.
  This was the (config-invariant, hence very confusing) residual floor in the cookie
  sketched-GMRES experiments — the iterate's true residual was already `~5e-8`, but a final
  `tt_rounding(x; tol=1e-8)` blew the *reported* residual up to `~3e-5`.

## BLAS/LAPACK backend (Apple Silicon)

On Apple Silicon, prefer the default **OpenBLAS** for any correctness-sensitive
work. Apple's **Accelerate** *LAPACK* (the ILP64 path Julia uses via
`AppleAccelerate`) has an intermittent, silent correctness bug on the large
chained SVD/QR factorizations in `tt_rounding`/`orthogonalize` (observed on
macOS 26 / M-series): the same deterministic rounding returns a wrong result on
~10–85% of repeated calls (dominant singular component dropped, ranks inflated),
while OpenBLAS is always correct. Accelerate's *BLAS* (gemm) is fine — only its
LAPACK is affected (this is why MATLAB, which pairs Accelerate BLAS with NAG
LAPACK, is unaffected). If you need Accelerate's matrix-engine speed, a hybrid
(Accelerate BLAS + OpenBLAS LAPACK via `lbt_set_forward`) is correct; otherwise
run benchmarks under plain OpenBLAS so adaptive and deterministic timings share
one backend.

## Git Conventions

- **Commits**: Do not add `Co-Authored-By` trailers to commit messages in this repository.

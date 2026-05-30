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
- `ttrand_rounding`: Randomized TT rounding
- `stta`: Sketched TT approximation
- `tt_hmt`: TT-HMT for randomized compression

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

## Git Conventions

- **Commits**: Do not add `Co-Authored-By` trailers to commit messages in this repository.

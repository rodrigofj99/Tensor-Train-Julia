#=
JSON migration for the TT core layout refactor.

Before the refactor:
  - TTvector cores were stored row-major-flat as (I, L, R) — physical, left rank, right rank.
  - TToperator cores were (i_out, i_in, L, R).

After the refactor (this branch):
  - TTvector cores are (L, I, R).
  - TToperator cores are (L, i_out, i_in, R).

`json_to_mps` and `json_to_mpo` reshape the on-disk flat array directly into
the new layout, so existing JSON files (~800 MB total in `examples/`) become
incorrect under the new readers. This script reads each JSON with the OLD
reshape semantics, permutes the core arrays into the NEW layout, then writes
back to the same path with `_v2` suffix.

Usage:
    julia --project=. tools/migrate_tt_json.jl <file_or_directory> [<file_or_directory> ...]

By default the script writes to `<original_basename>_v2.json` next to the
input. Pass `--inplace` as the first argument to overwrite the originals
(making backup copies as `<original>.bak`).

Files this script targets (verified):
  - examples/Be.ezfio.FCIDUMP_mpo_tol1.0em10.json (TToperator)
  - examples/lih_mpo_ncas*_tol*.json              (TToperator)
  - lih_sRR_ncas19.json                           (TTvector)
=#

using JSON3
using Serialization

function load_legacy_mps_core(flat::AbstractVector, dims, rks, k)
    # OLD layout (I, L, R): reshape with I fastest, then L, then R.
    return reshape(collect(flat), dims[k], rks[k], rks[k+1])
end

function load_legacy_mpo_core(flat::AbstractVector, dims, rks, k)
    # OLD layout (i, j, L, R).
    return reshape(collect(flat), dims[k], dims[k], rks[k], rks[k+1])
end

# Permute OLD (I, L, R) → NEW (L, I, R) then flatten in column-major.
function permute_mps_core_to_new(core_old)
    return vec(permutedims(core_old, (2, 1, 3)))
end

# Permute OLD (i, j, L, R) → NEW (L, i, j, R) then flatten.
function permute_mpo_core_to_new(core_old)
    return vec(permutedims(core_old, (3, 1, 2, 4)))
end

function infer_kind(obj)
    if haskey(obj, :ttv_vec) && haskey(obj, :ttv_dims)
        return :mps
    elseif haskey(obj, :tto_vec) && haskey(obj, :tto_dims)
        return :mpo
    else
        error("Unrecognised JSON: expected ttv_vec/ttv_dims (TTvector) or tto_vec/tto_dims (TToperator)")
    end
end

function migrate_one(path::AbstractString; inplace::Bool=false)
    print("migrating ", path, " … ")
    obj = JSON3.read(read(path, String))
    kind = infer_kind(obj)

    if kind === :mps
        dims = Tuple(convert(Vector{Int64}, obj[:ttv_dims]))
        rks  = convert(Vector{Int64}, obj[:ttv_rks])
        N    = obj[:N]
        new_flats = Vector{Any}(undef, N)
        T = eltype(obj[:ttv_vec][1])
        for i in 1:N
            core_old = load_legacy_mps_core(convert(Vector{T}, obj[:ttv_vec][i]), dims, rks, i)
            new_flats[i] = permute_mps_core_to_new(core_old)
        end
        # Build the new JSON-compatible object preserving the original keys/order.
        new_obj = Dict(string(k) => obj[k] for k in keys(obj))
        new_obj["ttv_vec"] = new_flats
    elseif kind === :mpo
        dims = Tuple(convert(Vector{Int64}, obj[:tto_dims]))
        rks  = convert(Vector{Int64}, obj[:tto_rks])
        N    = obj[:N]
        new_flats = Vector{Any}(undef, N)
        T = eltype(obj[:tto_vec][1])
        for i in 1:N
            core_old = load_legacy_mpo_core(convert(Vector{T}, obj[:tto_vec][i]), dims, rks, i)
            new_flats[i] = permute_mpo_core_to_new(core_old)
        end
        new_obj = Dict(string(k) => obj[k] for k in keys(obj))
        new_obj["tto_vec"] = new_flats
    end

    if inplace
        bak = path * ".bak"
        isfile(bak) || cp(path, bak)
        write(path, JSON3.write(new_obj))
        println("done (overwrote; original at ", bak, ")")
    else
        out = replace(path, r"\.json$" => "_v2.json")
        write(out, JSON3.write(new_obj))
        println("done → ", out)
    end
    return nothing
end

function expand_targets(args)
    targets = String[]
    for arg in args
        if isdir(arg)
            for entry in readdir(arg; join=true)
                if endswith(entry, ".json") && !endswith(entry, "_v2.json")
                    push!(targets, entry)
                end
            end
        elseif isfile(arg)
            push!(targets, arg)
        else
            @warn "skipping non-existent path: $arg"
        end
    end
    return targets
end

function main(args)
    if isempty(args)
        println("Usage: julia --project=. tools/migrate_tt_json.jl [--inplace] <file_or_directory> [...]")
        return
    end
    inplace = false
    if first(args) == "--inplace"
        inplace = true
        args = args[2:end]
    end
    for path in expand_targets(args)
        try
            migrate_one(path; inplace=inplace)
        catch e
            @error "failed to migrate $path" exception=(e, catch_backtrace())
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end

# Low-level buffer-reusing kernels for sketch contractions.
# `permutedims!!` and `mul!!` accept an optional preallocated `buffer`;
# when `buffer === nothing` they fall back to allocating versions.

function permutedims!!(A::AbstractArray{T, N}, perm; buffer=nothing) where {T,N}
  if buffer === nothing
    A_permuted = permutedims(A, perm)
  else
    @assert length(buffer) >= length(A) "Buffer too small: need $(length(A)), got $(length(buffer))"
    permuted_size = ntuple(i -> size(A, perm[i]), Val(N))
    A_permuted = unsafe_wrap(Array, pointer(buffer), permuted_size)
    permutedims!(A_permuted, A, perm)
  end
  return A_permuted
end

function mul!!(A::AbstractMatrix{T}, B::AbstractMatrix{T}; buffer=nothing) where {T}
  if buffer === nothing
    C = A*B
  else
    m = size(A,1)
    n = size(B,2)
    @assert length(buffer) >= m*n "Buffer too small: need $(m*n)=($m)⨯($n), got $(length(buffer))"
    C = unsafe_wrap(Array, pointer(buffer), (m,n))
    mul!(C, A,B)
  end
  return C
end


function mul!!(A::AbstractArray{T,3}, B::AbstractArray{T,3}; buffer=nothing) where {T}
  if buffer === nothing
    m = size(A,1)
    n = size(B,2)
    K = size(A,3)
    @assert size(B,3) == K "Incompatible sizes: expected size(A,3)==$(K)==size(B,3)==$(size(B,3))"
    C = Array{T,3}(undef, m,n,K)
    for k=1:K
      @views mul!(C[:,:,k], A[:,:,k], B[:,:,k])
    end
  else
    m = size(A,1)
    n = size(B,2)
    K = size(A,3)
    @assert length(buffer) >= m*n*K "Buffer too small: need $(m*n*k)=($m)⨯($n)⨯($K), got $(length(buffer))"
    C = unsafe_wrap(Array, pointer(buffer), (m,n,K))
    for k=1:K
      @views mul!(C[:,:,k], A[:,:,k], B[:,:,k])
    end  end
  return C
end

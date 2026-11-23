# ==========================================================================================
# Thus is the only unit in this folder that needs to be included.
# It includes all other units.
# ==========================================================================================

# How custom rrule works:

# ChainRulesCore.rrule is an interception point. 
# When an AD system like Zygote, encounters a forward call while differentiating, for example `geodesicr_t(P, Q, t)`,
# it does the following dispatch check before running anything:

# - Is there a method ChainRulesCore.rrule(::typeof(geodesicr_t), ::AbstractMatrix, ::AbstractMatrix, ::Real) defined?
#   If yes, Zygote calls that rrule directly. It uses the Y it returns as the primal output (i.e., the forward output), 
#       and registers the returned pullback for later use in the backward pass. 
#       The actual body of geodesicr_t is never executed at all.
#   If no, Zygote falls through and traces into the function body, differentiating each operation inside it individually.

# In details for the example of `geodesicr-t`:
#   → finds rrule(::typeof(geodesicr_t), ...)
#   → executes that rrule's body directly
#        → calls gevdr(P,Q,deepcopy(P))   ①  (ONE GEVD, period)
#        → builds Y, K, fλ
#        → returns Y, pullback
#   → later, backward pass calls pullback(Ȳ)
#        → reuses λ, A, B, K, fλ from closure — NO second gevdr call

# Writing custom rrules is convenient for stability reasons, but also for efficiency:
# Without caching, a differentiated call would need the decomposition once for the primal and, 
# in the absence of a fused rule, AD would either recompute it or need to retain a much larger 
# intermediate tape (every elementary op inside eigen's internals, if it were differentiable at all) 
# to reconstruct the gradient. The custom GEVD rules computes (𝜆,𝐴,𝐵) once, discards everything else, 
# and the backward pass is only base don matrix matrix products — no second decomposition, 
# and no tape of LAPACK internals to store.

# Throughout the code, Ȳ is the cotangent, the gradient returned by the AD system

#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================

##################
# helper functions
##################


SymOrHerm = Union{Hermitian, Symmetric}

"""
    sym(X)

Symmetric part ``(X+X^T)/2`` of a square matrix `X`.
"""
sym(X) = (X + X') * (one(eltype(X))/2) # (X + X') / 2


"""
    skew(X)

Skew-symmetric part ``(X-X^T)/2`` of a square matrix `X`.
"""
skew(X) = (X - X') * (one(eltype(X))/2) # (X - X') / 2

# Adjust the pullbacks to allow input matrix X (first argument) to be of any type. 
# With these adjustments the ChainRulesTestUtils.test_rrule passes
# regardless whether X is a Matrix, Symmetric or Hermitian julia type. 
# Thus, we avoid errors due to possible type instabilities or errors in coding.
# Two methods are given, arranging the different types of the X.
function desym!(::AbstractMatrix, ΔX::Matrix{T}) where {T<:AbstractFloat}
    n = size(ΔX,1)
    z = zero(T)
    t = T(2)

    @inbounds for j in 1:n, i in 1:j-1
        ΔX[i,j] *= t
        ΔX[j,i] = z
    end
    return ΔX
end

function desym!(::Union{LinearAlgebra.Symmetric, LinearAlgebra.Hermitian}, ΔX::Matrix{T}) where {T<:AbstractFloat}
    n = size(ΔX,1)
    t = T(2)

    @inbounds for j in 1:n, i in 1:j-1
        ΔX[i,j] *= t
        ΔX[j,i] *= t
    end
    return ΔX
end


@inline _drop_last_tangent(pb) = Ȳ -> Base.front(pb(Ȳ))




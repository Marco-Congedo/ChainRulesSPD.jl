# ========================================================================================================
#   Custom rules to be applied in back-propagation for the batched GEVD functions defined in
#   `fw_GEVD_batched.jl`.
#
#   The scalar (matrix) rules are defined in `bp_rules_GEVD.jl`.
#   This file provides the batched rules.
#
#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================


using LinearAlgebra


# ==========================================================================================
# Löwner matrices, one per slice
# ==========================================================================================

#=
Löwner matrix of f for every slice, computed when the rrule is called (forward pass) and
cached for the pullback. K_b depends on λ_b, so it is NOT shared across slices: one K per slice.
=#
function _loewner_batch(
        Λs::AbstractVector{Vector{T}},
        f::Function;

        f′::Union{Nothing,Function} = nothing
    ) where {T<:AbstractFloat}

    Ks = Vector{Matrix{T}}(undef, length(Λs))

    @inbounds for b in eachindex(Λs) # it can be multi-threaded
        Ks[b] = loewner(Λs[b], f; f′)
    end

    return Ks
end


# ==========================================================================================
# Generic batched pullback for phi
# ==========================================================================================

#=
Pullback of one slice of

    ϕ(f,P,Q) = A'f(Λ)A

with G = sym(A Ȳ A'):

    ΔQ = B (K ⊙ G) B'
    ΔP = B (Z ⊙ G) B',     Z_ij = ½ [ (f_i + f_j) - (λ_i + λ_j) K_ij ]

Z⊙G is symmetric, so no inner symmetrization is needed (the outer `sym` only removes rounding asymmetry).
Returns ΔP, ΔQ and G (G is used by the pullback of geodesicr_t for Δt).
=#
@inline function _phir_slice(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        λ::AbstractVector{T},
        fλ::AbstractVector{T},
        K::AbstractMatrix{T},
        Yb::AbstractMatrix{T}
    ) where {T<:AbstractFloat}

    G = sym(A * Yb * A')
    Z = @. T(0.5) * ((fλ + fλ') - (λ + λ') * K)

    ΔQb = sym(B * (K .* G) * B')
    ΔPb = sym(B * (Z .* G) * B')

    return ΔPb, ΔQb, G
end

#=
Batched pullback for

    ϕ(f,P,Q) = A'f(Λ)A

Returns a pullback computing ΔP and ΔQ.
It receives the quantities cached by the forward pass: (Λs, As, Bs, fλs) from `_phirGEVD_batch`
and the Löwner matrices Ks.
The pullback returns (NoTangent(), ΔP, ΔQ, NoTangent()) (the last slot is for `f`, or for `t` in geodesicr).
=#
function _phir_pullback_batch(
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        Λs,
        As,
        Bs,
        fλs,
        Ks
    ) where {T<:AbstractFloat}

    function pullback(Ȳ)

        Ȳ = ChainRulesCore.unthunk(Ȳ)

        ΔP = similar(P)
        ΔQ = similar(Q)

        @inbounds for b in axes(P,3) # it can be multi-threaded

            ΔPb, ΔQb, _ = _phir_slice(As[b], Bs[b], Λs[b], fλs[b], Ks[b], @view(Ȳ[:,:,b]))

            @views ΔP[:,:,b] .= ΔPb
            @views ΔQ[:,:,b] .= ΔQb
        end

        return (NoTangent(), ΔP, ΔQ, NoTangent())
    end

    return pullback
end


# ==========================================================================================
# Generic batched ϕ function
# ==========================================================================================

function ChainRulesCore.rrule(::typeof(phir),
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        f::Function;

        f′::Union{Nothing,Function} = nothing,
        precond::Bool = false
    ) where {T<:AbstractFloat}

    Λs, As, Bs, fλs, Y = _phirGEVD_batch(P, Q, f; precond)

    Ks = _loewner_batch(Λs, f; f′) # this passes f′ to loewner

    return Y, _phir_pullback_batch(P, Q, Λs, As, Bs, fλs, Ks)
end



#=
Forward pass + pullback for any batched ϕ wrapper: returns (Y, pullback), where
`pullback(Ȳ)` = (NoTangent(), ΔP, ΔQ, NoTangent()).
=#
function _phir_rrule_batch(
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        f::Function,
        f′::Function;

        precond::Bool = false
    ) where {T<:AbstractFloat}

    Λs, As, Bs, fλs, Y = _phirGEVD_batch(P, Q, f; precond)

    Ks = _loewner_batch(Λs, f; f′)

    return Y, _phir_pullback_batch(P, Q, Λs, As, Bs, fλs, Ks)
end



# ==========================================================================================
# Geodesic without learnable arc-length parameter
# ==========================================================================================


function ChainRulesCore.rrule(::typeof(geodesicr),
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        t::T;

        precond::Bool = false
    ) where {T<:AbstractFloat}

    # pullback = (NoTangent(), ΔP, ΔQ, NoTangent()), the last one being for t
    return _phir_rrule_batch(P, Q, Power(t), Power′(t); precond)
end

# ==========================================================================================
# Geodesic with learnable arc-length parameter
# ==========================================================================================


function ChainRulesCore.rrule(::typeof(geodesicr_t),
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        t::T;
        precond::Bool = false
    ) where {T<:AbstractFloat}

    f  = Power(t)
    f′ = Power′(t) # for loewner function

    Λs, As, Bs, fλs, Y = _phirGEVD_batch(P, Q, f; precond)

    Ks = _loewner_batch(Λs, f; f′)

    function pullback(Ȳ)

        Ȳ = ChainRulesCore.unthunk(Ȳ)

        ΔP = similar(P)
        ΔQ = similar(Q)

        Δt = zero(T)

        @inbounds for b in axes(P,3) # it can be multi-threaded

            λ  = Λs[b]
            fλ = fλs[b]

            ΔPb, ΔQb, G = _phir_slice(As[b], Bs[b], λ, fλ, Ks[b], @view(Ȳ[:,:,b]))

            # d/dt A' Λ^t A = A' Λ^t log(Λ) A  =>  Δt = Σ_i G_ii λ_i^t log(λ_i)
            Δt += dot(diag(G), fλ .* log.(λ))

            @views ΔP[:,:,b] .= ΔPb
            @views ΔQ[:,:,b] .= ΔQb
        end

        return (NoTangent(), ΔP, ΔQ, Δt)
    end

    return Y, pullback
end

# ==========================================================================================
# Geometric mean
# ==========================================================================================

function ChainRulesCore.rrule(::typeof(geomeanr),
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    Y, pb = _phir_rrule_batch(P, Q, Sqrt(), Sqrt′(); precond)

    return Y, _drop_last_tangent(pb)
end

# ==========================================================================================
# Logarithmic Map
# ==========================================================================================


function ChainRulesCore.rrule(::typeof(logmapr),
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    Y, pb = _phir_rrule_batch(P, Q, Log(), Log′(); precond)

    return Y, _drop_last_tangent(pb)
end

# ==========================================================================================
# Exponential Map
# ==========================================================================================


function ChainRulesCore.rrule(::typeof(expmapr),
        P::AbstractArray{T,3},
        Ξ::AbstractArray{T,3};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    Y, pb = _phir_rrule_batch(P, Ξ, Exp(), Exp′(); precond)

    return Y, _drop_last_tangent(pb)
end

############################################################################################

# ==========================================================================================
# Generic batched pullback for psi
# ==========================================================================================

#=
Pullback of one slice of

    ψ(f,P,Q) = B f(Λ) A

with J = A Ē' B  (Ē = upstream gradient ∂ℓ/∂ψ), Js = sym(J), H = K ⊙ Js:

    ΔQ = B H B'
    ΔP = B sym( skew(J) f(Λ) - H Λ ) B'

Since skew(J) is antisymmetric and H is symmetric, the symmetric part is obtained
entrywise, without any explicit sym or skew:

    sym( skew(J) f(Λ) - H Λ )_ij = ¼ (J_ij - J_ji) (f_j - f_i) - ½ H_ij (λ_i + λ_j)

Returns ΔP, ΔQ.
=#
@inline function _psir_slice(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        λ::AbstractVector{T},
        fλ::AbstractVector{T},
        K::AbstractMatrix{T},
        Yb::AbstractMatrix{T}
    ) where {T<:AbstractFloat}

    # Since skew(J) is antisymmetric and H is symmetric: 
    # sym(skew(J) f(Λ) − HΛ)_ij = ¼ (J_ij − J_ji)(f_j − f_i) − ½ H_ij (λ_i + λ_j)    
    J = A * Yb' * B
    H = K .* sym(J)
    W = @. T(0.25) * (J - J') * (fλ' - fλ) - T(0.5) * H * (λ + λ')

    ΔQb = sym(B * H * B')
    ΔPb = sym(B * W * B')

    return ΔPb, ΔQb
end

#=
    _psir_pullback_batch(P, Q, Λs, As, Bs, fλs, Ks)

Batched pullback for

    ψ(f,P,Q) = B f(Λ) A

It receives the quantities cached by the forward pass: (Λs, As, Bs, fλs) from `_psirGEVD_batch`
and the Löwner matrices Ks (one per slice).
The pullback returns (NoTangent(), ΔP, ΔQ, NoTangent()).
=#
function _psir_pullback_batch(
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        Λs,
        As,
        Bs,
        fλs,
        Ks
    ) where {T<:AbstractFloat}

    function pullback(Ȳ)

        Ȳ = ChainRulesCore.unthunk(Ȳ)

        ΔP = similar(P)
        ΔQ = similar(Q)

        @inbounds for b in axes(P,3) # it can be multi-threaded

            ΔPb, ΔQb = _psir_slice(As[b], Bs[b], Λs[b], fλs[b], Ks[b], @view(Ȳ[:,:,b]))

            @views ΔP[:,:,b] .= ΔPb
            @views ΔQ[:,:,b] .= ΔQb
        end

        return (NoTangent(), ΔP, ΔQ, NoTangent())
    end

    return pullback
end

# ==========================================================================================
# Generic batched ψ function
# ==========================================================================================

function ChainRulesCore.rrule(::typeof(psir),
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        f::Function;

        f′::Union{Nothing,Function} = nothing,
        precond::Bool = false
    ) where {T<:AbstractFloat}

    Λs, As, Bs, fλs, Y = _psirGEVD_batch(P, Q, f; precond)

    Ks = _loewner_batch(Λs, f; f′) # this passes f′ to loewner

    return Y, _psir_pullback_batch(P, Q, Λs, As, Bs, fλs, Ks)
end



# ==========================================================================================
# Parallel transport
# ==========================================================================================


function ChainRulesCore.rrule(::typeof(parallel_transportr),
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        S::AbstractArray{T,3};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    f  = Sqrt()
    f′ = Sqrt′()

    # Single GEVD-based computation: keep both E and (Λs, As, Bs, fλs) from the same call
    Λs, As, Bs, fλs, E = _psirGEVD_batch(P, Q, f; precond)

    Y = similar(S)

    @inbounds for b in axes(S,3) # it can be multi-threaded
        Eb = @view E[:,:,b]
        @views Y[:,:,b] .= sym(Eb' * S[:,:,b] * Eb)
    end

    # forward pass: Löwner matrices, one per slice
    Ks = _loewner_batch(Λs, f; f′)

    pb = _psir_pullback_batch(P, Q, Λs, As, Bs, fλs, Ks)

    function pullback(Ȳ)

        Ȳ = ChainRulesCore.unthunk(Ȳ)
        ΔS = similar(S)
        Ē  = similar(S)

        @inbounds for b in axes(S,3) # it can be multi-threaded

            Sb  = @view S[:,:,b]
            Eb  = @view E[:,:,b]
            Yb  = @view Ȳ[:,:,b]

            @views ΔS[:,:,b] .= sym(Eb * Yb * Eb')
            @views Ē[:,:,b]  .= Sb * Eb * (Yb + Yb')   # ∂ℓ/∂E, S symmetric
        end

        _, ΔP, ΔQ, _ = pb(Ē)

        return (NoTangent(), ΔP, ΔQ, ΔS)
    end

    return Y, pullback
end

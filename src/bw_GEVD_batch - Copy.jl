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
# Generic batched pullback for phi
# ==========================================================================================

#=
Batched pullback for

    ϕ(f,P,Q) =A'f(Λ)A
 
Batched pullback for
 
    ϕ(f,P,Q) = A'f(Λ)A
 
Returns a pullback computing ΔP and ΔQ.

f(λ_b) and the Löwner matrix K_b are computed here, i.e. when the rrule is
called (forward pass), and cached per slice, as the single-matrix rule does.
K_b depends on λ_b, so it is NOT shared across slices: one K per slice.
The pullback returns (NoTangent(), ΔP, ΔQ, NoTangent()) (the last slot is for `f`, or for `t` in geodesicr).
=#
function _phir_pullback_batch(
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        Λs,
        As,
        Bs,
        f::Function;
 
        f′::Union{Nothing,Function} = nothing
    ) where {T<:AbstractFloat}
 
    # cached in the forward pass, one per slice
    fλs = [f.(λ) for λ in Λs]
    Ks  = [loewner(λ, f; f′) for λ in Λs]
 
    function pullback(Ȳ)
 
        Ȳ = ChainRulesCore.unthunk(Ȳ)
 
        ΔP = similar(P)
        ΔQ = similar(Q)
 
        @inbounds for b in axes(P,3)
 
            Pb = @view P[:,:,b]
            Qb = @view Q[:,:,b]
 
            Yb = @view Ȳ[:,:,b]
 
            λ  = Λs[b]
            A  = As[b]
            B  = Bs[b]
            fλ = fλs[b]
            K  = Ks[b]
 
            G = sym(A * Yb * A')
            H = K .* G
 
            ΔQb = sym(B * H * B')
            ΔPb = sym(B * sym(G .* fλ' - H .* λ') * B')
 
            desym!(Pb, ΔPb)
            desym!(Qb, ΔQb)
 
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

    Λs, As, Bs, Y = _phirGEVD_batch(P, Q, f; precond)

    pullback = _phir_pullback_batch(P, Q, Λs, As, Bs, f; f′) # this passes f′ to loewner

    return Y, pullback
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
 
    Λs, As, Bs, Y = _phirGEVD_batch(P, Q, f; precond)
 
    return Y, _phir_pullback_batch(P, Q, Λs, As, Bs, f; f′)
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
 
    Λs, As, Bs, Y = _phirGEVD_batch(P, Q, f; precond)
 
    # cached in the forward pass, one per slice
    fλs = [λ .^ t for λ in Λs]
    Ks  = [loewner(λ, f; f′) for λ in Λs]
 
    function pullback(Ȳ)
 
        Ȳ = ChainRulesCore.unthunk(Ȳ)
 
        ΔP = similar(P)
        ΔQ = similar(Q)
 
        Δt = zero(T)
 
        @inbounds for b in axes(P,3)
 
            Pb = @view P[:,:,b]
            Qb = @view Q[:,:,b]
            Yb = @view Ȳ[:,:,b]
 
            λ  = Λs[b]
            A  = As[b]
            B  = Bs[b]
            fλ = fλs[b]
            K  = Ks[b]
 
            G = sym(A * Yb * A')
            H = K .* G
 
            ΔQb = sym(B * H * B')
            ΔPb = sym(B * sym(G .* fλ' - H .* λ') * B')
 
            Δt += dot(diag(G), fλ .* log.(λ))
 
            desym!(Pb, ΔPb)
            desym!(Qb, ΔQb)
 
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
    _psir_pullback_batch(P, Q, Λs, As, Bs, f; f′)
 
Batched pullback for
 
    ψ(f,P,Q) = B f(Λ) A
 
f(λ_b) and the Löwner matrix K_b are computed here, i.e. when the rrule is
called (forward pass), and cached per slice. K_b depends on λ_b: one K per slice.
The pullback returns (NoTangent(), ΔP, ΔQ, NoTangent()).
=#
function _psir_pullback_batch(
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        Λs,
        As,
        Bs,
        f::Function;
 
        f′::Union{Nothing,Function} = nothing
    ) where {T<:AbstractFloat}
 
    # cached in the forward pass, one per slice
    fλs = [f.(λ) for λ in Λs]
    Ks  = [loewner(λ, f; f′) for λ in Λs]
 
    function pullback(Ȳ)
 
        Ȳ = ChainRulesCore.unthunk(Ȳ)
 
        ΔP = similar(P)
        ΔQ = similar(Q)
 
        @inbounds for b in axes(P,3)
 
            Pb = @view P[:,:,b]
            Qb = @view Q[:,:,b]
 
            Yb = @view Ȳ[:,:,b]
 
            λ  = Λs[b]
            A  = As[b]
            B  = Bs[b]
            fλ = fλs[b]
            K  = Ks[b]
 
            G = A * Yb' * B
            H = K .* sym(G)
 
            ΔQb = sym(B * H * B')
 
            ΔPb = sym(B * sym(skew(G).*fλ' - H.*λ') * B')
 
            desym!(Pb, ΔPb)
            desym!(Qb, ΔQb)
 
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

    Λs, As, Bs, Y = _psirGEVD_batch(P, Q, f; f′, precond)

    pullback = _psir_pullback_batch(P, Q, Λs, As, Bs, f; f′) # this passes f′ to loewner 

    return Y, pullback
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
 
    # Single GEVD-based computation: keep both E and (Λs, As, Bs) from the same call
    Λs, As, Bs, E = _psirGEVD_batch(P, Q, f; f′, precond)
 
    Y = similar(S)
 
    @inbounds for b in axes(S,3)
        @views Y[:,:,b] .= sym(E[:,:,b]' * S[:,:,b] * E[:,:,b])
    end
 
    # forward pass: caches K_b and f(λ_b) for every slice
    pb = _psir_pullback_batch(P, Q, Λs, As, Bs, f; f′)
 
    function pullback(Ȳ)
 
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        ΔS = similar(S)
        Ē  = similar(S)
 
        @inbounds for b in axes(S,3)
 
            Sb  = @view S[:,:,b]
            Eb  = @view E[:,:,b]
            Yb  = @view Ȳ[:,:,b]
 
            @views ΔS[:,:,b] .= sym(Eb * Yb * Eb')
            @views Ē[:,:,b]  .= Sb * Eb * (Yb + Yb')
        end
 
        _, ΔP, ΔQ, _ = pb(Ē)
 
        return (NoTangent(), ΔP, ΔQ, ΔS)
    end
 
    return Y, pullback
end
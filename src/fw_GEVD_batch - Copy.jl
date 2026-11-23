# ========================================================================================================
#   GEVD-based SPD operations. Batched
#
#   Implements the generalized spectral function
#
#       ϕ(f,P,Q) = P f(P⁻¹Q)
#                = A' f(Λ) A
#
#   together with the auxiliary function
#
#       ψ(f,P,Q) = B f(Λ) A
#
#   where
#
#       P = A'A
#       Q = A'ΛA
#       B = A⁻¹.
# 
#   from paper: 
#   Congedo M, Bhatia R, Bouchard F (2026)
#   A new look at the Riemannian manifold of positive definite matrices
#
#   These routines are the forward operators corresponding to the custom
#   back-propagation rules implemented in bw_gevd_batch.jl.
#
#   They allow an efficient way to compute:
#   - The geodesic (including the geometric mean)
#   - The logarithmic map
#   - The exponential map
#   - The parallel transport
#
#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================

using LinearAlgebra

# ==========================================================================================
# Generic batched forward helper for auxiliary ϕ function
# ==========================================================================================

#=
    _phirGEVD_batch(P, Q, f; 
            precond::Bool = false)

Forward pass for batched generalized spectral function

    ϕ(f,P,Q) = A' f(Λ) A

For each batch element:

    λ, A, B = gevdr_phir(P[:,:,b], Q[:,:,b])

and stores

    Λs[b] = λ
    As[b] = A
    Bs[b] = B
    Y[:,:,b] = ϕ(f,P,Q)

Returns
-------
Λs : Vector of eigenvalue vectors
As : Vector of A matrices
Bs : Vector of B matrices
Y  : transformed batch
=#
function _phirGEVD_batch(
        P::AbstractArray{T,3}, 
        Q::AbstractArray{T,3},
        f::Function;

        precond::Bool = false 
    ) where {T<:AbstractFloat}

    Bsize = size(P,3)

    Y  = similar(P)

    Λs = Vector{Vector{T}}(undef, Bsize)
    As = Vector{Matrix{T}}(undef, Bsize)
    Bs = Vector{Matrix{T}}(undef, Bsize)

    @inbounds for b in axes(P,3)

        Pb = @view P[:,:,b]
        Qb = @view Q[:,:,b]

        λ, A, B = gevdr_phir(Pb, Qb; precond)

        Yb = sym(A' * Diagonal(f.(λ)) * A)

        Λs[b] = λ
        As[b] = A
        Bs[b] = B

        @views Y[:,:,b] .= Yb
    end

    return Λs, As, Bs, Y
end


# ==========================================================================================
# Generic batched ϕ function
# ==========================================================================================

#=
    phir(P, Q, f; f′, precond::Bool = false  )

Apply the generalized spectral function

    ϕ(f,P,Q)=A'f(Λ)A

independently to every matrix pair in the batch.
=#
function phir(
        P::AbstractArray{T,3}, 
        Q::AbstractArray{T,3}, 
        f::Function;

        f′::Union{Nothing,Function} = nothing,
        precond::Bool = false            
    ) where {T<:AbstractFloat}

    _, _, _, Y = _phirGEVD_batch(P, Q, f; precond)

    return Y
end



# ==========================================================================================
# Generic batched forward helper for auxiliary ψ function
# ==========================================================================================

#=
    _psirGEVD_batch(P, Q, f; f′, precond::Bool = false  )

Forward pass for

    ψ(f,P,Q)=B f(Λ) A

Stores all GEVD quantities required by the backward pass.
=#
function _psirGEVD_batch(
        P::AbstractArray{T,3}, 
        Q::AbstractArray{T,3}, 
        f::Function;

        f′::Union{Nothing,Function} = nothing,
        precond::Bool = false
    ) where {T<:AbstractFloat}

    Bsize = size(P,3)
    Y  = similar(P)

    Λs = Vector{Vector{T}}(undef, Bsize)
    As = Vector{Matrix{T}}(undef, Bsize)
    Bs = Vector{Matrix{T}}(undef, Bsize)

    @inbounds for b in axes(P,3)

        Pb = @view P[:,:,b]
        Qb = @view Q[:,:,b]

        λ, A, B = gevdr_psir(Pb, Qb; precond)

        Yb = B * Diagonal(f.(λ)) * A

        Λs[b] = λ
        As[b] = A
        Bs[b] = B

        @views Y[:,:,b] .= Yb
    end

    return Λs, As, Bs, Y
end


# ==========================================================================================
# Generic batched ψ function
# ==========================================================================================

#=
    psir(P, Q, f; f′, precond::Bool = false  )

Apply the generalized auxiliary spectral function

    ψ(f,P,Q)=Bf(Λ)A

independently over the batch.
=#
function psir(
        P::AbstractArray{T,3}, 
        Q::AbstractArray{T,3}, 
        f::Function;

        f′::Union{Nothing,Function} = nothing, # dummy, used to have same signature as rrule(::psir,...)
        precond::Bool = false            
    ) where {T<:AbstractFloat}

    _, _, _, Y = _psirGEVD_batch(P, Q, f; f′, precond)

    return Y
end


# ==========================================================================================
# Common batched GEVD wrappers
# Rrule are intercepted by phir
# ==========================================================================================

#=
Affine-invariant geodesic with fixed arc-length t.
=#
geodesicr(
        P::AbstractArray{T,3}, 
        Q::AbstractArray{T,3}, 
        t::T;

        precond::Bool = false            
    ) where {T<:AbstractFloat} = phir(P, Q, Power(t); f′ = Power′(t), precond)

#=
Affine-invariant geodesic with learnable arc-length t.

A specialised rrule provide Δt.
=#
geodesicr_t(
        P::AbstractArray{T,3}, 
        Q::AbstractArray{T,3}, 
        t::T;

        precond::Bool = false     
    ) where {T<:AbstractFloat} = phir(P, Q, Power(t); f′ = Power′(t), precond)


#=
Geometric mean.
=#
geomeanr(
        P::AbstractArray{T,3}, 
        Q::AbstractArray{T,3};
        
        precond::Bool = false                 
    ) where {T<:AbstractFloat} = phir(P, Q, Sqrt(); f′ = Sqrt′(), precond)


#=
Logarithmic map.
=#
logmapr(
        P::AbstractArray{T,3}, 
        Q::AbstractArray{T,3};
        precond::Bool = false                 
    ) where {T<:AbstractFloat} = phir(P, Q, Log(); f′ = Log′(), precond)


#=
Exponential map.
=#
expmapr(
        P::AbstractArray{T,3},
        Ξ::AbstractArray{T,3};
        precond::Bool = false                 
    ) where {T<:AbstractFloat} = phir(P, Ξ, Exp(); f′ = Exp′(), precond)


# ==========================================================================================
# Parallel transport
# ==========================================================================================


#=
Parallel transport of S from P to Q.

Computes

    E' S E

for every batch element.
=#
function parallel_transportr(
        P::AbstractArray{T,3},
        Q::AbstractArray{T,3},
        S::AbstractArray{T,3};

        precond::Bool = false                 
    ) where {T<:AbstractFloat}

    E = psir(P, Q, Sqrt(); f′=Sqrt′(), precond)
    Y = similar(S)
    @inbounds for b in axes(S,3)
        @views Y[:,:,b] .= sym(E[:,:,b]' * S[:,:,b] * E[:,:,b])
    end
    return Y
end



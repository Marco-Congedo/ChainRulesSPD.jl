# ========================================================================================================
#   SPD forward operations. The corresponding custom rules for back-propagation are defined in
#   `bw_EVD_batched.jl`.
#
#   All batched spectral functions are built upon the generic primitive `spfr`.
#
#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================


# ------------------------------------------------------------------------------------------
# Generic batched forward helper
# ------------------------------------------------------------------------------------------


#=
    _spfrEVD_batch(X, f)

Forward pass for batched EVD-based spectral functions.

For each SPD matrix `X[:,:,b]`, computes

    λ, V, Y = _spfrEVD(X[:,:,b], f)

and stores

- `Vs[b] = V`
- `Λs[b] = λ`
- `Y[:,:,b] = Y`

Arguments
---------
X : batch of SPD matrices of size (p,p,B)
f : scalar spectral function

Returns
-------
Vs : vector of eigenvector matrices
Λs : vector of eigenvalue vectors
Y  : transformed batch, same size as `X`
=#
function _spfrEVD_batch(X::AbstractArray{T,3}, f::Function) where {T<:AbstractFloat}

    B = size(X,3)

    Y  = similar(X)
    Vs = Vector{Matrix{T}}(undef, B)
    Λs = Vector{Vector{T}}(undef, B)

    @inbounds for b in axes(X,3)

        λ, V, Yb = _spfrEVD(@view(X[:,:,b]), f)

        Λs[b] = λ
        Vs[b] = V

        #copyto!(@view(Y[:,:,b]), Yb)
        @views Y[:,:,b] .= Yb
    end

    return Vs, Λs, Y
end


# ------------------------------------------------------------------------------------------
# Generic batched spectral primitive
# ------------------------------------------------------------------------------------------

#=
    spfr(X, f; f′)

Apply the scalar spectral function `f` independently to each SPD matrix
stored in the third dimension of `X`.
=#
function spfr(
    X::AbstractArray{T,3}, 
    f::Function;
    
    f′::Union{Nothing,Function} = nothing, # used by the rrule only
    ) where {T<:AbstractFloat}

    _, _, Y = _spfrEVD_batch(X, f)

    return Y
end


# ------------------------------------------------------------------------------------------
# One-line batched spectral functions
# ------------------------------------------------------------------------------------------

#=
Batched symmetric matrix ReLU: λ -> max(λ, ϵ)
=#
relur(X::AbstractArray{T,3}, ϵ::T) where {T<:AbstractFloat} =
    spfr(X, ReLU(ϵ); f′=ReLU′(ϵ))


#=
Batched symmetric matrix log
=#    
logr(X::AbstractArray{<:AbstractFloat,3}) = 
    spfr(X, Log(); f′=Log′())


#=
Batched symmetric matrix log with ReLU activation
=#    
logr_relu(X::AbstractArray{T,3}, ϵ::T) where {T<:AbstractFloat} =
    spfr(X, LogReLU(ϵ); f′=LogReLU′(ϵ))

#=
Batched symmetric matrix exp
=#    
expr(X::AbstractArray{<:AbstractFloat,3}) = 
    spfr(X, Exp(); f′=Exp′())

#=
Batched symmetric matrix sqrt
=#    
sqrtr(X::AbstractArray{<:AbstractFloat,3}) = 
    spfr(X, Sqrt(); f′=Sqrt′())

#=
Batched symmetric matrix Invsqrt
=#    
invsqrtr(X::AbstractArray{<:AbstractFloat,3}) = 
    spfr(X, Invsqrt(); f′=Invsqrt′())

#= 
Batched symmetric matrix power for fixed arc-length t
=#
powerr(X::AbstractArray{T,3}, t::T) where {T<:AbstractFloat} = 
    spfr(X, Power(t); f′=Power′(t))

#= 
Batched symmetric matrix power for learnable arc-length t
=#
powerr_t(X::AbstractArray{T,3}, t::T) where {T<:AbstractFloat} = 
    spfr(X, Power(t); f′=Power′(t))
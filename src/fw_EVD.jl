
# ========================================================================================================
#   SPD forward operations. The corresponding custom rule for back-prop are defined in `bp_EVD.jl`.
#   
#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================

# NB: All functions are cached. Only eigenr and spfr don't do any caching. 
# That's fine because every function actually exposed for differentiation (the wrappers named above) 
# either has its own rrule or delegates entirely to one that does.

#=
```julia
eigenr(X)
```
Return 2-tuple ``(λ, U)``, with ``λ`` the eigenvalues and ``U`` the eigenvectors of symmetric matrix `X`, i.e., its eigendecomposition (EVD).

The elemets of `X` must be floating-point numbers.

`Symmetric(X)` is passed to the standard julia `eigen` method, thus `X` can also be of type `Matrix`
(a plain `Matrix` is first symmetrized as `Hermitian((X+X')/2)`; `Symmetric` and `Hermitian` inputs are used unchanged).
=#
@inline function eigenr(X::AbstractMatrix{T}; ϵ::T=sqrt(eps(T))) where {T<:AbstractFloat}
    # Symmetric/Hermitian inputs are used as-is; a plain Matrix is symmetrized
    # so the result (and the gradient) depends on both triangles of X.
    Xs = X isa SymOrHerm ? X : Hermitian((X + X') / T(2))
    return eigen(Xs)
end


""" 
```julia
function spfr(P, f::Function; f′=nothing)
```
Generic method to compute **any function** ``f(P)`` based on the eigendecomposition (EVD). 

!!! tip "Use wrappers"
    This function is rarely needed; for usual choices of ``f`` use wrappers:
    
    [`relur`](@ref), [`logr`](@ref), [`logr_relu`](@ref), [`expr`](@ref), [`sqrtr`](@ref), [`invsqrtr`](@ref), [`powerr`](@ref), [`powerr_t`](@ref).

`P` can be either an ``n \\times n`` matrix ``\\in \\mathcal{S}_{++}`` or a ``n \\times n \\times m`` batch (tensor),
holding ``m`` of such matrices (slices). The elements of `P` must be floating-point numbers. 

``f`` can be any scalar function or one of these [functors](@ref "Parametric and non-parametric functors").

Keyword argument ``f'`` is used only if the `rrule` for `spfr` is needed (e.g., if a call to `spfr` is used in a network).
If you pass ``f'``, this must be a function or functor computing the derivative of ``f``.
If you pass it, it will be used, otherwise it will be derived automatically.

**Examples**
```julia
using ChainRulesSPD
using PosDefManifold, PosDefManifoldML # data generation

P = randP(3) # a random matrix of Julia's `Hermitian` type
t = 0.25

# The following calls are all equivalent
Pᵗ = spfr(P, x->x^t; f′=x->t*x^(t-1))
Pᵗ = spfr(P, Power(t), f′=Power′(t))
Pᵗ = spfr(Matrix(P), Power(t); f′=Power′(t))

# batched version (Example: Tensor 3x3x10)
P_batch = hermVec2tensor(randP(3, 10)) 

P_batchᵗ = spfr(P_batch, Power(t); f′=Power′(t))

# Examples with other functions
logP = spfr(P, Log(); f′=Log′())
myP = spfr(P, x->x^2+x; f′=x->2*x+1) 
```
"""
function spfr(
        X::AbstractMatrix{T}, 
        f::Function; 

        f′::Union{Nothing,Function} = nothing, # used by the rrule only
        ϵ::T= sqrt(eps(T))
    ) where {T<:AbstractFloat}
    
    λ, V = eigenr(X; ϵ)
    return sym(V * Diagonal(f.(λ)) * V') # return a generic matrix object
end


#= 
Compute ANY spectral functions and return also the EVD - epsilon is not used for the moment being
This is used internally in rrule functions
=#
function _spfrEVD(
        X::AbstractMatrix{T}, 
        f::Function; 
        
        ϵ::T=sqrt(eps(T))
    ) where {T<:AbstractFloat}

    λ, V = eigenr(X; ϵ)
    return λ, V, sym(V * Diagonal(f.(λ)) * V') # return a generic matrix object
end

# ==========================================================================================
# Wrappers for common EVD functions
# ==========================================================================================

""" 
```julia
relur(P, ϵ)
```
ReLU activation. Equivalent to calling [`spfr`](@ref) with the [ReLU](@ref "Parametric and non-parametric functors") function.

``ϵ`` is the activation threshold. It must be a floating-point number.

**Examples**
```julia
using ChainRulesSPD
using PosDefManifold, PosDefManifoldML # data generation

P = randP(3) # a random matrix of Julia's `Hermitian` type
ϵ = eps(eltype(P))

# Single-matrix
P_relu = relur(P, ϵ)

# batched version (Example: Tensor 3x3x10)
P_batch = hermVec2tensor(randP(3, 10)) 
P_relu_batch = relur(P_batch, ϵ)
```
"""
relur(X::AbstractMatrix{T}, ϵ::T) where {T<:AbstractFloat} = spfr(X, ReLU(ϵ); f′=ReLU′(ϵ))


""" 
```julia
logr(P)
```
Symmetric matrix logarithm. Equivalent to calling [`spfr`](@ref) with the [log](@ref "Parametric and non-parametric functors") function.

**Examples**
```julia
using ChainRulesSPD
using PosDefManifold, PosDefManifoldML # data generation

P = randP(3) # a random matrix of Julia's `Hermitian` type

# Single-matrix
P_log = logr(P)

# batched version (Example: Tensor 3x3x10)
P_batch = hermVec2tensor(randP(3, 10)) 
P_log_batch = logr(P_batch)
```
"""
logr(X::AbstractMatrix{<:AbstractFloat}) = spfr(X, Log(); f′=Log′()) 


""" 
```julia
logr_relu(P, ϵ)
```
Symmetric matrix logarithm with ReLU activation. Equivalent to calling [`spfr`](@ref) with the [LogReLU](@ref "Parametric and non-parametric functors") function.

``ϵ`` is the activation threshold. It must be a floating-point number.

**Examples** 

Use `logr_relu` instead of `relur` in the examples of [`relur`](@ref).
"""
logr_relu(X::AbstractMatrix{T}, ϵ::T) where {T<:AbstractFloat} = spfr(X, LogReLU(ϵ); f′=LogReLU′(ϵ))


""" 
```julia
expr(P)
```
Symmetric matrix exponential. Equivalent to calling [`spfr`](@ref) with the [exp](@ref "Parametric and non-parametric functors") function.

**Examples** 

Use `expr` instead of `logr` in the examples of [`logr`](@ref).
"""
expr(X::AbstractMatrix{<:AbstractFloat}) = spfr(X, Exp(); f′=Exp′())


""" 
```julia
sqrtr(P)
```
Symmetric matrix square root. Equivalent to calling [`spfr`](@ref) with the [sqrt](@ref "Parametric and non-parametric functors") function.

**Examples** 

Use `sqrtr` instead of `logr` in the examples of [`logr`](@ref).
"""
sqrtr(X::AbstractMatrix{<:AbstractFloat}) = spfr(X, Sqrt(); f′=Sqrt′())


""" 
```julia
invsqrtr(P)
```
Symmetric inverse of matrix square root. Equivalent to calling [`spfr`](@ref) with the [1/sqrt](@ref "Parametric and non-parametric functors") function.

**Examples** 

Use `invsqrtr` instead of `logr` in the examples of [`logr`](@ref).
"""
invsqrtr(X::AbstractMatrix{<:AbstractFloat}) = spfr(X, Invsqrt(); f′=Invsqrt′())


""" 
```julia
powerr(P, t)
```
Symmetric matrix power with fixed floating-point exponent `t`. Equivalent to calling [`spfr`](@ref) with the [x^t](@ref "Parametric and non-parametric functors") function.

!!! warning "Learnable `t`"
    This primal function is to be used when exponent `t` is **not** to be learnt by the layer using it.
    If `t` is to be learnt, use instead [`powerr_t`](@ref).

**Examples**
```julia
using ChainRulesSPD
using PosDefManifold, PosDefManifoldML # data generation

# Single-matrix
P = randP(3) # a random matrix of Julia's `Hermitian` type
t = 0.7
P_pow = powerr(P, t)

# batched version (Example: Tensor 3x3x10)
P_batch = hermVec2tensor(randP(3, 10)) 
P_pow_batch = powerr(P_batch, t)
```
"""
powerr(X::AbstractMatrix{T}, t::T) where {T<:AbstractFloat} = spfr(X, Power(t); f′=Power′(t))

"""
```julia
powerr_t(P, t)
```
Symmetric matrix power with learnable floating-point exponent `t`. Equivalent to calling [`spfr`](@ref) with the [x^t](@ref "Parametric and non-parametric functors") function.

!!! warning "Learnable `t`"
    This primal function is to be used when exponent `t` is to be learnt by the layer using it.
    If `t` is **not** to be learnt, use instead [`powerr`](@ref).

**Examples**

See the examples of [`powerr`](@ref), which can be used verbatim.
"""
powerr_t(X::AbstractMatrix{T}, t::T) where {T<:AbstractFloat} = spfr(X, Power(t); f′=Power′(t))


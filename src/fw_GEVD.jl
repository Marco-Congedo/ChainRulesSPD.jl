#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================

# ========================================================================================================
#   GEVD-based SPD operations.
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
#   back-propagation rules implemented in bw_GEVD.jl.
#
#   They allow the most efficient possible way to compute:
#   - The geodesic (including the geometric mean)
#   - The logarithmic map
#   - The exponential map
#   - The parallel transport
#
#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) September 2026
# ========================================================================================================

# NB: All functions are cached. Only phir, psir and gevdr don't do any caching. 
# That's fine because every function actually exposed for differentiation (the wrappers named above) 
# either has its own rrule or delegates entirely to one that does.

# ==========================================================================================
# Generalized eigenvalue decomposition
# ==========================================================================================

#=
    gevdr(P, Q; checks::Bool = true, precond::Bool = false
)

Generalized eigendecomposition Q*v = λ P*v satisfying

    P = A'A
    Q = A'ΛA
    B = A⁻¹.

Implements pre-conditioning. Use Julia eigen(Q, P) standard function,
which is the most efficient.

return

    λ : generalized eigenvalues
    B : (P-orthonormal) generalized eigenvectors

If `checks` = true, checks will be performed and warning will be printed if they fails.
If `precond` = true, the input matrices P and Q are pre-conditioned by congruence with 
    diagonal matrix inv(sqrt(Diagonal(P)) (Jacobi-like preconditioning). 
    This bounds D'PD's diagonal to all-ones before Cholesky ever sees it, shrinking the condition number
    of the Cholesky factor. This can improve the accuracy.

NB. In input matrices are not Hermitian or Symmetric they are typecasted to Hermitian
because in all pullback functions it is assumed so (see the use of desym there)    
=#
@inline function gevdr(
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    Psym = P isa SymOrHerm ? P : Hermitian(P)
    Qsym = Q isa SymOrHerm ? Q : Hermitian(Q)

    if precond       
        D = inv(sqrt(Diagonal(Psym)))
        F = eigen(Hermitian(D * Qsym * D), Hermitian(D * Psym * D))    
        return F.values, D * F.vectors
    else
        F = eigen(Qsym, Psym)     
        return F.values, F.vectors
    end
    
end


#=
function gevdr_phir(
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat}

return
    λ : generalized eigenvalues
    A : inverse of B (left eigenvectors of P^-1 * Q)
    B : (P-orthonormal) generalized eigenvectors (right eigenvectors of P^-1 * Q)

Version of gevdr suitable for the phi function.
It should not be used for the psi function, as it engenders a 10dB loss of accuracy.
Computes A as B' * Symmetric(Pcopy) (efficient computation of A)

NB. If input matrices are not Hermitian or Symmetric, they are typecasted to Hermitian
because in all pullback functions it is assumed so (see the use of desym there)
=#
function gevdr_phir(
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    λ, B = gevdr(P, Q; precond)
#    A = B' * Symmetric(copy(P)) # this is the efficient fix
    A = B' * Symmetric(P) # this is the efficient fix
    return λ, A, B
end


#=
function gevdr_psir(
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat}

return
    λ : generalized eigenvalues
    A : inverse of B (left eigenvectors of P^-1 * Q)
    B : (P-orthonormal) generalized eigenvectors (right eigenvectors of P^-1 * Q)

Version of gevdr suitable for the phi function. It can be used also for the phi function, but is slightly less efficient.
It computes A as inv(B) --- see the paper
=#
function gevdr_psir(
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    λ, B = gevdr(P, Q; precond)
    A = inv(B)
    return λ, A, B
end

# ==========================================================================================
# Auxiliary ϕ function
# ==========================================================================================

"""
```julia
function phir(P, Q, f; f′ = nothing, precond = false)
```
Generic method to compute **any function** with form 

``\\varphi(f, P, Q)=Pf(P^{-1}Q)=P^{1/2} f(P^{-1/2} Q P^{-1/2}) P^{1/2}``

based on the generalized eigendecomposition (GEVD) of the pair ``(P, Q)``, yielding

``\\varphi(f, P, Q)=A^T f(\\Lambda) A``.

The ``\\varphi`` function allows to compute the geodesic, which includes the geometric mean, as well as the logarithmic and exponential map.
(See paper xxx).

!!! tip "Use wrappers"
    This function is rarely needed; for usual choices of ``f`` use wrappers:
    
    [`geodesicr`](@ref), [`geomeanr`](@ref), [`logmapr`](@ref), [`expmapr`](@ref).

``P`` and ``Q`` can be either ``n \\times n`` matrices or ``n \\times n \\times m`` batches (tensors),
holding ``m`` of such matrices (slices) of floating-point elements. 

The matrix(ces) in ``P`` must be ``\\in \\mathcal{S}_{++}``, while the matrix(ces) in ``Q`` may be 
``\\in \\mathcal{S}_{++}`` or ``\\in \\mathcal{S}``, depending on ``f``.

``f`` can be any scalar function or one of these [functors](@ref "Parametric and non-parametric functors").

Keyword argument ``f'`` is used only if the `rrule` for `phir` is needed (e.g., if a call to `phir` is used in a network).
If you pass ``f'``, this must be a function or functor computing the derivative of ``f``.
If you pass it, it will be used, otherwise it will be derived automatically.

See [pre-conditioning](@ref "Pre-Conditioning") for keyword argument `precond`.

**Examples**
```julia
using ChainRulesSPD
using PosDefManifold, PosDefManifoldML # data generation

P, Q = randP(3), randP(3) # random matrices of Julia's `Hermitian` type
ξ = log(Q) # tangent vector (symmetric)
t = 0.1

p = phir(P, Q, Power(t); f′=Power′(t)) # geodesic
g = phir(P, Q, Sqrt(); f′=Sqrt′()) # geometric mean
l = phir(P, Q, Log(); f′=Log′()) # logarithmic map
e = phir(P, ξ, Exp(); f′=Exp′()) # Exponential map

# batched version (Example: Tensor 3x3x10)
P_batch = hermVec2tensor(randP(3, 10)) 
Q_batch = hermVec2tensor(randP(3, 10)) 

p_batch = phir(P_batch, Q_batch, Power(t); f′=Power′(t)) # geodesic

# Example with a custom function
myf = phir(P, Q, x->x^2+x; f′=x->2*x+1) 
```
"""
function phir(
        P::AbstractMatrix{T}, 
        Q::AbstractMatrix{T},
        f::Function;

        f′::Union{Nothing,Function} = nothing, # dummy, used to have same signature as rrule(::phir,...)
        precond::Bool = false
    ) where {T<:AbstractFloat}

    λ, A, _ = gevdr_phir(P, Q; precond)

    return sym(A' * Diagonal(f.(λ)) * A)
end


# ==========================================================================================
# Wrappers for common GEVD functions
# Note: these functions are not intercepted by AD since no corresponding rrule exists,
# however,  `phir` is intercepted and the corresponding rrule is used.
# ==========================================================================================

""" 
```julia
function geodesicr(P, Q, t; precond = false)
```
Affine-invariant geodesic ``P \\#_t Q`` with fixed arc-length parameter ``t``. 

With ``t=0`` returns ``P``, with ``t=1`` returns ``Q``, and with ``t=\\frac{1}{2}`` 
returns the geometric mean (see [`geomeanr`](@ref)).

``t`` can be any floating-point number in the whole real line.

``P`` and ``Q`` can be either ``n \\times n`` matrices or ``n \\times n \\times m`` batches (tensors),
holding ``m`` of such matrices (slices) of floating-point elements. 

The matrix(ces) in both ``P`` and ``Q`` must be ``\\in \\mathcal{S}_{++}``.

See [pre-conditioning](@ref "Pre-Conditioning") for keyword argument `precond`.

!!! warning "Learnable `t`"
    This primal function is to be used when exponent `t` is **not** to be learnt by the layer using it.
    If `t` is to be learnt, use instead [`geodesicr_t`](@ref).

**Examples**
```julia
using ChainRulesSPD
using PosDefManifold, PosDefManifoldML # data generation

P, Q = randP(3), randP(3) # random matrices of Julia's `Hermitian` type
t = 0.1

g = geodesicr(P, Q, t)

# batched version (Example: Tensor 3x3x10)
P_batch = hermVec2tensor(randP(3, 10)) 
Q_batch = hermVec2tensor(randP(3, 10)) 

g_batch = geodesicr(P_batch, Q_batch, t)
```
"""
geodesicr(
        P::AbstractMatrix{T}, 
        Q::AbstractMatrix{T}, 
        t::T;

        precond::Bool = false
    ) where {T<:AbstractFloat} =  phir(P, Q, Power(t); f′ = Power′(t), precond)


""" 
```julia
function geodesicr_t(P, Q, t; precond = false)
```
Affine-invariant geodesic ``P \\#_t Q`` with learnable arc-length parameter ``t``. 

It accepts the same input and uses the same syntax as [`geodesicr`](@ref).

!!! warning "Learnable `t`"
    This primal function is to be used when exponent `t` is to be learnt by the layer using it.
    If `t` is **not** to be learnt, use instead [`geodesicr`](@ref).

**Examples**

See the examples of [`geodesicr`](@ref), which can be used verbatim.
"""
geodesicr_t(
        P::AbstractMatrix{T}, 
        Q::AbstractMatrix{T}, 
        t::T;

        precond::Bool = false
    ) where {T<:AbstractFloat} =  phir(P, Q, Power(t); f′ = Power′(t), precond)


""" 
```julia
function geomeanr(P, Q; precond = false)
```
Geometric mean ``P \\# Q``.

It is the same as a call to [`geodesicr`](@ref) with arc-length parameter ``t=\\frac{1}{2}``. 

``P`` and ``Q`` can be either ``n \\times n`` matrices or ``n \\times n \\times m`` batches (tensors),
holding ``m`` of such matrices (slices) of floating-point elements. 

The matrix(ces) in both ``P`` and ``Q`` must be ``\\in \\mathcal{S}_{++}``.

See [pre-conditioning](@ref "Pre-Conditioning") for keyword argument `precond`.

**Examples**
```julia
using ChainRulesSPD
using PosDefManifold, PosDefManifoldML # data generation

P, Q = randP(3), randP(3) # random matrices of Julia's `Hermitian` type

m = geomeanr(P, Q)

# batched version (Example: Tensor 3x3x10)
P_batch = hermVec2tensor(randP(3, 10)) 
Q_batch = hermVec2tensor(randP(3, 10)) 

m_batch = geomeanr(P_batch, Q_batch)
```
"""
geomeanr(
        P::AbstractMatrix{T}, 
        Q::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat} = phir(P, Q, Sqrt(); f′ = Sqrt′(), precond)


""" 
```julia
function logmapr(P, Q; precond = false)
```
Logarithmic map ``\\mathrm{Log}_P(Q)`` of ``Q \\in \\mathcal{S}_{++}`` onto the tangent vector at base point ``P``.

It accepts the same input and uses the same syntax as [`geomeanr`](@ref). 

**Examples**

Use `logmapr` instead of [`geomeanr`](@ref) in the examples there.

"""
logmapr(
        P::AbstractMatrix{T}, 
        Q::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat} = phir(P, Q, Log(); f′ = Log′(), precond) 


""" 
```julia
function expmapr(P, ξ; precond = false)
```
Exponential map ``\\mathrm{Exp}_P(\\xi)`` of tangent vector ``\\xi \\in \\mathcal{S}`` 
in the tangent space at base point ``P`` onto the manifold.

It uses the same syntax as [`geomeanr`](@ref), with `ξ` playing the role of ``Q``. 

The matrix(ces) in both ``P`` must be ``\\in \\mathcal{S}_{++}``, while the matrix(ces) 
in ``\\xi`` can be ``\\in \\mathcal{S}``.

**Examples**

Use `expmapr` instead of [`geomeanr`](@ref) in the examples there.

"""
expmapr(
        P::AbstractMatrix{T}, 
        ξ::AbstractMatrix{T};

        precond::Bool = false            
    ) where {T<:AbstractFloat} = phir(P, ξ, Exp(); f′ = Exp′(), precond)



# ==========================================================================================
# Numerically robust exponential map (row-decomposed log-sum-exp reconstruction)
# Experimental
# ==========================================================================================

#=
    expmapr_stable(P, Ξ; precond::Bool=true)

Numerically robust version of the exponential map

    expmapr(P, Ξ) = phir(P, Ξ, Exp())

Mathematically identical to `expmapr`, but avoids overflow in the
reconstruction step when some generalized eigenvalue `λ_i` (of the
pencil `(Ξ,P)`) is large enough that `exp(λ_i)` overflows `Float64` on
its own, even though the true output `Y = A' exp(Λ) A` may still be
finite.

Writing each row of `A` as `A[i,:] = n_i * u_i`, with `n_i = ‖A[i,:]‖`
and `u_i` a unit vector, mode `i`'s contribution to `Y` is
`exp(λ_i) * n_i^2 * u_i u_i'`. Rather than forming `exp(λ_i)` and `n_i^2`
separately (which overflows the moment `exp(λ_i)` alone exceeds
`floatmax(T)`), the two magnitudes are combined ADDITIVELY in log-space,

    L_i = λ_i + 2*log(n_i),

and only a single `exp(L_i)` is evaluated on the (typically much more
moderate) combined exponent — a per-mode log-sum-exp trick, equivalent
to choosing a non-identity generalized eigenvector normalization
(V'PV = D_p ≠ I) without touching `gevdr`'s own P-orthonormal
convention.

Verified numerically against a high-precision (mpmath) reference:
recovers a finite, correctly-scaled answer in regimes where the naive
`phir(P,Ξ,Exp())` reconstruction returns Inf/NaN. Any residual error in
such regimes is fully explained by the exponential map's own
conditioning (cond(exp,λ)=|λ|) amplifying whatever small error the
GEVD itself already carries in λ — this trick removes the AVOIDABLE
reconstruction overflow, not that INTRINSIC sensitivity, which is a
property of the map itself, not of this implementation.
=#
function expmapr_stable(
        P::AbstractMatrix{T}, 
        Ξ::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    λ, A, B = gevdr_phir(P, Ξ; precond)

    n = length(λ)
    row_norms = [norm(@view A[i, :]) for i in 1:n]
    L     = λ .+ T(2) .* log.(row_norms)
    coeff = exp.(L)
    U     = A ./ row_norms   # unit-norm rows

    return sym(U' * Diagonal(coeff) * U)
end


# ==========================================================================================
# Auxiliary ψ function
# ==========================================================================================
"""
```julia
function psir(P, Q, f; f′ = nothing, precond = false)
```
Generic method to compute **any function** with form 

``\\psi(f, P, Q)=f(P^{-1}Q)``

based on the generalized eigendecomposition (GEVD) of the pair ``(P, Q)``, yielding

``\\psi(f, P, Q)=B T f(\\Lambda) A``.

The ``\\psi`` function allows to compute the parallel transport
(See paper xxx).

!!! tip "Use wrapper"
    This function is rarely needed; for the parallel transport use [`parallel_transportr`](@ref).      

``P`` and ``Q`` can be either ``n \\times n`` matrices or ``n \\times n \\times m`` batches (tensors),
holding ``m`` of such matrices (slices) of floating-point elements. 

The matrix(ces) in ``P`` must be ``\\in \\mathcal{S}_{++}``, while the matrix(ces) in ``Q`` may be 
``\\in \\mathcal{S}_{++}`` or ``\\in \\mathcal{S}``, depending on ``f``.

``f`` can be any scalar function or one of these [functors](@ref "Parametric and non-parametric functors").

Keyword argument ``f'`` is used only if the `rrule` for `psir` is needed (e.g., if a call to `psir` is used in a network).
If you pass ``f'``, this must be a function or functor computing the derivative of ``f``.
If you pass it, it will be used, otherwise it will be derived automatically.

See [pre-conditioning](@ref "Pre-Conditioning") for keyword argument `precond`.

**Examples**
```julia
using ChainRulesSPD
using PosDefManifold, PosDefManifoldML # data generation

# random matrices of Julia's `Hermitian` type
P, Q, S = randP(3), randP(3), randP(3)  
t = 0.5

T = psir(P, Q, Sqrt(); f′=Sqrt′()) 
pt = T' * S *T # or, 

# More simply:
pt = parallel_transportr(P, Q, S)

# For a tangent vector:
ζ = log(S) # tangent vector
pt2 = parallel_transportr(P, Q, ζ)

# batched version (Example: Tensor 3x3x10)
P_batch = hermVec2tensor(randP(3, 10)) 
Q_batch = hermVec2tensor(randP(3, 10)) 
S_batch = hermVec2tensor(randP(3, 10)) 

T_batch = psir(P_batch, Q_batch, Sqrt(); f′=Sqrt′()) 
```
"""
function psir(
            P::AbstractMatrix{T},
            Q::AbstractMatrix{T},
            f::Function;

            f′::Union{Nothing,Function} = nothing, # dummy, used to have same signature as rrule(::psir,...)
            precond::Bool = false
    ) where {T<:AbstractFloat}

    λ, A, B = gevdr_psir(P, Q; precond)
    return B * Diagonal(f.(λ)) * A # can xe use (B * Diagonal(f.(λ)) / B ?
end


#=
    _psirGEVD(P,Q,f; precond::Bool = false)

Return λ, A, B, Y with Y = psir(P, Q, f).
This is used internally in rrule functions
=#
function _psirGEVD(
            P::AbstractMatrix{T},
            Q::AbstractMatrix{T},
            f::Function;
            
            f′::Union{Nothing,Function} = nothing,
            precond::Bool = false
    ) where {T<:AbstractFloat}

    λ, A, B = gevdr_psir(P, Q; precond)
    Y = B * Diagonal(f.(λ)) * A # can xe use (B * Diagonal(f.(λ)) / B ?
    return λ, A, B, Y
end



# ==========================================================================================
# Parallel transport
# ==========================================================================================

"""
```julia
function parallel_transportr(P, Q, S; precond = false)
```
Parellel transport 

- ``\\mathcal{P}_{P \\to Q}(\\zeta)``: parallel transport of tangent vector ``\\zeta \\in \\mathcal{S}`` from the tangent space at base point ``P`` to the tangent space at base point ``Q``.
- ``\\mathcal{P}_{P \\to Q}(S)``: parallel transport of a point ``S \\in \\mathcal{S}_{++}`` along the geodesic from ``P`` to ``Q``.

``P`` and ``Q`` can be either ``n \\times n`` matrices or ``n \\times n \\times m`` batches (tensors),
holding ``m`` of such matrices (slices) of floating-point elements. 

The matrix(ces) in both ``P`` and ``Q`` must be ``\\in \\mathcal{S}_{++}``.

See [pre-conditioning](@ref "Pre-Conditioning") for keyword argument `precond`.

**Examples**
```julia
using ChainRulesSPD
using PosDefManifold, PosDefManifoldML # data generation

# random matrices of Julia's `Hermitian` type
P, Q, S = randP(3), randP(3), randP(3)  
t = 0.5
ζ = log(S) # tangent vector

ptS = parallel_transportr(P, Q, S)
ptζ = parallel_transportr(P, Q, ζ)

# batched version (Example: Tensor 3x3x10)
P_batch = hermVec2tensor(randP(3, 10)) 
Q_batch = hermVec2tensor(randP(3, 10)) 
S_batch = hermVec2tensor(randP(3, 10)) 
ζ_batch = hermVec2tensor(HermitianVector(log.randP(3, 10)))
# `HermitianVector` is declared in PosDefManifold.jl; 
# the input of `hermVec2tensor` must be of this type

ptS_batch = parallel_transportr(P_batch, Q_batch, S_batch)
ptζ_batch = parallel_transportr(P_batch, Q_batch, ζ_batch)
```
"""
function parallel_transportr(   
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T},
        S::AbstractMatrix{T};

        precond::Bool = false            
    ) where {T<:AbstractFloat}

    E = psir(P, Q, Sqrt(); f′=Sqrt′(), precond)
    return sym(E' * S * E)
end
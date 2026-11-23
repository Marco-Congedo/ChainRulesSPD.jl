# ========================================================================================================
#   Custom rules to be applied in back-propagation for the GEVD functions defined in `fw_GEVD.jl`
#
#   All rules can take as input objects of type Matrix, Symmetric or Hermitian type. See test_bp_rules_GEVD.jl.
#   Each rule returns the primal and the `pullback` function, to be used later by any AutoDiff system.
#   
#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================



# ==========================================================================================
# Common forward + pullback for the single-matrix ϕ wrappers
# (same math as the rrule of `phir`; `f` and `f′` are provided by each wrapper).
#
# Returns (Y, pullback), where
#     pullback(Ȳ) = (NoTangent(), ΔP, ΔQ, NoTangent())
# the last slot being the tangent of `f`, or of `t` in `geodesicr`.
# ==========================================================================================
#=
#### OLD version
function _phir_rrule(
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T},
        f::Function,
        f′::Function;

        precond::Bool = false
    ) where {T<:AbstractFloat}

    λ, A, B = gevdr_phir(P, Q; precond)

    fλ = f.(λ)
    Y  = sym(A' * Diagonal(fλ) * A)     # primal ϕ(f,P,Q) = A' f(Λ) A
    K  = loewner(λ, f; f′)              # Löwner matrix, cached for the pullback

    function pullback(Ȳ)

        J  = sym(A * ChainRulesCore.unthunk(Ȳ) * A')     # J = A Ȳ A'
        H  = K .* J                                      # K ⊙ J
        ΔQ = sym(B * H * B')                             # ∂ℓ/∂Q
        ΔP = sym(B * sym(J .* fλ' - H .* λ') * B')       # ∂ℓ/∂P

        desym!(P, ΔP)
        desym!(Q, ΔQ)

        return (NoTangent(), ΔP::Matrix{T}, ΔQ::Matrix{T}, NoTangent())
    end

    return Y, pullback
end
=#
function _phir_rrule(
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T},
        f::Function,
        f′::Function;

        precond::Bool = false
    ) where {T<:AbstractFloat}

    λ, A, B = gevdr_phir(P, Q; precond)

    fλ = f.(λ)
    Y  = sym(A' * Diagonal(fλ) * A)     # primal ϕ(f,P,Q) = A' f(Λ) A
    K  = loewner(λ, f; f′)              # Löwner matrix, cached for the pullback
    Z = @. T(0.5) * ((fλ + fλ') - (λ + λ') * K)   # build in the forward pass a,d cached, O(n²)

    function pullback(Ȳ)

        J  = sym(A * unthunk(Ȳ) * A')       # sym needed: Z-form assumes symmetric J
        ΔQ = sym(B * (K .* J) * B')         # ∂ℓ/∂Q = B(K⊙J)Bᵀ
        ΔP = sym(B * (Z .* J) * B')         # ∂ℓ/∂P = B(Z⊙J)Bᵀ

        desym!(P, ΔP)
        desym!(Q, ΔQ)

        return (NoTangent(), ΔP::Matrix{T}, ΔQ::Matrix{T}, NoTangent())
    end

    return Y, pullback
end



#=
Back Propagation rule for the ϕ function

pullback for:   ϕ(f,P,Q) = A' f(Λ) A, where

    P = A'A,
    Q = A'ΛA,
    B = A⁻¹.

Computes ΔP and ΔQ according to Proposition 2 in Congedo, Bhatia and Bouchard (2027).

---------
B,A : Generalized eigenvector matrix (B) and its inverse (A).
λ: Generalized eigenvalues.
fλ: Vector f.(λ).
K: Loewner matrix of f evaluated at λ.
Ȳ: Upstream sensitivity.

Note: denote Δϕ = ∂ℓ/∂ϕ, ΔQ = ∂ℓ/∂Q and ΔP = ∂ℓ/∂P. We have
ΔQ = B (K⊙G) B' 
ΔP ​= B sym (A Δϕ ​A' f(Λ) − A ΔQ ​A' Λ) B' = B sym (A Δϕ ​A' f(Λ) − (K⊙G) Λ) B'
=#
ChainRulesCore.rrule(::typeof(phir),
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T},
        f::Function; 

        f′::Union{Nothing,Function} = nothing,
        precond::Bool = false
    ) where {T<:AbstractFloat} = _phir_rrule(P, Q, f, _derivative(f, f′); precond)


# ==========================================================================================
# Geodesic without learnable arc-length parameter
# ==========================================================================================
function ChainRulesCore.rrule(::typeof(geodesicr),
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T},
        t::T;

        precond::Bool = false
    ) where {T<:AbstractFloat}

    # pullback = (NoTangent(), ΔP, ΔQ, NoTangent()), the last one being for t
    return _phir_rrule(P, Q, Power(t), Power′(t); precond)
end


# ==========================================================================================
# Geometric mean
# ==========================================================================================
function ChainRulesCore.rrule(::typeof(geomeanr),
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    Y, pb = _phir_rrule(P, Q, Sqrt(), Sqrt′(); precond)

    return Y, _drop_last_tangent(pb)
end


# ==========================================================================================
# Logarithmic map
# ==========================================================================================
function ChainRulesCore.rrule(::typeof(logmapr),
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    Y, pb = _phir_rrule(P, Q, Log(), Log′(); precond)

    return Y, _drop_last_tangent(pb)
end


# ==========================================================================================
# Exponential map
# ==========================================================================================
function ChainRulesCore.rrule(::typeof(expmapr),
        P::AbstractMatrix{T},
        Ξ::AbstractMatrix{T};

        precond::Bool = false
    ) where {T<:AbstractFloat}

    Y, pb = _phir_rrule(P, Ξ, Exp(), Exp′(); precond)

    return Y, _drop_last_tangent(pb)
end


#=
Specific back-propagation rule for the geodesic when the gradient for the arc-length parameter is to be returned.
Backward rule of forward function `geodesicr_t`.

    In addition to the gradients for P and Q, it returns the gradient for t:

        ∂t/∂ℓ=∑_i (G_ii λ_i^t log(λ_i))

=#
function ChainRulesCore.rrule(::typeof(geodesicr_t),
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T},
        t::T;
        precond::Bool = false
    ) where {T<:AbstractFloat}

    f = Power(t)

    λ, A, B = gevdr_phir(P, Q; precond)
    fλ = f.(λ)
    Y = sym(A' * Diagonal(fλ) * A)
    K  = loewner(λ, f; f′ = derivative(f))

    function pullback(Ȳ)
        J = sym(A * ChainRulesCore.unthunk(Ȳ) * A')
        H = K .* J
        ΔQ = sym(B * H * B')
        ΔP = sym(B * sym(J .* fλ' - H .* λ') * B')
        Δt = dot(diag(J), fλ .* log.(λ))

        desym!(P, ΔP)
        desym!(Q, ΔQ)

        return (NoTangent(), ΔP, ΔQ, Δt)
    end

    return Y, pullback
end


#=
Experimental:
Back-propagation rule for `expmapr_stable(P, Ξ; precond::Bool = false)`.

Reuses the SAME `(λ, A, B)` cached from the single GEVD of `(P, Ξ)` used
by the stable forward pass — no second decomposition. The backward
formula is mathematically identical to `phir`'s own rrule for `Exp()`:
the row-decomposition only changes the ARITHMETIC used to form the
primal `Y` from `λ,A` (to avoid overflow), not the mathematical map
itself, so the Fréchet-derivative formula is unchanged.

⚠ CAVEAT: `fλ = exp.(λ)` and the diagonal of the Loewner matrix
`K[i,i] = exp(λ_i)` computed below carry the SAME class of overflow
risk that motivated stabilizing the forward pass. This rrule is exact
whenever those intermediate quantities remain finite. For `λ_i` extreme
enough to overflow `exp.(λ)` directly, a fully robust backward pass
would need an analogous PAIRWISE log-space stabilization of the
Loewner matrix (harder than the forward fix, since `K[i,j]` mixes
`λ_i` and `λ_j` rather than a single eigenvalue) — NOT implemented
here. This rrule should be treated as correct in the same regime as
`phir`'s own `Exp()` rrule, and as the caching-consistent match for
`expmapr_stable`'s primal — not as a complete fix for the backward
pass's own, separate overflow risk.
=#
function ChainRulesCore.rrule(::typeof(expmapr_stable),
        P::AbstractMatrix{T},
        Ξ::AbstractMatrix{T};
        precond::Bool = false
    ) where {T<:AbstractFloat}

    λ, A, B = gevdr_phir(P, Ξ; precond)

    n = length(λ)
    row_norms = [norm(@view A[i, :]) for i in 1:n]
    L     = λ .+ T(2) .* log.(row_norms)
    coeff = exp.(L)
    U     = A ./ row_norms

    Y = sym(U' * Diagonal(coeff) * U)

    fλ = exp.(λ)                              # see caveat above
    K  = loewner(λ, Exp(); f′ = Exp′())        # see caveat above

    function pullback(Ȳ)

        J  = sym(A * ChainRulesCore.unthunk(Ȳ) * A')
        H  = K .* J
        ΔΞ = sym(B * H * B')
        ΔP = sym(B * sym(J .* fλ' - H .* λ') * B')

        desym!(P, ΔP)
        desym!(Ξ, ΔΞ)

        return (NoTangent(), ΔP::Matrix{T}, ΔΞ::Matrix{T})
    end

    return Y, pullback
end


############################################################

# Helper psir_pullback
function _psir_pullback(
        A::AbstractMatrix{T},
        B::AbstractMatrix{T},
        λ::AbstractVector{T},
        fλ::AbstractVector{T},
        K::AbstractMatrix{T},
        Ē
    ) where {T<:AbstractFloat}

    J = A * ChainRulesCore.unthunk(Ē)' * B              # J = A Ē B, , Ē = Δψ = ∂ℓ/∂ψ (upstream gradient)
    H = K .* sym(J)                                     # Ĵ = K ⊙ sym(J) = A ∂ℓ/∂Q A'
    ΔQ = sym(B * H * B')                                # B (K ⊙ sym(A Ē A')) B' = ∂ℓ/∂Q
    ΔP = sym(B * sym(skew(J) .* fλ' - H .* λ' ) * B')   # B sym(skew(J) f(Λ) - (K ⊙ J) Λ) B' = = ∂ℓ/∂P
    # NB: The inner sym in ΔP above is most probably useless. I leave it as it costs only O(n²)

    return ΔP, ΔQ
end


#=
Back Propagation rule for the ψ function

Common pullback for

    ψ(f,P,Q; precond::Bool = false) 
    
= B f(Λ) A.

Compute ΔP and ΔQ according to Proposition 3 of Congedo, Bhatia and Bouchard (2027).
The skew-symmetric contribution is intrinsic to ψ because
its codomain is the full matrix space.
Support any ψ, not just with f=sqrt (for doing parallel transport)
=#
function ChainRulesCore.rrule(::typeof(psir),
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T},
        f::Function;

        f′::Union{Nothing,Function} = nothing,
        precond::Bool = false           
    ) where {T<:AbstractFloat}

    λ, A, B, Y = _psirGEVD(P, Q, f; precond)

    fλ = f.(λ)
    K  = loewner(λ, f; f′)

    function pullback(Ȳ)

        ΔP, ΔQ = _psir_pullback(A, B, λ, fλ, K, Ȳ)
        desym!(P, ΔP)
        desym!(Q, ΔQ)

        return (NoTangent(), ΔP::Matrix{T}, ΔQ::Matrix{T}, NoTangent())
    end

    return Y, pullback
end



#=
Back Propagation rule for parallel_transportr
=#
function ChainRulesCore.rrule(::typeof(parallel_transportr),
        P::AbstractMatrix{T},
        Q::AbstractMatrix{T},
        S::AbstractMatrix{T};

        precond::Bool = false   
    ) where {T<:AbstractFloat}

    f = Sqrt()
    f′ = Sqrt′()

    λ, A, B, E = _psirGEVD(P, Q, f; f′, precond)
    fλ = f.(λ)
    K  = loewner(λ, f; f′)
    Y = sym(E' * S * E)

    function pullback(Ȳ)

        X̄ = Matrix(ChainRulesCore.unthunk(Ȳ))

        # gradient w.r.t S
        ΔS = sym(E * X̄ * E')
        #ΔS = E * sym(X̄) * E'

        # gradient w.r.t E
        Ē = S * E * (X̄ + X̄')

        ΔP, ΔQ = _psir_pullback(A, B, λ, fλ, K, Ē)

        !(S isa Matrix) && desym!(S, ΔS) # call desym! only if S is Symmetric or Hermitian
        desym!(P, ΔP)
        desym!(Q, ΔQ)

        return (NoTangent(), ΔP::Matrix{T}, ΔQ::Matrix{T}, ΔS::Matrix{T})
    end

    return Y, pullback
end
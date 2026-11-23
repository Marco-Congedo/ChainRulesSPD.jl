# Special rules for divided differences to be used to construct the Loewner matrix;


# ------------------------------------------------------------------------------------------
# (3) Divided differences for exp, log, power: one formula each, no series, no cutoffs,
#     no ArgumentError for missing cutoffs, valid for Float32/Float64 alike.
#
#     exp:   (eᵃ-eᵇ)/(a-b) = eʰⁱ · expm1(d)/d,           hi=max(a,b), lo=min(a,b), d=lo-hi ≤ 0
#     log:   (ln a-ln b)/(a-b) = log1p(w)/(hi-lo),        w=(hi-lo)/lo   (w computed directly)
#     power: (aᵖ-bᵖ)/(a-b) = loᵖ·expm1(p·log1p(w))/(hi-lo)
#
#     Here log1p/expm1 get the small quantity itself (w, d), which is what they are for.
#     Ordering hi/lo makes the result bitwise symmetric in (a,b) and avoids overflow beyond
#     exp(hi). a == b is handled exactly (limit = f′).
# ------------------------------------------------------------------------------------------
function divided_diff(::Exp, a::T, b::T; args...) where {T<:AbstractFloat}
    hi, lo = a ≥ b ? (a, b) : (b, a)
    d = lo - hi
    return iszero(d) ? exp(hi) : exp(hi) * (expm1(d) / d)
end

divided_diff(::typeof(exp), a::T, b::T; args...) where T <: AbstractFloat = 
    divided_diff(Exp(), a, b; args...)

function divided_diff(::Log, a::T, b::T; args...) where {T<:AbstractFloat}
    hi, lo = a ≥ b ? (a, b) : (b, a)
    lo > 0 || throw(DomainError("ChainRulesSPD.loewner (with log function): one eigenvalue is not positive: $lo" ))
    hi == lo && return inv(hi)
    w = (hi - lo) / lo                          # ≥ 0
    # w overflows only if hi/lo > floatmax (denormal lo): use the textbook form there
    return isfinite(w) ? log1p(w) / (hi - lo) : (log(hi) - log(lo)) / (hi - lo)
end

divided_diff(::typeof(log), a::T, b::T; args...) where T <: AbstractFloat = 
    divided_diff(Log(), a, b; args...)


function divided_diff(f::Power, a::T, b::T; args...) where {T<:AbstractFloat}
    p = T(f.p)                                  # decouples the type of p from that of λ
    iszero(p) && return zero(T)
    hi, lo = a ≥ b ? (a, b) : (b, a)
    lo > 0 || throw(DomainError("ChainRulesSPD.loewner (with power function): one eigenvalue is not positive: $lo" ))
    hi == lo && return p * hi^(p - one(T))
    w = (hi - lo) / lo
    isfinite(w) || return (hi^p - lo^p) / (hi - lo)
    return lo^p * expm1(p * log1p(w)) / (hi - lo)
end

function divided_diff(::typeof(Sqrt()), 
            a::T, 
            b::T; 
            
            args...
    ) where T <: AbstractFloat

    # println("Sqrt brach, args =$(args...)")

    return one(T) / (sqrt(a) + sqrt(b))
end

divided_diff(::typeof(sqrt), a::T, b::T; args...) where T <: AbstractFloat = 
    divided_diff(Sqrt(), a, b; args...)


# f(x) = 1/√x 
# Exact: (x^(-1/2) - y^(-1/2)) / (x - y) = -1 / (√x · √y · (√x + √y))
# Addition-only in the denominator: no cancellation, continuous at x = y,
# with limit -1/(2x^(3/2)) = f′(x), exactly the analytic derivative.
divided_diff(::typeof(Invsqrt()), a::T, b::T; args...) where T <: AbstractFloat = 
    -one(T) / (sqrt(a) * sqrt(b) * (sqrt(a) + sqrt(b)))


# ------------------------------------------------------------------------------------------
# Generic fallback and loewner (the cutoffs coExp/coLog/args are no longer needed)
#     - abs() on the eigenvalues: with `max(λi, λj)` the threshold is NEGATIVE for negative
#       eigenvalues (tangent vectors in expmapr), so equal eigenvalues gave 0/0 = NaN.
#     - `≤` instead of `<`: covers λi = λj = 0 (threshold 0).
#     - cbrt(eps) instead of sqrt(eps) balances the two errors (worst case 8e-12 vs 3e-9).
# ------------------------------------------------------------------------------------------
@inline function _divided_diff_generic(f, df, a::T, b::T, epsilon::T) where {T<:AbstractFloat}
    den = a - b
    return abs(den) ≤ epsilon * max(abs(a), abs(b)) ? (df(a) + df(b)) / 2 : (f(a) - f(b)) / den
end

"""
```julia
function loewner(
        λ::AbstractVector{T},
        f::Function;

        f′::Union{Function,Nothing} = nothing,
        epsilon::Real = cbrt(eps(T))
    ) where {T<:AbstractFloat}
```

Löwner (finite difference) matrix, defined here above, given the vector ``\\lambda`` of eigenvalues and a scalar function ``f``
or a [functor](@ref "Parametric and non-parametric functors"). The use of functors is preferred.

If ``f'`` is not provided and ``f`` is one of these functors `{Exp(), Sqrt(), Log(), Invsqrt(), Power(t)}`, 
the derivative ``f'`` is assigned automatically,
otherwise it is computed using [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl).

For the special functors above, precise computations are carried-out. See the Appendix E.1 in paper xxx for details.

For the others, the usual cut-off ``\\epsilon``, given by optional keyword argument `epsilon`, is used:

if ``i \\ne j`` and ``\\left|\\lambda_i - \\lambda_j\\right| \\le \\epsilon \\times \\text{max}(\\left|\\lambda_i\\right|, \\left|\\lambda_j\\right|)``, then

``(K_{f,\\lambda})_{ij} = \\frac{f'(\\lambda_i) + f'(\\lambda_j)}{2}``.

"""
function loewner(
        λ::AbstractVector{T},
        f::Function;

        f′::Union{Function,Nothing} = nothing,
        epsilon::Real = cbrt(eps(T))
    ) where {T<:AbstractFloat}

    n  = length(λ)
    K  = Matrix{T}(undef, n, n)
    df = _derivative(f, f′) # see pFunctions.jl
    special = f isa Union{Exp, Sqrt, Log, Invsqrt, Power} || f === exp || f === log || f === sqrt
    ϵ  = T(epsilon)

    @inbounds for i in 1:n
        K[i, i] = df(λ[i])
        for j in 1:i-1                          # one triangle; mirrored below (exact symmetry)
            kᵢⱼ = special ? divided_diff(f, λ[i], λ[j]) :
                            _divided_diff_generic(f, df, λ[i], λ[j], ϵ)
            K[i, j] = kᵢⱼ
            K[j, i] = kᵢⱼ
        end
    end
    return K
end
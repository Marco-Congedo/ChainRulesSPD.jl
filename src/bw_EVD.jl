# ========================================================================================================
#   Custom rules to be applied in back-propagation for the SPD functions defined in `fw_EVD.jl`
#
#   All rules can take as input a Matrix, Symmetric or Hermitian type. See test_bp_rules_EVD.jl.
#   Each rule returns the primal and the `pullback` function, to be used later by any AutoDiff system.
#   
#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) - Nov 2025
#
#   NB: Numerical argument that are not to be learnt receive a NoTangent() (see e.g. `relur`), 
#   If they are to be learnt they receive a gradient (see e.g., `powerr_t`)

#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================

#=
    _spfr_pullback(V, λ, f, Ȳ; f′ = nothing)

Common pullback for EVD-based spectral functions

    Y = V * Diagonal(f.(λ)) * V'

Computes the cotangent ΔX given the upstream sensitivity Ȳ.

Arguments
---------
V  : eigenvectors
λ  : eigenvalues
f  : scalar spectral function
f′ : optional derivative of f (if not provided it will be derived by ForwardDiff.jl)
Ȳ  : upstream sensitivity

Returns
-------
ΔX
=#
function _spfr_pullback(
        V, 
        λ, 
        f::Function; 
        
        f′::Union{Nothing,Function} = nothing
    )

    function pullback(Ȳ)

        ΔX = _spfr_frechet(V, λ, Ȳ, f; f′)

        return NoTangent(), ΔX
    end

    return pullback
end


#=
    ChainRulesCore.rrule(::typeof(spfr), X, f; ϵ = sqrt(eps(T)))

Back-prop rule for a generic spectral function f(X) = Uf(λ)U', with no
assumed analytic derivative for `f`. This is the fallback rule used
whenever a user calls `spfr` directly with an arbitrary `f` that has
no dedicated wrapper (`relur`, `logr`, `expr`, ...) and therefore no
dedicated rrule of its own.

If no function for f′ is supplied, `loewner` must derive it itself
via automatic differentiation of `f`; this is
slower and potentially less numerically stable than the wrappers with
a hand-supplied f′, so prefer the named wrappers when one exists
or write one for your specific function.
=#
function ChainRulesCore.rrule(::typeof(spfr), 
        X::AbstractMatrix{T}, 
        f::Function;
        
        f′::Union{Nothing,Function} = nothing,
        ϵ = sqrt(eps(T)::T)
    ) where {T<:AbstractFloat}

    λ, V, Y = _spfrEVD(X, f; ϵ)
    pullback = _spfr_pullback(V, λ, f; f′)

    function pullback_spfr(Ȳ)
        _, ΔX = pullback(Ȳ)
        return NoTangent(), ΔX, NoTangent()
    end

    return Y, pullback_spfr
end

################################################################################
# These rules are not necessary as they fall back to spfr, 
# but are kept for direct testing
################################################################################

#=
relur back-prop of λ <- max(λ, =1e-8)
=#
function ChainRulesCore.rrule(::typeof(relur), 
        X::AbstractMatrix{T}, 
        ϵ::T
    ) where {T<:AbstractFloat}

    f = ReLU(ϵ)
    λ, V, Y = _spfrEVD(X, f)
    pullback = _spfr_pullback(V, λ, f; f′ = derivative(f))

    function pullback_relur(Ȳ)

        _, ΔX = pullback(Ȳ)
        return NoTangent(), ΔX, NoTangent()
    end

    return Y, pullback_relur
end


#=
logr back-prop of f(λ) = log(λ). Derivative: f′(λ) = 1/λ 
=#
function ChainRulesCore.rrule(::typeof(logr), 
        X::AbstractMatrix{T}
    ) where {T<:AbstractFloat}

    f = Log()
    λ, V, Y = _spfrEVD(X, f)
    pullback = _spfr_pullback(V, λ, f; f′=derivative(f))

    return Y, pullback
end


#=
logr_relu back-prop of f(λ) = log(max(λ,ϵ)). Derivative: f′(λ) = {1/x if x >ϵ; 0 if x≤ϵ}​
=#
function ChainRulesCore.rrule(::typeof(logr_relu), 
        X::AbstractMatrix{T}, ϵ::T
    ) where {T<:AbstractFloat}

    f = LogReLU(ϵ)
    λ, V, Y = _spfrEVD(X, f)
    pullback = _spfr_pullback(V, λ, f; f′ = derivative(f))

    function pullback_relur(Ȳ)

        _, ΔX = pullback(Ȳ)
        return NoTangent(), ΔX, NoTangent()
    end

    return Y, pullback_relur
end


#= 
expr back-prop of f(λ) = exp(λ). Derivative: f′(λ) = exp(λ)
=#
function ChainRulesCore.rrule(::typeof(expr), 
        X::AbstractMatrix{T}
    ) where {T<:AbstractFloat}
    
    f = Exp()
    λ, V, Y = _spfrEVD(X, f)
    pullback = _spfr_pullback(V, λ, f; f′=derivative(f))

    return Y, pullback
end


#= 
sqrtr back-prop of f(λ) = λ^(1/2). Derivative: f′(λ) = 1 / (2√x). 
=#
function ChainRulesCore.rrule(::typeof(sqrtr), 
        X::AbstractMatrix{T}
    ) where {T<:AbstractFloat}
    
    f = Sqrt()
    λ, V, Y = _spfrEVD(X, f)
    pullback = _spfr_pullback(V, λ, f; f′=derivative(f))

    return Y, pullback
end


#= 
invsqrtr back-prop of f(λ) = λ^(-1/2). Derivative: f′(λ) = −1 / (2x^(3/2)) = −1 / (x * √x)
=#
function ChainRulesCore.rrule(::typeof(invsqrtr), 
        X::AbstractMatrix{T}
    ) where {T<:AbstractFloat}
    
    f = Invsqrt()
    λ, V, Y = _spfrEVD(X, f)
    pullback = _spfr_pullback(V, λ, f; f′=derivative(f))

    return Y, pullback
end


#=
powerr back-prop of f(λ) = λ^(p). Derivative: f′(λ) = p λ^(p−1)
no gradient for p
NB: one(p) allows p to be an integer as well
=#
function ChainRulesCore.rrule(::typeof(powerr), 
        X::AbstractMatrix{T}, p::T
    ) where {T<:AbstractFloat}

    f = Power(p)
    λ, V, Y = _spfrEVD(X, f)

    function pullback(Ȳ)
        ΔX = _spfr_frechet(V, λ, Ȳ, f; f′=derivative(f))
        return NoTangent(), ΔX, NoTangent()
    end

    return Y, pullback
end

###################################################################
# SPECIAL RULES (these are necessary as they do not 
# fall back to the rule for spfr)
###################################################################

#=
powerr_t back-prop of f(λ) = λ^(p). Derivative: f′(λ) = p λ^(p−1) 
with gradient for p
This cannot use _spfr_pullback as the pullback returns more arguments.
=#
function ChainRulesCore.rrule(::typeof(powerr_t), 
        X::AbstractMatrix{T}, p::T
    ) where {T<:AbstractFloat}
    
    f = Power(p)
    λ, V, Y = _spfrEVD(X, f)
    
    function pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        #G_sym = (Ȳ + Ȳ') ./ T(2)
        G = sym(V' * sym(Ȳ) * V)
        K = loewner(λ, f; f′=derivative(f))
        M = K .* G

        # Fréchet derivative (returned already symmetric; the forward pass
        # symmetrizes plain Matrix inputs, so no doubling/zeroing is needed)
        ΔX = sym(V * M * V')

        # derivative with respect to exponent p
        
        # old version
        # dY_dp = V * Diagonal(λ .^ p .* log.(λ)) * V'
        # Δp = T(sum(G_sym .* dY_dp))

        # equivalent and faster version, since tr(G'VDV') = tr(V'G'VD) and G is symmetric
        Δp = dot(diag(G), λ.^p .* log.(λ))

        return (NoTangent(), ΔX, Δp::T)
    end

    return Y, pullback
end

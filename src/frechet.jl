
"""
```julia
frechet(V, K, E)
```

Fréchet derivative defined here above, given eigenvectors in the columns of `U`, 
Löwner matrix `K` and perturbation `E` (``\\vec{P}`` in the formula above).
"""
frechet(U, K, E) = sym(U * (K .* (U' * sym(E) * U)) * U')


#=
    _spfr_frechet(X, V, λ, Ȳ, f; f′=nothing)

Compute the Fréchet derivative of a spectral function.

This is the core linear-algebra routine used by all pullbacks for
EVD-based spectral functions.

Arguments
---------
X  : primal SPD matrix
V  : eigenvectors of X
λ  : eigenvalues of X
Ȳ  : upstream cotangent
f  : scalar spectral function
f′ : optional derivative of f (if not provided, it will be computed by Zygote)

Returns
-------
ΔX : cotangent with respect to X.
=#
function _spfr_frechet(
    X::AbstractMatrix{T},
    V::AbstractMatrix{T},
    λ::AbstractVector{T},
    Ȳ,
    f::Union{Function, Nothing};
    f′::Union{Function, Nothing}=nothing
) where {T<:AbstractFloat}

    K = loewner(λ, f; f′)
    ΔX = frechet(V, K, ChainRulesCore.unthunk(Ȳ))
    desym!(X, ΔX)

    return ΔX
end

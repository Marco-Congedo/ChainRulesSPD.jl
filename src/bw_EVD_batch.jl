# ========================================================================================================
#   Custom rules to be applied in back-propagation for the batched SPD spectral functions defined in
#   `fw_EVD_batch.jl`.
#
#   The scalar (matrix) rules are defined in `fw_EVD.jl`.
#
#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================


# ------------------------------------------------------------------------------------------
# Generic batched pullback helper
# ------------------------------------------------------------------------------------------

#=
    _spfr_pullback_batch(X, Vs, Λs, f; f′=nothing)

Construct the pullback for batched EVD-based spectral functions.
=#
function _spfr_pullback_batch(
        X::AbstractArray{T,3}, 
        Vs::Vector{Matrix{T}}, 
        Λs::Vector{Vector{T}}, 
        f::Function; 

        f′::Union{Nothing,Function} = nothing
    ) where {T<:AbstractFloat}

    function pullback(Ȳ)
        Ȳ = ChainRulesCore.unthunk(Ȳ)
        ΔX = similar(X)

        @inbounds for b in axes(X,3)
            Δ = _spfr_frechet(Vs[b], Λs[b], @view(Ȳ[:,:,b]), f; f′=f′)
            @views ΔX[:,:,b] .= Δ
        end

        return NoTangent(), ΔX, NoTangent()
    end

    return pullback
end

# ------------------------------------------------------------------------------------------
# Generic batched spectral primitive
# ------------------------------------------------------------------------------------------
function ChainRulesCore.rrule(::typeof(spfr), 
        X::AbstractArray{T,3}, 
        f::Function; 

        f′::Union{Nothing,Function} = nothing    
    ) where {T<:AbstractFloat}

    Vs, Λs, Y = _spfrEVD_batch(X,f)
    pullback = _spfr_pullback_batch(X, Vs, Λs, f; f′=f′)

    return Y, pullback
end


function _spfr_batch_wrapper_rrule(
        X::AbstractArray{T,3}, 
        f::Function; 

        f′::Union{Nothing,Function} = nothing
    ) where {T<:AbstractFloat}

    Y, pb = ChainRulesCore.rrule(spfr, X, f; f′ = f′)

    function pullback(Ȳ)
        _, ΔX, _ = pb(Ȳ)
        return NoTangent(), ΔX
    end

    return Y, pullback
end



#=
relur back-prop of λ <- max(λ, ϵ).
No gradient with respect to ϵ.
=#
function ChainRulesCore.rrule(::typeof(relur), 
        X::AbstractArray{T,3}, 
        ϵ::T
    ) where {T<:AbstractFloat}

    Y, pb = ChainRulesCore.rrule(spfr, X, ReLU(ϵ); f′ = ReLU′(ϵ))

    function pullback(Ȳ)
        _, ΔX, _ = pb(Ȳ)
        return NoTangent(), ΔX, NoTangent()
    end

    return Y, pullback
end


ChainRulesCore.rrule(::typeof(logr), X::AbstractArray{T,3}) where {T<:AbstractFloat} =
    _spfr_batch_wrapper_rrule(X, Log(); f′ = Log′())


#=
logr_relu back-prop of f(λ) = log(max(λ,ϵ)).
No gradient with respect to ϵ.
=#
function ChainRulesCore.rrule(::typeof(logr_relu), 
        X::AbstractArray{T,3}, 
        ϵ::T
    ) where {T<:AbstractFloat}

    Y, pb = ChainRulesCore.rrule(spfr, X, LogReLU(ϵ); f′ = LogReLU′(ϵ))

    function pullback(Ȳ)
        _, ΔX, _ = pb(Ȳ)
        return NoTangent(), ΔX, NoTangent()
    end

    return Y, pullback
end


ChainRulesCore.rrule(::typeof(expr), 
        X::AbstractArray{T,3}
    ) where {T<:AbstractFloat} = _spfr_batch_wrapper_rrule(X, Exp(); f′ = Exp′())

    
ChainRulesCore.rrule(::typeof(sqrtr), 
        X::AbstractArray{T,3}
    ) where {T<:AbstractFloat} = _spfr_batch_wrapper_rrule( X, Sqrt(); f′ = Sqrt′())


ChainRulesCore.rrule(::typeof(invsqrtr), 
        X::AbstractArray{T,3}
    ) where {T<:AbstractFloat} = _spfr_batch_wrapper_rrule(X, Invsqrt(); f′ = Invsqrt′())


function ChainRulesCore.rrule(::typeof(powerr), 
        X::AbstractArray{T,3}, 
        t::T
    ) where {T<:AbstractFloat} 

    Y, pb = ChainRulesCore.rrule(spfr, X, Power(t); f′ = Power′(t))

    function pullback(Ȳ)
        _, ΔX, _ = pb(Ȳ)
        return NoTangent(), ΔX, NoTangent()
    end

    return Y, pullback
end

# ------------------------------------------------------------------------------------------
# Batched power with trainable exponent
# ------------------------------------------------------------------------------------------

function ChainRulesCore.rrule(::typeof(powerr_t), 
        X::AbstractArray{T,3}, 
        t::T
    ) where {T<:AbstractFloat}

    f  = Power(t)
    f′ = Power′(t)

    Vs, Λs, Y = _spfrEVD_batch(X,f)

    function pullback(Ȳ)

        Ȳ = ChainRulesCore.unthunk(Ȳ)

        ΔX = similar(X)
        Δt = zero(T)

        @inbounds for b in axes(X,3)

            Ȳb  = @view Ȳ[:,:,b]

            λ = Λs[b]
            V = Vs[b]

            @views ΔX[:,:,b] .= _spfr_frechet(V, λ, Ȳb, f; f′=f′)

            #G = V' * Symmetric((Ȳb + Ȳb') / T(2)) * V
            #G = V' * sym(Ȳb) * V # to do : compute quadratic forms only
            #Δt += dot(diag(G), (λ .^ t) .* log.(λ))

            temp = sym(Ȳb)
            g = [dot(v, temp, v) for v in eachcol(V)]
            Δt += dot(g, (λ .^ t) .* log.(λ))
        end

        return NoTangent(), ΔX, Δt
    end

    return Y, pullback
end
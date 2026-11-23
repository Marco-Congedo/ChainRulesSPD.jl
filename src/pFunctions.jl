#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) 2026
# ========================================================================================================

#########################################
# Parametric and Non-Parametric Functions
#########################################

# ReLU

struct ReLU{T<:Real} <: Function
    ϵ::T
end

(f::ReLU)(x::Real) = max(x, f.ϵ)

struct ReLU′{T<:Real} <: Function
    ϵ::T
end
(f::ReLU′)(x::Real) = x > f.ϵ ? one(x) : zero(x)

derivative(f::ReLU) = ReLU′(f.ϵ)

# 

# Natural log
struct Log <: Function end
(f::Log)(x::Real) = log(x)

struct Log′ <: Function end
(f::Log′)(x::Real) = inv(x)

derivative(f::Log) = Log′()

#


# Natural log + ReLU
struct Log_ReLU{T<:Real} <: Function
    ϵ::T
end
(f::Log_ReLU)(x::Real) = log(max(x, f.ϵ)) # no need of log1p as ir is > ϵ

struct Log_ReLU′{T<:Real} <: Function
    ϵ::T
end
(f::Log_ReLU′)(x::Real) = x > f.ϵ ? inv(x) : zero(x)

derivative(f::Log_ReLU) = Log_ReLU′(f.ϵ)

# 

# Exponential
struct Exp <: Function end
(f::Exp)(x::Real) = exp(x)

struct Exp′ <: Function end
(f::Exp′)(x::Real) = exp(x)

derivative(f::Exp) = Exp′()

#

# Square Root
struct Sqrt <: Function end
(f::Sqrt)(x::Real) = sqrt(x)

struct Sqrt′ <: Function end
(f::Sqrt′)(x::Real) = inv(2*sqrt(x))

derivative(f::Sqrt) = Sqrt′()
#

# Inverse Square Root

struct Invsqrt <: Function end
(f::Invsqrt)(x::Real) = inv(sqrt(x))

struct Invsqrt′ <: Function end
(f::Invsqrt′)(x::Real) = -inv(2*x*sqrt(x))

derivative(f::Invsqrt) = Invsqrt′()

#

# Power
struct Power{T<:Real} <: Function
    p::T
end
(f::Power)(x::Real) = x^f.p

struct Power′{T<:Real} <: Function
    p::T
end
(f::Power′)(x::Real) = f.p * x^(f.p - one(f.p))

derivative(f::Power) = Power′(f.p)

# 

# softplus, numerically stable
function softplus(x::T) where T<:Real
    log1p(exp(-abs(x))) + max(x, zero(T))
end

# its derivative is the sigmoid, numerically stable
function softplus′(x::T) where T<:Real
    if x ≥ zero(T)
        z = exp(-x)
        return one(T)/(one(T)+z)
    else
        z = exp(x)
        return z/(one(T)+z)
    end
end

struct Softplus <: Function end
(f::Softplus)(x::Real) = softplus(x)

struct Softplus′ <: Function end
(f::Softplus′)(x::Real) = softplus′(x)

derivative(f::Softplus) = Softplus′()

# 

# softexp function and its derivative
function softexp(x::T; ϵ::T = T(1e-4), c::T = T(6.0)) where T<:Real
    Lk = softplus(c)
    return ϵ + (one(T) - ϵ) * (softplus(c * x) / Lk)
end

function softexp′(x::T; ϵ::T = T(1e-4), c::T = T(6.0)) where T<:Real
    Lk = softplus(c)
    σ  = softplus′(c * x)      # = sigmoid(c*x), stable; replaces NNlib.sigmoid
    return (one(T) - ϵ) * (c * σ) / Lk
end

struct Softexp{T<:Real} <: Function
    ϵ::T
    c::T # curvature
end
(f::Softexp)(x::T)  where T<:Real = softexp(x; ϵ = T(f.ϵ), c = T(f.c))

struct Softexp′{T<:Real} <: Function
    ϵ::T
    c::T
end
(f::Softexp′)(x::T) where T<:Real = softexp′(x; ϵ = T(f.ϵ), c = T(f.c))

derivative(f::Softexp) = Softexp′(f.ϵ, f.c)

###################################################

# fall-back using ForwadDiff
derivative(f::Function) = x -> ForwardDiff.derivative(f, x)

# given a function f and f′ either a function (which in this case must be the exact derivative) of nothing, 
# return the right derivative f′
_derivative(f, f′::Function) = f′
_derivative(f, ::Nothing)    = derivative(f)
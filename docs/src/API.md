## Tools

```@docs
sym
skew
```

## Parametric and non-parametric functors

The package defines several **functors** (e.g., `Invsqrt()`).

For all of them, `derivative(f)` returns the corresponding derivative functor. If $f$ is not among the functor listed here below, `derivative(f::Function)` falls back to `ForwardDiff.derivative(f, x)`.

| function | ``y=f(x)`` | ``y'=f'(x)`` | functor | derivative |
|----------|------------|-------------|---------|------------|
| ReLU | ``\max(x, \epsilon)`` | ``\begin{cases} 1 & x > \epsilon \\ 0 & x \leq \epsilon \end{cases}`` | ReLU(ϵ) | ReLU′(ϵ) |
| log | ``\ln(x)`` | ``\frac{1}{x}`` | Log() | Log′() |
| log-ReLU | ``\ln(\max(x, \epsilon))`` | ``\begin{cases} \frac{1}{x} & x > \epsilon \\ 0 & x \leq \epsilon \end{cases}`` | Log_ReLU(ϵ) | Log_ReLU′(ϵ) |
| exp | ``e^x`` | ``e^x`` | Exp() | Exp′() |
| sqrt | ``\sqrt{x}`` | ``\frac{1}{2\sqrt{x}}`` | Sqrt() | Sqrt′() |
| 1/sqrt | ``\frac{1}{\sqrt{x}} = x^{-1/2}`` | ``-\frac{1}{2x^{3/2}}`` | Invsqrt() | Invsqrt′() |
| x^t | ``x^t`` | ``t\,x^{t-1}`` | Power(t) | Power′(t) |
| softplus | ``\ln(1+e^x)`` | ``\frac{1}{1+e^{-x}}`` | Softplus() | Softplus′() |
| softexp | ``\epsilon + (1-\epsilon)\frac{\ln(1+e^{kx})}{\ln(1+e^k)}`` | ``\frac{(1-\epsilon)k}{\ln(1+e^k)}\frac{1}{1+e^{-kx}}`` | Softexp(ϵ, c) | Softexp′(ϵ, c) |

!!! details "Examples"
    ```repl
    using ChainRulesSPD

    julia> f = Sqrt()
    (::Sqrt) (generic function with 1 method)

    julia> f(4)
    2.0

    julia> df_ = Sqrt′()
    (::Sqrt′) (generic function with 1 method)

    julia> df_(4)
    0.25

    julia> df = derivative(f)
    (::Sqrt′) (generic function with 1 method)

    julia> df(4)
    0.25

    julia> g = Softexp(eps(), 6.0)
    (::Softexp{Float64}) (generic function with 1 method)

    julia> g(4.0)
    3.9983502239688047
    ```

## Löwner matrix and Fréchét derivative

!!! details "Maths"
    For a matrix ``P\in \mathcal{S}_{++}``, we denote a **spectral function** by ``f(P) = Uf(\Lambda)U^T``, where ``(U, \Lambda)`` is the eigenpair of ``P`` and ``f`` is a function defined on the spectrum of ``P``. The **Fréchet derivative** ``L_f(P)[\vec{P}]`` is the first-order change in ``f(P)`` caused by a small perturbation ``\vec{P}`` to ``P``.

    The classical Daletskii–Krein formula yields

    `` L_f(P)[\vec{P}] = U\bigl(K_{f,\lambda}\circ(U^{T}\vec{P}U)\bigr)U^{T},``

    where ``\circ`` is the Hadamard product and ``K_{f,\lambda}`` is the **Löwner matrix** (finite differences) with elements 

    ``(K_{f,\lambda})_{ij}=\left \{ \begin{array}{rl}\displaystyle\frac{f(\lambda_i)-f(\lambda_j)}{\lambda_i-\lambda_j}, & \lambda_i\neq \lambda_j\\[0.5ex]f'(\lambda_i), & \lambda_i=\lambda_j\end{array} \right.``.

The Löwner matrix is used in all backpropagation rules. The Fréchet derivative is used for the pullback in all EVD-based functions. For GEVD-based functions, see the pullback functions in the paper xxx.


```@docs
loewner
frechet
```

## EVD-based methods

Given eigendecomposition ``P=U \Lambda U^T``

| operation       | expression                                   | method               | learnable | hyperparameter |
|-----------------|----------------------------------------------|----------------------|-----------|-----------------|
| **Generic**     | ``f(P)``                                     |[`spfr`](@ref)        | depending on ``f``|depending on ``f``|
| ReLu Activation | ``U \mathrm{max} (\epsilon I, \Lambda) U^T`` |[`relur`](@ref)       | nothing   | ``\epsilon``|
| Logarithm       | ``U \mathrm{Log} (\Lambda) U^T``             |[`logr`](@ref)        | nothing   | nothing   |
| log-ReLU        | ``U \mathrm{Log} (\mathrm{max}(\epsilon I, \Lambda)) U^T``|[`logr_relu`](@ref)   | nothing   | ``\epsilon``|
| Exponential     | ``U \mathrm{Exp} (\Lambda) U^T``             |[`expr`](@ref)        | nothing   | nothing   |
| Square Root     | ``U \Lambda^{1/2} U^T``                      |[`sqrtr`](@ref)       | nothing   | nothing   |
| Inv. Square Root| ``U \Lambda^{-1/2} U^T``                     |[`invsqrtr`](@ref)    | nothing   | nothing   |
| Power           | ``U \Lambda^{t} U^T``                        |[`powerr`](@ref)      | nothing   | ``t``   |
| Power           | ``U \Lambda^{t} U^T``                        |[`powerr_t`](@ref)    | ``t``     | nothing   |

```@docs
spfr
```

### Wrappers for EVD-based methods

```@docs
relur
logr
logr_relu
expr
sqrtr
invsqrtr
powerr
powerr_t
```

## GEVD-based methods

We give here the simplified expressions based on the generalized eigendecomposition (GEVD).
These functions will compute the GEVD of ``P^{-1}Q`` such as ``P^{-1}Q=B \Lambda A``, where ``B^TPB=I``, ``B^TQB=\Lambda``, ``A=B^{-1}``.

| operation        | expression                             | method               | learnable | hyperparameter |
|------------------|----------------------------------------|----------------------|-----------|-----------|
| **Generic**      | ``\varphi(f, P, Q)=A^T f(\Lambda) A``  |[`phir`](@ref)        | depending on ``f``| depending on ``f`` |
| geodesic         | ``A^T \Lambda^t A``                    |[`geodesicr`](@ref)   | nothing   | ``t`` |
| geodesic         | ``A^T \Lambda^t A``                   |[`geodesicr_t`](@ref)  | ``t``     | nothing   |
| geometric mean   | ``A^T \Lambda^{1/2} A``               |[`geomeanr`](@ref)     | nothing   | nothing   |
| Logarithmic map  | ``A^T \mathrm{Log}(\Lambda) A``        |[`logmapr`](@ref)     | nothing   | nothing   |
| Exponential map† | ``A^T \mathrm{Exp}(\Lambda) A``        |[`expmapr`](@ref)     | nothing   | nothing   |

| operation           | expression                                | method               | learnable | hyperparameter |
|---------------------|-------------------------------------------|----------------------|-----------|-----------|
| **Generic**         | ``\psi(f, P, Q)=B f(\Lambda) A``          |[`psir`](@ref)        | depending on ``f`` | depending on ``f`` |
| parallel transport  | ``T^T S \; T``, ``\:`` ``T=B \Lambda^{1/2} A``      |[`parallel_transportr`](@ref)   | nothing   | nothing   |
| parallel transport‡ | ``T^T \zeta \; T``, ``\:`` ``T=B \Lambda^{1/2} A`` |[`parallel_transportr`](@ref)   | nothing    | nothing   |

† For the exponential map of a tangent vector ``\xi \in \mathcal{S}``, the GEVD is ``P^{-1} \xi =B \Lambda A``, ``B^TPB=I``, ``B^T \xi B=\Lambda``, ``A=B^{-1}``.

‡ The quantity computed by [`parallel_transportr`](@ref) is the same for a matrix ``S \in \mathcal{S}_{++}`` and 
a tangent vector ``\zeta \in \mathcal{S}``.

```@docs
phir
geodesicr
geodesicr_t
geomeanr
logmapr
expmapr
psir
parallel_transportr
```


# ChainRulesSPD

!!! note "Notation"
    In the following, ``\mathcal{S}_{++}`` denotes the set of symmetric positive definite matrices and
    ``\mathcal{S}`` the set of symmetric matrices.

Two kinds of primal functions are implemented:
- functions based on the *eigendecomposition* (EVD) of each matrix input ``P \in \mathcal{S}_{++}``, such as ``\mathrm{Log}(P)``, ``P^t``, etc.
- functions based on the *generalized eigendecomposition* (GEVD) of an input matrix pair ``P \in \mathcal{S}_{++}``, ``X \in \mathcal{S}``, such as the geometric mean, the geodesic, the logarithmic map, etc.

For all details, see the accompanying [paper](@ref reference).

!!! tip
    All functions are **non-mutating**, thus they can be used in [Zygote.jl](https://github.com/fluxml/zygote.jl).

## EVD-based functions

All EVD-based functions are instances of 

`` f(P) ``, 

where ``P \in \mathcal{S}_{++}`` and ``f`` is a function defined on the spectrum of ``P``. The generic method is [`spfr`](@ref) (short for *spectral function*). 

Wrappers are available for the most common functions:

| Functions | Expression | Method | Output |
|-----------|------------|--------|--------|
| Logarithm | ``\mathrm{Log}(P)`` | [`logr`](@ref) | ``\mathcal{S}_{++}``|
| Exponential | ``\mathrm{Exp}(P)`` | [`expr`](@ref) |``\mathcal{S}_{++}``|
| Square root | ``P^{1/2}`` | [`sqrtr`](@ref) |``\mathcal{S}_{++}``|
| Inverse of the square root | ``P^{-1/2}`` | [`invsqrtr`](@ref) |``\mathcal{S}_{++}``|
| Power | ``P^{t}`` | [`powerr`](@ref) |``\mathcal{S}_{++}``|

## GEVD-based functions

Given a matrix ``P \in \mathcal{S}_{++}``, ``X`` either `` \in \mathcal{S}_{++} `` or a tangent vector ``\in \mathcal{S}``, and ``f`` a function defined on the spectrum of ``P^{-1}X``, all GEVD-based functions are instances of either
- ``\varphi(f,P,X)=Pf(P^{-1}X)=P^{1/2}f(P^{-1/2}XP^{-1/2})P^{1/2}.``
The generic method for those is [`phir`](@ref).
- ``\psi(f,P,X)=f(P^{-1}Q).``
The generic method for those is [`psir`](@ref).

Wrappers are available for the most common functions:

| Functions | Notation | Expression | Method | Output |
|-----------|------------|--------|--------|--------|
| Geodesic with ``t`` real | ``P\#_tQ`` |  ``P^{1/2} (P^{-1/2} Q P^{-1/2})^t P^{1/2}``| [`geodesicr`](@ref)|  ``\mathcal{S}_{++}``|
| Geometric mean (geodesic with ``t=\frac{1}{2}``) | ``P\#Q`` |  ``P^{1/2} (P^{-1/2} Q P^{-1/2})^\frac{1}{2} P^{1/2}``| [`geomeanr`](@ref)| ``\mathcal{S}_{++}``|
| Logarithmic map | ``\mathrm{Log}_P(Q)`` |    ``P^{1/2} \mathrm{Log}(P^{-1/2} Q P^{-1/2}) P^{1/2}``| [`logmapr`](@ref)|  ``\mathcal{S}``|
| Exponential map | ``\mathrm{Exp}_P(\xi)`` |  ``P^{1/2} \mathrm{Exp}(P^{-1/2} \xi P^{-1/2}) P^{1/2}``| [`expmapr`](@ref)| ``\mathcal{S}_{++}``|
| Parallel Transport of ``S \in \mathcal{S}_{++}`` | ``\mathcal{P}_{P \to Q}(S)`` | ``(P^{-1}Q)^\frac{T}{2} S (P^{-1}Q)^\frac{1}{2}``| [`parallel_transportr`](@ref)| ``\mathcal{S}_{++}``|
| Parallel Transport of tang. vector ``\zeta \in \mathcal{S}`` | ``\mathcal{P}_{P \to Q}(\zeta)`` | ``(P^{-1}Q)^\frac{T}{2} \zeta (P^{-1}Q)^\frac{1}{2}``| [`parallel_transportr`](@ref)| ``\mathcal{S}``|


## Input
All methods support Julia's standard type `Matrix`, `Symmetric` and `Hermitian`,
and also tensors (batches) of matrices with shape ``n \times n \times m``, where ``n`` is the order of input matrices
and ``m`` is the number of slices in the tensors.

## Pre-Conditioning

For all GEVD-based methods, a keyword argument `precond` is available. This allows to reduce the 
condition number of matrix ``P``. For details, see the last paragraph of Appendix E.2 of the accompanying [paper](@ref reference).

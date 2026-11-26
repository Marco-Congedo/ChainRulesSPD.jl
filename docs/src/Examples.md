# Examples

In order to use the [primal functions](@ref "ChainRulesSPD") in a neural network, just use them in your layers --- see
the API for examples.

Here below it is shown how to get the gradient of primal functions with respect to a dummy loss function using [Zygote.jl](https://github.com/fluxml/zygote.jl).
Furthermore, the gradient is tested against the analytical expression computed as detailed in the accompanying [paper](@ref reference).

The EVD-based primal functions are all instances of [`spfr`](@ref), while the GEVD-based primal functions
are all instances of [`phir`](@ref) or [`psir`](@ref). Here below, we give one example for each.

!!! details "spfr example: logarithm of a SPD matrix"
    ```julia
    using PosDefManifold
    using ChainRulesSPD
    using LinearAlgebra
    using Zygote
    using Test

    P = randP(3) # symmetric positive-definite input

    loss(P) = sum(abs2, logr(P))
    gradP = Zygote.gradient(loss, P)[1]

    # We will check the gradient regardless the Julia type of input matrix
    # so we compute it also for a Symmetric type input
    gradP_Sym = Zygote.gradient(loss, Symmetric(P))[1]

    # ---------------------------------------------------------------------------
    # Analytic-gradient check against the analytical gradient given in Eq. (33)
    # of the paper.
    #
    # Daletskii–Krein: for a spectral function f(X) = U f(D) Uᵀ, the pullback of a
    # scalar loss ℓ is
    #
    #     ΔP = U (K_{f,D} ⊙ (Uᵀ Ȳ U)) Uᵀ  =  frechet(U, K, Ȳ)              (33)
    #
    # where ⊙ is the Hadamard product, Ȳ is the upstream cotangent and K is the
    # Löwner matrix of Eq. (34).  Here f = log, f′(x) = 1/x, and Ȳ = 2·Log(P) is
    # the derivative of ℓ = sum(abs2, ·).
    # ---------------------------------------------------------------------------
    Y    = logr(P)                        # primal:  Y = Log(P)
    Ȳ    = 2 .* Y                         # upstream cotangent of ℓ
    λ, U = eigen(Symmetric(P))            # eigenpair (D, U) of P
    K    = loewner(λ, Log(); f′ = Log′()) # Löwner matrix of f(x) = log(x)
    G    = frechet(U, K, Ȳ)               # analytic gradient, Eq. (33)

    # frechet returns the symmetric Fréchet derivative, which is exactly what the
    # rrule of logr returns for plain-Matrix and Symmetric inputs alike.
    @test gradP ≈ gradP_Sym ≈ G           # Zygote gradient vs. analytic Eq. (33)
    ```

!!! details "phir example: geometric mean of two SPD matrices"
    ```julia
    using PosDefManifold
    using ChainRulesSPD
    using LinearAlgebra
    using Zygote
    using Test

    P, Q = randP(3), randP(3)    # symmetric positive-definite input

    loss(P, Q) = sum(abs2, geomeanr(P, Q))
    gP, gQ = Zygote.gradient(loss, P, Q)

    # We will check the gradient regardless the Julia type of input matrix
    # so we compute it also for a Symmetric type input
    gP_Sym, gQ_Sym = Zygote.gradient(loss, Symmetric(P), Symmetric(Q))

    # ---------------------------------------------------------------------------
    # Analytic-gradient check against Proposition 2 of the paper.
    #
    # Verbatim from Prop. 2 / the comment in bw_GEVD.jl: for the φ function
    # φ(f,P,Q) = A' f(Λ) A, with the GEVD
    #     P = A'A,  Q = A' Λ A,  B = A⁻¹,
    # and J = sym(A Ȳ A'), the pullback of a scalar loss ℓ is
    #     ΔQ = B (K ⊙ J) B'
    #     ΔP = B sym(J f(Λ) − (K ⊙ J) Λ) B'
    # where ⊙ is the Hadamard product and K is the Löwner matrix of Eq. (34).
    # For the geometric mean f = sqrt, and Ȳ = 2·φ(P,Q) is the derivative of
    # ℓ = sum(abs2, ·).
    # ---------------------------------------------------------------------------
    Y    = geomeanr(P, Q)                   # primal: geometric mean
    Ȳ    = 2 .* Y                           # upstream cotangent of ℓ
    λ, A, B = gevdr_phir(P, Q)              # GEVD of the pair (P, Q)
    f      = Sqrt()
    fλ     = f.(λ)
    K      = loewner(λ, f; f′ = Sqrt′())    # Löwner matrix of f(x) = √x (Eq. 34)

    J  = sym(A * Ȳ * A')
    H  = K .* J
    GP = sym(B * sym(J .* fλ' - H .* λ') * B')  # analytic ∂ℓ/∂P (Prop. 2)
    GQ = sym(B * H * B')                        # analytic ∂ℓ/∂Q (Prop. 2)

    @test gP ≈ gP_Sym ≈ GP
    @test gQ ≈ gQ_Sym ≈ GQ
    ```

!!! details "psir example: parallel transport of a SPD matrices"
    ```julia
    using PosDefManifold
    using ChainRulesSPD
    using LinearAlgebra
    using Zygote
    using Test

    P, Q, S = randP(3), randP(3), randP(3) # S: transported point from P to Q

    loss(P, Q, S) = sum(abs2, parallel_transportr(P, Q, S))
    gP, gQ, gS = Zygote.gradient(loss, P, Q, S)

    # We will check the gradient regardless the Julia type of input matrix
    # so we compute it also for a Symmetric type input
    gP_Sym, gQ_Sym, gS_Sym = Zygote.gradient(loss, Symmetric(P), Symmetric(Q), Symmetric(S))

    # ---------------------------------------------------------------------------
    # Analytic-gradient check against Proposition 3 of the paper.
    #
    # Parallel transport is  Y = E' S E,  with  E = ψ(f, P, Q) = B f(Λ) A.
    # Propagating the upstream cotangent Ȳ = 2Y (derivative of ℓ = sum(abs2, ·))
    # through Y = E' S E (chain rule) gives
    #     ΔS = sym(E Ȳ E')                       and    Ē = S E (Ȳ + Ȳ'),
    # and Proposition 3 gives the gradients of the loss w.r.t. the inputs P, Q
    # of the E (ψ) matrix:
    #     J  = A Ē' B,            H = K ⊙ sym(J)
    #     ΔQ = B H B'                                        (Prop. 3, eq. 29)
    #     ΔP = B sym( skew(J) f(Λ) − (K ⊙ sym(J)) Λ ) B'     (Prop. 3, eq. 28)
    # ---------------------------------------------------------------------------
    Y    = parallel_transportr(P, Q, S)   # primal
    Ȳ    = 2 .* Y                         # upstream cotangent of ℓ

    λ, A, B = gevdr_psir(P, Q)            # GEVD for ψ: B, A = B⁻¹
    f      = Sqrt()
    fλ     = f.(λ)
    K      = loewner(λ, f; f′ = Sqrt′())  # Löwner matrix of f(x) = √x (Eq. 34)
    E      = B * Diagonal(fλ) * A         # ψ(f, P, Q)

    ΔS = sym(E * Ȳ * E')                                  # ∂ℓ/∂S
    Ē  = S * E * (Ȳ + Ȳ')                                 # ∂ℓ/∂E
    J  = A * Ē' * B
    H  = K .* sym(J)
    ΔQ = sym(B * H * B')                                  # Prop. 3, eq. (29)
    ΔP = sym(B * sym(skew(J) .* fλ' - H .* λ') * B')      # Prop. 3, eq. (28)

    @test gP ≈ gP_Sym ≈ ΔP
    @test gQ ≈ gQ_Sym ≈ ΔQ
    @test gS ≈ gS_Sym ≈ ΔS
    ```

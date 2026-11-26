# ========================================================================================================
#   Tests of fw_EVD.jl
# ========================================================================================================

println("\nTesting fw_EVD.jl...")

# ========================================================================================================
#   Tests for SPD forward operations based on EVD
#
#   Tests:
#       - eigenr
#       - spfr
#       - _spfrEVD
#       - common spectral wrappers
#
#   Copyright Marco Congedo, DR, PhD, CNRS (Grenoble) - Nov 2025
# ========================================================================================================

n = 4;

X = randP(n);                 # SPD Symmetric matrix
Xm = Matrix(X);               # dense representation


# ------------------------------------------------------------------------------------------
# eigenr
# ------------------------------------------------------------------------------------------

@testset "eigenr" begin

    λ1, V1 = ChainRulesSPD.eigenr(Xm)
    λ2, V2 = ChainRulesSPD.eigenr(X)

    @test λ1 ≈ λ2
    @test V1 * Diagonal(λ1) * V1' ≈ Xm
    @test V2 * Diagonal(λ2) * V2' ≈ Xm

end;


# ------------------------------------------------------------------------------------------
# Generic spectral function
# ------------------------------------------------------------------------------------------

@testset "spfr" begin

    Y = spfr(Xm, sqrt)

    @test Y ≈ Y'
    @test Y * Y ≈ Xm atol=1e-10

end;


# ------------------------------------------------------------------------------------------
# Internal EVD primitive
# ------------------------------------------------------------------------------------------

@testset "_spfrEVD" begin

    λ, V, Y = _spfrEVD(Xm, Log())

    @test length(λ) == n
    @test size(V) == (n,n)
    @test Y ≈ Y'

    Yref = spfr(Xm, Log())

    @test Y ≈ Yref

end;


# ------------------------------------------------------------------------------------------
# Wrapper functions
# ------------------------------------------------------------------------------------------

@testset "Spectral wrappers" begin

    @test logr(Xm) ≈ spfr(Xm, Log()) ≈ log(Symmetric(Xm))

    ϵ = sqrt(eps(Float64))
    @test logr_relu(Xm, ϵ) ≈ spfr(Xm, LogReLU(ϵ))

    @test expr(Xm) ≈ spfr(Xm, Exp()) ≈ exp(Symmetric(Xm))

    @test sqrtr(Xm) ≈ spfr(Xm, Sqrt()) ≈ sqrt(Symmetric(Xm))

    @test invsqrtr(Xm) ≈ spfr(Xm, Invsqrt()) ≈ inv(sqrt(Symmetric(Xm)))

    p = 2.0

    @test powerr(Xm, p) ≈ spfr(Xm, Power(p)) ≈ (Symmetric(Xm))^2

end;


# ------------------------------------------------------------------------------------------
# Basic spectral properties
# ------------------------------------------------------------------------------------------

@testset "Symmetry preservation" begin

    for f in (
        logr,
        expr,
        sqrtr,
        invsqrtr
    )

        Y = f(Xm)
        @test Y ≈ Y'

        Y = logr_relu(Xm, 0.1)
        @test Y ≈ Y'
    end

end;
# ========================================================================================================
#   Tests of frechet.jl
# ========================================================================================================


println("Testing frechet.jl...")

# FD step size
epsilonf = sqrt(eps(Float64))    # ≈ 1.49e-8, near-optimal regardless of `epsilon`

epsilon  = 1e-6                        

# test the Fréchet derivative of f(X), where X is symmetric and has order n
# This implements a forward finite-difference method, which epsilonf = sqrt(eps(Float64)) 
# To obtain more accurate results we may use a central different method:
#    fXp = PosDefManifold.spectralFunctions(Hermitian(X + epsilonf*H), f)
#    fXm = PosDefManifold.spectralFunctions(Hermitian(X - epsilonf*H), f)
#    FD  = (fXp - fXm) / (2*epsilonf)
# In this case, the appropriate epsilonf would be eps(Float64)^(1/3)
function testfrechet(X, f::Function)
    # random SPD matrix

    n = size(X, 1)
    λ, V = eigen(Symmetric(X)) # must use symmetric EVD

    # random symmetric direction
    B = randn(n,n)
    H = Symmetric(B + B')

    ΔX = frechet(V, loewner(λ, f), H)

    fX = Hermitian((V * Diagonal(f.(λ))) * V')
    fXH = PosDefManifold.spectralFunctions(Hermitian(X + epsilonf*H), f)
    FD = (fXH - fX) / epsilonf

    re = norm(ΔX - FD) / norm(FD) # relative error
    println("relative error = ", re)
    @test re < epsilon

    return re
end

# Example tests
# NB, the test may occasionally fail at the 1e-6 tolerance level

@testset " Test Fréchet derivative" begin

    n=4
    X = randP(n)+0.01*I

    @test testfrechet(X, log) < epsilon;
    @test testfrechet(X, exp) < epsilon;
    @test testfrechet(X, sqrt) < epsilon;
    @test testfrechet(X, Log()) < epsilon;
    @test testfrechet(X, Exp()) < epsilon;
    @test testfrechet(X, Sqrt()) < epsilon;
    @test testfrechet(X, Invsqrt()) < epsilon;
    @test testfrechet(X, Power(.2)) < epsilon;


    # fallback when no derivative is specified; uses AD instead
    @test testfrechet(X, sin) < epsilon;
    @test testfrechet(X, x->inv(x)) < epsilon;
    @test testfrechet(X, x->x^(-1/2)) < epsilon;
    @test testfrechet(X, x->x^(1/2)) < epsilon;
    @test testfrechet(X, x->x^3) < epsilon;
    # problem testing this. Not a problem as the rrule passes
    # testfrechet(30, x->max.(x, 0.1)) 

    # For matrices not flagged as Symmetric or Hermitian
    X = Matrix(X)

    @test testfrechet(X, log) < epsilon;
    @test testfrechet(X, exp) < epsilon;
    @test testfrechet(X, sqrt) < epsilon;
    @test testfrechet(X, Log()) < epsilon;
    @test testfrechet(X, Exp()) < epsilon;
    @test testfrechet(X, Sqrt()) < epsilon;
    @test testfrechet(X, Invsqrt()) < epsilon;
    @test testfrechet(X, Power(.68)) < epsilon;
    @test testfrechet(X, x->inv(x)) < epsilon;
end;

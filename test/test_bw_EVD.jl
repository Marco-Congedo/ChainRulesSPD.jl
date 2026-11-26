# ========================================================================================================
#   Tests of bp_rules_EVD.jl
# ========================================================================================================

# Test rrules with X either a generic Matrix, a Symmetric or an Hermitian Matrix

println("\nTesting bw_EVD.jl...")

n = 4;

# Test for Float64
X = randP(n);          # Hermitian
S = Symmetric(X);      # Symmetric wrapper
M = Matrix(X);         # plain symmetric Matrix

for W in (M, S, X)
    @testset "$(typeof(W)) rrules" begin
        ChainRulesTestUtils.test_rrule(spfr, W, sqrt; rtol, atol);

        ChainRulesTestUtils.test_rrule(relur, W, 1e-3; rtol, atol);
        ChainRulesTestUtils.test_rrule(logr, W; rtol, atol);
        ChainRulesTestUtils.test_rrule(logr_relu, W, 0.1; rtol, atol);
        ChainRulesTestUtils.test_rrule(expr, W; rtol, atol);
        ChainRulesTestUtils.test_rrule(sqrtr, W; rtol, atol);
        ChainRulesTestUtils.test_rrule(invsqrtr, W; rtol, atol);

        t = 0.1
        # t ⊢ ChainRulesCore.NoTangent() tell not to test the tangent for t, which is not returned as t is not learnable
        ChainRulesTestUtils.test_rrule(powerr, W, t ⊢ ChainRulesCore.NoTangent())
        # Here t is learnable
        ChainRulesTestUtils.test_rrule(powerr_t, W, t)
    end
end


# -------------------------------------------------------------------------------------------------------
# Integration through an AD engine (Zygote), as used inside a neural network layer.
#
# The pullbacks must return a symmetric gradient for a plain Matrix input (the forward
# pass symmetrizes `Hermitian((X+X')/2)`), and the gradient must not depend on whether
# the input was passed as Matrix, Symmetric or Hermitian.
# -------------------------------------------------------------------------------------------------------

@testset "Neural-network integration (Zygote)" begin

    W = Matrix(randP(n) + 0.1I);  # now exactly symmetric

    # (1) gradients are symmetric for plain Matrix inputs, and independent of the wrapper type
    for f in (logr, expr, sqrtr, invsqrtr)
        gM = Zygote.gradient(W -> sum(abs2, f(W)), W)[1]
        @test gM ≈ gM'
        gH = Zygote.gradient(H -> sum(abs2, f(H)), Hermitian(W))[1]
        gS = Zygote.gradient(S -> sum(abs2, f(S)), Symmetric(W))[1]
        @test gM ≈ Matrix(gH) ≈ Matrix(gS)
    end

    # (2) a nonsymmetric plain Matrix is symmetrized: same gradient as for its symmetric part
    A = randn(n, n)
    Wskew = W .+ 0.1 .* (A .- transpose(A))  # skew perturbation, same symmetric part as W
    gW   = Zygote.gradient(W -> sum(abs2, logr(W)), W)[1]
    gWs  = Zygote.gradient(W -> sum(abs2, logr(W)), Wskew)[1]
    @test gW ≈ gWs

    # (3) learnable exponent: both ΔW (symmetric) and Δt must flow back
    t = 0.7
    gW, gt = Zygote.gradient((W, t) -> sum(abs2, powerr_t(W, t)), W, t)
    @test gW ≈ gW'
    @test gt isa Real && isfinite(gt)

    # same gradients through an Hermitian variable directly (typical NN usage)
    gWH, gtH = Zygote.gradient((H, t) -> sum(abs2, powerr_t(H, t)), Hermitian(W), t)
    @test Matrix(gWH) ≈ gW
    @test gtH ≈ gt

    # (4) finite-difference check of the plain-Matrix gradient against the actual loss
    loss = W -> sum(abs2, logr(W))
    gW = Zygote.gradient(loss, W)[1]
    for _ in 1 : 3
        Hdir = Hermitian(randn(n, n) + n * I)  # symmetric SPD-ish direction
        epsh = 1e-6
        dd_fd = (loss(W .+ epsh .* Hdir) - loss(W .- epsh .* Hdir)) / (2epsh)
        dd_ad = sum(gW .* Hdir)
        @test dd_ad ≈ dd_fd rtol=1e-5 atol=1e-7
    end
end;

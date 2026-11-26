# ========================================================================================================
#   Tests of bw_EVD.jl
# ========================================================================================================


println("\nTesting bw_GEVD.jl...")

n = 4;
P = randP(n)+0.01*I;
Q = randP(n)+0.01*I;
S = randP(n)+0.01*I;  # tangent-vector-like SPD input for transportr

#=
 NB: the wrappers such as geodesicr cannot be tested using ChainRulesTestUtils
 because it does not know that these wrappers actually call phir.
 So, we test phir directly.
 The alternatively is to do a manual test, e.g.

using Zygote, FiniteDifferences

f_loss(P, Q, t) = sum(geodesicr(P, Q, t))  # or any scalar wrapper matching your real usage

g_ad = Zygote.gradient(f_loss, P, Q, t)
g_fd = FiniteDifferences.grad(central_fdm(5,1), f_loss, P, Q, t)

@test g_ad[1] ≈ g_fd[1]  atol=1e-6
@test g_ad[2] ≈ g_fd[2]  atol=1e-6
@test g_ad[3] ≈ g_fd[3]  atol=1e-6
=#


# --- phir(P,Q,f) : ϕ(f,P,Q) = P f(P⁻¹Q) ---------------------------------
ChainRulesTestUtils.test_rrule(phir, Matrix(P), Matrix(Q), sqrt);
ChainRulesTestUtils.test_rrule(phir, P, Q, sqrt);
ChainRulesTestUtils.test_rrule(phir, P, Matrix(Q), sqrt); # mixed type
ChainRulesTestUtils.test_rrule(phir, Matrix(P), Q, sqrt); # mixed type

ChainRulesTestUtils.test_rrule(phir, Matrix(P), Matrix(Q), log);
ChainRulesTestUtils.test_rrule(phir, P, Q, log);
ChainRulesTestUtils.test_rrule(phir, P, Matrix(Q), log); # mixed type
ChainRulesTestUtils.test_rrule(phir, Matrix(P), Q, log); # mixed type

# note: positional argument `t` receives NoTangent, so it must be pinned by ⊢
t =0.1
ChainRulesTestUtils.test_rrule(
    phir, Matrix(P), Matrix(Q), Power(t) ⊢ NoTangent();
    fkwargs = (; f′ = Power′(t)) # we even pass explicitly the derivative
    # which otherwise is computed by ForwardDiff in function loewner see SPRmanifolds.jl)
)

# --- geodesicr(P, Q, t::T)
t = 0.1
ChainRulesTestUtils.test_rrule(geodesicr, Matrix(P), Matrix(Q), t ⊢ ChainRulesCore.NoTangent());
ChainRulesTestUtils.test_rrule(geodesicr, P, Q, t ⊢ ChainRulesCore.NoTangent());
ChainRulesTestUtils.test_rrule(geodesicr, P, Matrix(Q), t ⊢ ChainRulesCore.NoTangent()); # mixed type
ChainRulesTestUtils.test_rrule(geodesicr, Matrix(P), Q, t ⊢ ChainRulesCore.NoTangent()); # mixed type

# --- geodesicr_t(P, Q, t::T)
ChainRulesTestUtils.test_rrule(geodesicr_t, Matrix(P), Matrix(Q), 0.1);
ChainRulesTestUtils.test_rrule(geodesicr_t, P, Q, 0.1);
ChainRulesTestUtils.test_rrule(geodesicr_t, P, Matrix(Q), 0.1); # mixed type
ChainRulesTestUtils.test_rrule(geodesicr_t, Matrix(P), Q, 0.1); # mixed type

# --- geomean(P, Q)
ChainRulesTestUtils.test_rrule(geomeanr, Matrix(P), Matrix(Q));
ChainRulesTestUtils.test_rrule(geomeanr, P, Q);
ChainRulesTestUtils.test_rrule(geomeanr, P, Matrix(Q)); # mixed type
ChainRulesTestUtils.test_rrule(geomeanr, Matrix(P), Q); # mixed type

# --- logmap(P, Q)
ChainRulesTestUtils.test_rrule(logmapr, Matrix(P), Matrix(Q));
ChainRulesTestUtils.test_rrule(logmapr, P, Q);
ChainRulesTestUtils.test_rrule(logmapr, P, Matrix(Q)); # mixed type
ChainRulesTestUtils.test_rrule(logmapr, Matrix(P), Q); # mixed type

# --- expmap(P, Q)
ChainRulesTestUtils.test_rrule(expmapr, Matrix(P), Matrix(Q));
ChainRulesTestUtils.test_rrule(expmapr, P, Q);
ChainRulesTestUtils.test_rrule(expmapr, P, Matrix(Q)); # mixed type
ChainRulesTestUtils.test_rrule(expmapr, Matrix(P), Q); # mixed type

# --- expmapr_stable(P, Q)
ChainRulesTestUtils.test_rrule(expmapr_stable, Matrix(P), Matrix(Q));
ChainRulesTestUtils.test_rrule(expmapr_stable, P, Q);
ChainRulesTestUtils.test_rrule(expmapr_stable, P, Matrix(Q)); # mixed type
ChainRulesTestUtils.test_rrule(expmapr_stable, Matrix(P), Q); # mixed type


# --- psir(P,Q,f) : ψ(f,P,Q) = B f(Λ) A -----------------------------------
ChainRulesTestUtils.test_rrule(psir, Matrix(P), Matrix(Q), sqrt);
ChainRulesTestUtils.test_rrule(psir, P, Q, sqrt);
ChainRulesTestUtils.test_rrule(psir, P, Matrix(Q), sqrt); # mixed type
ChainRulesTestUtils.test_rrule(psir, Matrix(P), Q, sqrt); # mixed type

# --- psir(P,Q,f) with another function -----------------------------------
ChainRulesTestUtils.test_rrule(psir, Matrix(P), Matrix(Q), log);
ChainRulesTestUtils.test_rrule(psir, P, Q, log);
ChainRulesTestUtils.test_rrule(psir, P, Matrix(Q), exp); # mixed type
ChainRulesTestUtils.test_rrule(psir, Matrix(P), Q, exp); # mixed type

# --- transportr(S,P,Q) : parallel transport of S from P to Q ------------
ChainRulesTestUtils.test_rrule(parallel_transportr, Matrix(P), Matrix(Q), Matrix(S));
ChainRulesTestUtils.test_rrule(parallel_transportr, P, Q, S);
ChainRulesTestUtils.test_rrule(parallel_transportr, Matrix(P), Matrix(Q), S); # mixed type
ChainRulesTestUtils.test_rrule(parallel_transportr, Matrix(P), Q, Matrix(S)); # mixed type
ChainRulesTestUtils.test_rrule(parallel_transportr, P, Matrix(Q), Matrix(S)); # mixed type
ChainRulesTestUtils.test_rrule(parallel_transportr, Matrix(P), Q, S); # mixed type
ChainRulesTestUtils.test_rrule(parallel_transportr, P, Matrix(Q), S); # mixed type
ChainRulesTestUtils.test_rrule(parallel_transportr, P, Q, Matrix(S)); # mixed type


# -------------------------------------------------------------------------------------------------------
# Gradients of phir and psir through an AD engine (Zygote), as used inside a neural network layer.
#
# The forward pass symmetrizes plain Matrix inputs (Hermitian((X+X')/2)), so the gradient is
# the symmetric Frobenius one for Matrix, Symmetric and Hermitian inputs alike, and does not
# depend on the wrapping type (same convention as the EVD wrappers).
# -------------------------------------------------------------------------------------------------------

@testset "phir/psir gradients through Zygote" begin

    f  = Sqrt()
    f′ = Sqrt′()
    ℓ(A) = sum(abs2, A)

    for (name, fn) in (("phir", phir), ("psir", psir))
        @testset "$name" begin

            lossfn = (P, Q) -> ℓ(fn(P, Q, f; f′ = f′))

            # gradients are symmetric for every input type, and independent of the wrapper type:
            # plain-Matrix gradients equal the wrapped-input (Frobenius) ones
            gPH, gQH = Zygote.gradient(lossfn, P, Q)
            @test Matrix(gPH) ≈ Matrix(gPH)'
            @test Matrix(gQH) ≈ Matrix(gQH)'

            gPM, gQM = Zygote.gradient(lossfn, Matrix(P), Matrix(Q))
            @test gPM ≈ Matrix(gPH)
            @test gQM ≈ Matrix(gQH)
            @test gPM ≈ gPM'
            @test gQM ≈ gQM'

            # a nonsymmetric plain-Matrix input is symmetrized:
            # same gradient as for a skew perturbation of the same symmetric part
            A = randn(n, n)
            Pskew = Matrix(P + 0.1 .* (A .- transpose(A)))
            Qskew = Matrix(Q + 0.1 .* (A .- transpose(A)))
            gPMs, gQMs = Zygote.gradient(lossfn, Pskew, Qskew)
            @test gPMs ≈ gPM
            @test gQMs ≈ gQM

            # finite-difference check of the gradients
            for _ in 1 : 3
                Hp = Hermitian(randn(n, n))
                Hq = Hermitian(randn(n, n))
                e = 1e-6
                dd_fd = (lossfn(P + e*Hp, Q + e*Hq) - lossfn(P - e*Hp, Q - e*Hq)) / (2e)
                dd_ad = sum(Matrix(gPH) .* Hp) + sum(Matrix(gQH) .* Hq)
                @test dd_ad ≈ dd_fd rtol=1e-5 atol=1e-7
            end
        end
    end
end;
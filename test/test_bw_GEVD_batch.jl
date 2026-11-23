println("\nTesting batched GEVD rrules...")

n = 4;
B = 5;

# Random SPD matrix
function randSPD(n)
    return Matrix(randP(n)) + 0.001 * I
end

# Batch of symmetric matrices (random upstream sensitivities)
symbatch(X) = (X .+ permutedims(X, (2, 1, 3))) ./ 2

# Batched SPD matrices
Pbatch = cat([randSPD(n) for _ in 1:B]..., dims = 3);
Qbatch = cat([randSPD(n) for _ in 1:B]..., dims = 3);
Sbatch = cat([randSPD(n) for _ in 1:B]..., dims = 3);


# ------------------------------------------------------------------
# 1. Finite-difference tests
# ------------------------------------------------------------------

# generic ϕ(f,P,Q) = A' f(Λ) A
@testset "phir (batch) rrule" begin
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Log())
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Exp())
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Sqrt())
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Invsqrt())
    # explicit derivative f′
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Log(); fkwargs = (; f′ = Log′()))
end;

# generic ψ(f,P,Q) = B f(Λ) A (non-symmetric output)
@testset "psir (batch) rrule" begin
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Sqrt())
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Log())
    # explicit derivative f′
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Sqrt(); fkwargs = (; f′ = Sqrt′()))
end;

# geodesic with learnable arc-length: gradients for P, Q and t
@testset "geodesicr_t (batch) rrule" begin
    ChainRulesTestUtils.test_rrule(geodesicr_t, Pbatch, Qbatch, 0.5)
end;

# wrappers of the phir rule. For geodesicr the arc-length t is NOT learnable: its tangent is
# NoTangent(), so test_rrule is told not to check it (`t ⊢ NoTangent()`, ⊢ typed \vdash<TAB>)
@testset "geodesicr, geomeanr, logmapr, expmapr (batch) rrules" begin
    ChainRulesTestUtils.test_rrule(geodesicr, Pbatch, Qbatch, 0.5 ⊢ ChainRulesCore.NoTangent())
    ChainRulesTestUtils.test_rrule(geomeanr, Pbatch, Qbatch)
    ChainRulesTestUtils.test_rrule(logmapr,  Pbatch, Qbatch)
    ChainRulesTestUtils.test_rrule(expmapr,  Pbatch, Qbatch)
end;

# parallel transport: gradients for S, P and Q
@testset "parallel_transportr (batch) rrule" begin
    ChainRulesTestUtils.test_rrule(parallel_transportr, Pbatch, Qbatch, Sbatch)
end;


# ------------------------------------------------------------------
# 2. Wiring of the wrappers: same primal, ΔP, ΔQ as the generic phir rule
#    with the wrapper's own f and f′, and the right number of tangents:
#      geodesicr (f,P,Q,t) -> 4;   geomeanr, logmapr, expmapr (f,P,Q) -> 3
# ------------------------------------------------------------------

@testset "wrappers agree with the generic phir rrule" begin

    Ȳ = symbatch(randn(n, n, B))
    t = 0.3

    cases = (
        ("geodesicr", (geodesicr, Pbatch, Qbatch, t), Power(t), Power′(t), 4),
        ("geomeanr",  (geomeanr,  Pbatch, Qbatch),    Sqrt(),  Sqrt′(),   3),
        ("logmapr",   (logmapr,   Pbatch, Qbatch),    Log(),   Log′(),    3),
        ("expmapr",   (expmapr,   Pbatch, Qbatch),    Exp(),   Exp′(),    3),
    )

    for (name, args, f, f′, ntan) in cases
        @testset "$name" begin
            Yw, pbw = ChainRulesCore.rrule(args...)
            Yp, pbp = ChainRulesCore.rrule(phir, Pbatch, Qbatch, f; f′ = f′)
            tw, tp  = pbw(Ȳ), pbp(Ȳ)

            @test Yw ≈ Yp
            @test length(tw) == ntan
            @test tw[2] ≈ tp[2]
            @test tw[3] ≈ tp[3]
            ntan == 4 && @test tw[4] isa ChainRulesCore.NoTangent
        end
    end
end;


# ------------------------------------------------------------------
# 3. Closed-form gradients of the geodesic (Ȳ symmetric)
#      t = 1 :  P (P⁻¹Q) = Q    =>  ΔP = 0,   ΔQ = Ȳ
#      t = 0 :  P (P⁻¹Q)⁰ = P   =>  ΔP = Ȳ,   ΔQ = 0
# ------------------------------------------------------------------

# Gradient convention for plain-Matrix inputs (see desym!): the GEVD reads only the upper triangle,
# so the gradient is upper triangular with doubled off-diagonal entries.
function upper_conv(X)
    Y = zero(X)
    for b in axes(X, 3), j in axes(X, 2), i in 1:j
        Y[i, j, b] = i == j ? X[i, j, b] : 2X[i, j, b]
    end
    return Y
end

@testset "geodesicr: closed-form gradients at t = 1 and t = 0" begin

    Ȳ   = symbatch(randn(n, n, B))
    tol = 1e-7 * norm(Ȳ)

    t = 1.0
    Y, pb = ChainRulesCore.rrule(geodesicr, Pbatch, Qbatch, t)
    _, ΔP, ΔQ, _ = pb(Ȳ)
    @test Y ≈ Qbatch
    @test norm(ΔP) ≤ tol
    @test ΔQ ≈ upper_conv(Ȳ)

    t = 0.0
    Y, pb = ChainRulesCore.rrule(geodesicr, Pbatch, Qbatch, t)
    _, ΔP, ΔQ, _ = pb(Ȳ)
    @test Y ≈ Pbatch
    @test ΔP ≈ upper_conv(Ȳ)
    @test norm(ΔQ) ≤ tol
end;


# ------------------------------------------------------------------
# 4. Batch vs per-slice single-matrix rules (Zygote)
#    Primal and gradient of slice b must equal those of the single-matrix function
#    applied to that slice; the gradient of a loss depending on slice b only must
#    vanish on all the other slices.
# ------------------------------------------------------------------

@testset "batched matches scalar per-slice" begin

    functions = (
        ("phir",        (P, Q) -> phir(P, Q, Sqrt())),
        ("psir",        (P, Q) -> psir(P, Q, Sqrt())),
        ("geodesicr_t", (P, Q) -> geodesicr_t(P, Q, 0.4)),
        ("geodesicr",   (P, Q) -> geodesicr(P, Q, 0.4)),
        ("geomeanr",    geomeanr),
        ("logmapr",     logmapr),
        ("expmapr",     expmapr),
    )

    for (name, g) in functions
        @testset "$name" begin
            Y = g(Pbatch, Qbatch)

            for b in 1:B
                @test Y[:, :, b] ≈ g(Pbatch[:, :, b], Qbatch[:, :, b])

                gb = Zygote.gradient((P, Q) -> sum(g(P, Q)[:, :, b]), Pbatch, Qbatch)
                gs = Zygote.gradient((P, Q) -> sum(g(P, Q)),          Pbatch[:, :, b], Qbatch[:, :, b])

                @test gb[1][:, :, b] ≈ gs[1]
                @test gb[2][:, :, b] ≈ gs[2]

                for c in 1:B
                    c == b && continue
                    @test norm(gb[1][:, :, c]) < 1e-10
                    @test norm(gb[2][:, :, c]) < 1e-10
                end
            end
        end
    end
end;


# ------------------------------------------------------------------
# 5. Batch sizes, including the single-slice case
# ------------------------------------------------------------------

@testset "different batch sizes" begin

    for Btest in (1, 3)

        P = cat([randSPD(n) for _ in 1:Btest]..., dims = 3)
        Q = cat([randSPD(n) for _ in 1:Btest]..., dims = 3)
        S = cat([randSPD(n) for _ in 1:Btest]..., dims = 3)

        @testset "B = $Btest" begin
            ChainRulesTestUtils.test_rrule(phir, P, Q, Log())
            ChainRulesTestUtils.test_rrule(psir, P, Q, Sqrt())
            ChainRulesTestUtils.test_rrule(geodesicr_t, P, Q, 0.25)
            t = 0.25
            ChainRulesTestUtils.test_rrule(geodesicr, P, Q, t ⊢ ChainRulesCore.NoTangent())
            ChainRulesTestUtils.test_rrule(geomeanr, P, Q)
            ChainRulesTestUtils.test_rrule(logmapr, P, Q)
            ChainRulesTestUtils.test_rrule(expmapr, P, Q)
            ChainRulesTestUtils.test_rrule(parallel_transportr, P, Q, S)
        end
    end
end;

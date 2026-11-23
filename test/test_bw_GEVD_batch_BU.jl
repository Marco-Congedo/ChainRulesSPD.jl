# ========================================================================================================
#   Tests of bw_GEVD_batch.jl
# ========================================================================================================

println("\nTesting bw_GEVD.jl...")

n = 4;
B = 5;

# Random SPD matrix
function randSPD(n)
    return Matrix(randP(n))+0.001*I
end

# Batched SPD matrices
Pbatch = cat([randSPD(n) for _ in 1:B]..., dims=3);
Qbatch = cat([randSPD(n) for _ in 1:B]..., dims=3);
Sbatch = cat([randSPD(n) for _ in 1:B]..., dims=3);


# ------------------------------------------------------------------
# Generic batched phi rrule
#
#     phir(P,Q,f)
#
# Tests the generic GEVD spectral function
#     A' f(Λ) A
# ------------------------------------------------------------------

@testset "phir (batch) rrule" begin
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Log())
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Exp())
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Sqrt())
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Invsqrt())
end;

# providing kwarg f′
@testset "phir (batch) rrule (explicit f′)" begin
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Log();   fkwargs=(; f′=Log′()))
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Sqrt();  fkwargs=(; f′=Sqrt′()))
    ChainRulesTestUtils.test_rrule(phir, Pbatch, Qbatch, Exp();   fkwargs=(; f′=Exp′()))
end;

# verify the batch functions agree slice-by-slice with looping the scalar version, 
# both for the primal and for the gradient:
@testset "batched matches scalar per-slice" begin
    Y_batch = psir(Pbatch, Qbatch, Sqrt())
    for b in 1:B
        Y_scalar = psir(Pbatch[:,:,b], Qbatch[:,:,b], Sqrt())
        @test Y_batch[:,:,b] ≈ Y_scalar

        # gradient cross-check
        gb = Zygote.gradient((P,Q) -> sum(psir(P, Q, Sqrt())[:,:,b]), Pbatch, Qbatch)
        gs = Zygote.gradient((P,Q) -> sum(psir(P, Q, Sqrt())), Pbatch[:,:,b], Qbatch[:,:,b])
        @test gb[1][:,:,b] ≈ gs[1]
        @test gb[2][:,:,b] ≈ gs[2]
    end
end;


# ------------------------------------------------------------------
# Geodesic with trainable arc-length parameter
#
#     geodesic_t(P,Q,t)
#
# The rrule must propagate gradients with respect to
#     P, Q and t.
# ------------------------------------------------------------------

@testset "geodesic_t (batch) rrule" begin
    ChainRulesTestUtils.test_rrule(geodesicr_t, Pbatch, Qbatch, 0.5)
    ChainRulesTestUtils.test_rrule(geodesicr_t, Pbatch, Qbatch, 1.0)
end;

# ------------------------------------------------------------------
# Generic batched psi rrule
#
#     psir(P,Q,f)
#
# Tests the non-symmetric GEVD spectral transformation
#     B f(Λ) A
# ------------------------------------------------------------------

@testset "psir (batch) rrule" begin
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Log())
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Exp())
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Sqrt())
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Invsqrt())
end;

# providing kwarg f′
@testset "psir (batch) rrule (explicit f′)" begin
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Log();   fkwargs=(; f′=Log′()))
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Sqrt();  fkwargs=(; f′=Sqrt′()))
    ChainRulesTestUtils.test_rrule(psir, Pbatch, Qbatch, Exp();   fkwargs=(; f′=Exp′()))
end;

# verify the batch functions agree slice-by-slice with looping the scalar version, 
# both for the primal and for the gradient:
@testset "batched matches scalar per-slice" begin
    Y_batch = phir(Pbatch, Qbatch, Sqrt())
    for b in 1:B
        Y_scalar = phir(Pbatch[:,:,b], Qbatch[:,:,b], Sqrt())
        @test Y_batch[:,:,b] ≈ Y_scalar

        # gradient cross-check
        gb = Zygote.gradient((P,Q) -> sum(phir(P, Q, Sqrt())[:,:,b]), Pbatch, Qbatch)
        gs = Zygote.gradient((P,Q) -> sum(phir(P, Q, Sqrt())), Pbatch[:,:,b], Qbatch[:,:,b])
        @test gb[1][:,:,b] ≈ gs[1]
        @test gb[2][:,:,b] ≈ gs[2]
    end
end;



# ------------------------------------------------------------------
# Parallel transport
#     parallel_transport(S,P,Q)
# The rrule must propagate gradients with respect to
#     S, P and Q.
# ------------------------------------------------------------------

@testset "parallel_transport (batch) rrule" begin
    ChainRulesTestUtils.test_rrule(parallel_transportr, Pbatch, Qbatch, Sbatch)
end;


# ------------------------------------------------------------------
# Additional batch-size sanity checks
#
# Verify that all batched rules work for different numbers
# of slices, including the single-slice case.
# ------------------------------------------------------------------

@testset "different batch sizes" begin

    for Btest in (1, 2, 7)

        P = cat([randSPD(n) for _ in 1:Btest]..., dims=3)
        Q = cat([randSPD(n) for _ in 1:Btest]..., dims=3)
        S = cat([randSPD(n) for _ in 1:Btest]..., dims=3)

        @testset "B = $Btest" begin
            ChainRulesTestUtils.test_rrule(phir, P, Q, Log())
            ChainRulesTestUtils.test_rrule(psir, P, Q, Sqrt())
            t = 0.25
            # t ⊢ ChainRulesCore.NoTangent() tell not to test the tangent for t, which is not returned as t is not learnable
            ChainRulesTestUtils.test_rrule(geodesicr_t, P, Q, t)
            ChainRulesTestUtils.test_rrule(geomeanr, P, Q)
            ChainRulesTestUtils.test_rrule(logmapr, P, Q)
            ChainRulesTestUtils.test_rrule(expmapr, P, Q)
            ChainRulesTestUtils.test_rrule(parallel_transportr, P, Q, S)
        end
    end
end;


# ------------------------------------------------------------------
# Explicit output-shape sanity checks
#
# These are not derivative tests; they ensure that every slice
# preserves the expected matrix dimensions.
# ------------------------------------------------------------------

@testset "batched output dimensions" begin

    Y = phir(Pbatch, Qbatch, Log())
    @test size(Y) == (n, n, B)

    Y = psir(Pbatch, Qbatch, Sqrt())
    @test size(Y) == (n, n, B)

    Y = geodesicr_t(Pbatch, Qbatch, 0.5)
    @test size(Y) == (n, n, B)

    Y = geomeanr(Pbatch, Qbatch)
    @test size(Y) == (n, n, B)

   Y = expmapr(Pbatch, Qbatch)
    @test size(Y) == (n, n, B)

   Y = logmapr(Pbatch, Qbatch)
    @test size(Y) == (n, n, B)

    Y = parallel_transportr(Pbatch, Qbatch, Sbatch)
    @test size(Y) == (n, n, B)
end;

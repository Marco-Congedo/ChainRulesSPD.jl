# ========================================================================================================
#   Tests of bw_EVD_batch.jl
# ========================================================================================================

println("\nTesting bw_EVD_batch.jl...")


n = 4;
B = 4;

Xbatch = cat([randP(n)+0.01*I for _ in 1:B]..., dims=3);

# ------------------------------------------------------------------
# Generic spfr rrule
# ------------------------------------------------------------------

@testset "spectral_batch rrule" begin
    ChainRulesTestUtils.test_rrule(spfr, Xbatch, Log())
    ChainRulesTestUtils.test_rrule(spfr, Xbatch, Exp())
    ChainRulesTestUtils.test_rrule(spfr, Xbatch, Sqrt())
end;


# ------------------------------------------------------------------
# Wrapper rules through spfr
# ------------------------------------------------------------------

@testset "batched spectral wrappers" begin
    ChainRulesTestUtils.test_rrule(logr, Xbatch)
    ChainRulesTestUtils.test_rrule(expr, Xbatch)
    ChainRulesTestUtils.test_rrule(sqrtr, Xbatch)
    ChainRulesTestUtils.test_rrule(invsqrtr, Xbatch)
end;


# ------------------------------------------------------------------
# Fixed exponent power
#
# No gradient w.r.t p is expected.
# We do not test because ChainRulesTestUtils
# will check p although the rrule intentionally returns NoTangent.
# ------------------------------------------------------------------

# ChainRulesTestUtils.test_rrule(
#     power_batch,
#     Xbatch,
#     2.0
# )


# ------------------------------------------------------------------
# Trainable exponent power
# ------------------------------------------------------------------

@testset "powerr_t (batch) rrule" begin
    ChainRulesTestUtils.test_rrule(powerr_t, Xbatch, 2.0)
end;


# ------------------------------------------------------------------
# Additional batch size sanity check
# ------------------------------------------------------------------

@testset "different batch sizes" begin
    for Btest in (1,2,7)
        X = cat([randP(n) for _ in 1:Btest]..., dims=3)
        ChainRulesTestUtils.test_rrule(spfr, X, Log())
    end
end;
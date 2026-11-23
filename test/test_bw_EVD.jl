# ========================================================================================================
#   Tests of bp_rules_EVD.jl
# ========================================================================================================

# Test rrules with X either a generic Matrix or an Hermitian Matrix

println("\nTesting bw_EVD.jl...")

n = 4;

# Test for Float64
X = randP(n);

ChainRulesTestUtils.test_rrule(spfr, Matrix(X), sqrt; rtol, atol);
ChainRulesTestUtils.test_rrule(spfr, X, sqrt; rtol, atol);

ChainRulesTestUtils.test_rrule(relur, Matrix(X), 1e-3; rtol, atol);
ChainRulesTestUtils.test_rrule(relur, X, 1e-3; rtol, atol);
ChainRulesTestUtils.test_rrule(logr, Matrix(X); rtol, atol);
ChainRulesTestUtils.test_rrule(logr, X; rtol, atol);
ChainRulesTestUtils.test_rrule(logr_relu, Matrix(X), 0.1; rtol, atol);
ChainRulesTestUtils.test_rrule(logr_relu, X, 0.1; rtol, atol);
ChainRulesTestUtils.test_rrule(expr, Matrix(X); rtol, atol);
ChainRulesTestUtils.test_rrule(expr, X; rtol, atol);
ChainRulesTestUtils.test_rrule(sqrtr, Matrix(X); rtol, atol);
ChainRulesTestUtils.test_rrule(sqrtr, X; rtol, atol);
ChainRulesTestUtils.test_rrule(invsqrtr, Matrix(X); rtol, atol);
ChainRulesTestUtils.test_rrule(invsqrtr, X; rtol, atol);
t = 0.1
# t ⊢ ChainRulesCore.NoTangent() tell not to test the tangent for t, which is not returned as t is not learnable
ChainRulesTestUtils.test_rrule(powerr, Matrix(X), t ⊢ ChainRulesCore.NoTangent())
# Here t is learnable
ChainRulesTestUtils.test_rrule(powerr_t, Matrix(X), t)



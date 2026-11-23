# ========================================================================================================
#   Accuracy tests of the Löwner (Daleckiĭ–Kreĭn) matrix, `loewner(λ, f)`
#
#   METHODOLOGY
#   -----------
#   What is measured
#     The matrix K has entries K[i,j] = (f(λᵢ) - f(λⱼ)) / (λᵢ - λⱼ), K[i,i] = f′(λᵢ).
#     The tests measure the RELATIVE ERROR of one off-diagonal entry, computed by `loewner`
#     on a 2×2 problem, K = loewner([a, b], f)[2,1], against a reference value.
#
#   Reference
#     The reference is the plain textbook formula evaluated in BigFloat (256 bits, ≈ 77 digits)
#     on the EXACT binary values of a and b (`big(a)`, `big(b)` are exact conversions).
#       - input rounding is excluded: only the error of the algorithm under test is measured;
#       - the textbook formula can be used as reference even for near-equal a, b: the pairs
#         below are separated by at least 1e-15 (relative), so the cancellation costs ≤ 16 of the
#         77 digits and the reference keeps > 60 correct digits;
#       - for Power(p), the reference uses big(p), the exact value of the Float64 exponent
#         (not the rational number, e.g. 1/3, that p approximates).
#
#   Test pairs (a, b), drawn at random (seeded, hence reproducible)
#     Half of the pairs are NEAR-EQUAL: b = a(1 ± δ), δ = 10^(-15 u), u ~ U(0,1), so the relative
#     separation spans 1e-15 … 1. This is where the textbook formula suffers catastrophic cancellation
#     and where the special forms must be accurate.
#     The other half are INDEPENDENT draws over a wide dynamic range (up to 28 decades for Log and
#     Sqrt): this is where formulas built on differences/ratios of the arguments can lose accuracy
#     (e.g. an atanh-based Log form, where u = (a-b)/(a+b) → 1). The sign of ± is random, so both
#     orders a>b and a<b are covered. Domains: a,b > 0 for Log, Sqrt, Invsqrt, Power;
#     a,b ∈ (-100,100) for Exp (no overflow, exp(100) ≈ 2.7e43).
#
#   Statistic and pass criterion
#     The test statistic is the WORST relative error over n draws. It must be below a tolerance
#     chosen ≥ 10× above the worst case observed with the intended implementation, and many orders of
#     magnitude below the error of implementations known to be inaccurate. For Float64 (eps ≈ 2.2e-16):
#
#         function            worst error observed*     tolerance
#         log, exp, sqrt, Invsqrt   ≈ 3e-16               1e-14
#         Power(p)                  ≤ 1e-15               1e-13
#
#   Excluded pair: a == b is skipped (the reference is 0/0). That limit, K[i,j] = f′(λ), is tested
#   separately in the last testset.
#
#   Float32: the same procedure applies; scale the tolerances by eps(Float32)/eps(Float64).
# ========================================================================================================

using Test, Random

"""
    check_loewner(f, fbig, gen; n = 5000)

Worst relative error of `loewner([a,b], f)[2,1]` over `n` pairs `(a,b) = gen()`, against
`(fbig(big(a)) - fbig(big(b))) / (big(a) - big(b))`. `fbig` is `f` evaluated in BigFloat.
"""
function check_loewner(f, fbig, gen; n = 5000)
    worst = 0.0
    for _ in 1:n
        a, b = gen()
        a == b && continue
        K    = loewner([a, b], f)[2, 1]
        Kref = (fbig(big(a)) - fbig(big(b))) / (big(a) - big(b))
        worst = max(worst, Float64(abs((K - Kref) / Kref)))
    end
    return worst
end;

# Positive eigenvalues in 10^[lo, hi]; half of the pairs near-equal (relative separation 1e-15…1)
function gen_pos(lo, hi)
    return () -> begin
        a = 10.0^(lo + (hi - lo) * rand())
        b = rand() < 0.5 ? a * (1 + (rand(Bool) ? 1 : -1) * 10.0^(-15 * rand())) :
                           10.0^(lo + (hi - lo) * rand())
        (a, b)
    end
end;

# Eigenvalues of either sign in (-L, L); half of the pairs near-equal
function gen_sym(L)
    return () -> begin
        a = L * (2rand() - 1)
        b = rand() < 0.5 ? a + (rand(Bool) ? 1 : -1) * 10.0^(-15 * rand()) * max(abs(a), 1e-3) :
                           L * (2rand() - 1)
        (a, b)
    end
end;


begin
    checklog = check_loewner(Log(), log, gen_pos(-14, 14));
    println("Relative Error Löwner matrix with f = log: ", checklog)

    checkexp = check_loewner(Exp(), exp, gen_sym(100));
    println("Relative Error Löwner matrix with f = exp: ", checkexp)

    checksqrt = check_loewner(Sqrt(), sqrt, gen_pos(-14, 14));
    println("Relative Error Löwner matrix with f = sqrt: ", checksqrt)

    checkinvsqrt = check_loewner(Invsqrt(), x -> inv(sqrt(x)), gen_pos(-14, 14));
    println("Relative Error Löwner matrix with f = inv(sqrt): ", checkinvsqrt)

    p=0.001;
    checkpower0001 = check_loewner(Power(p), x -> x^big(p), gen_pos(-8, 8));
    println("Relative Error Löwner matrix with f = x^0.001: ", checkpower0001);

    p=0.5;
    checkpower05 = check_loewner(Power(p), x -> x^big(p), gen_pos(-8, 8));
    println("Relative Error Löwner matrix with f = x^0.5: ", checkpower05);

    p=0.999;
    checkpower0999 = check_loewner(Power(p), x -> x^big(p), gen_pos(-8, 8));
    println("Relative Error Löwner matrix with f = x^0.999: ", checkpower0999);
end

@testset "Löwner Matrix" begin

    @testset "Log" begin
        @test checklog < 1e-14
        @test check_loewner(log,   log, gen_pos(-14, 14)) < 1e-14     # Base function wrapper
    end

    @testset "Exp" begin
        @test checkexp < 1e-14
        @test check_loewner(exp,   exp, gen_sym(100)) < 1e-14         # Base function wrapper
    end

    @testset "Sqrt" begin
        @test checksqrt < 1e-14
        @test check_loewner(sqrt,   sqrt, gen_pos(-14, 14)) < 1e-14   # Base function wrapper
    end

    @testset "Invsqrt" begin
        @test checkinvsqrt < 1e-14
    end

    @testset "Power" begin
        for p in (0.5, -0.5, 2.0, 0.1, 0.9, -2.0)
            @test check_loewner(Power(p), x -> x^big(p), gen_pos(-8, 8)) < 1e-13
        end
    end

    # a == b is skipped above (0/0 in the reference): the limit must equal the derivative f′(λ)
    @testset "Limit λᵢ = λⱼ is f′(λ)" begin
        for (f, x) in ((Log(), 0.37), (Exp(), -1.3), (Sqrt(), 0.37),
                       (Invsqrt(), 0.37), (Power(0.5), 0.37), (Power(-2.0), 4.1))
            K = loewner([x, x], f)
            @test K[2, 1] ≈ derivative(f)(x) rtol = 1e-14
            @test K[2, 1] == K[1, 2]                                  # exact symmetry
        end
    end

end;

#=
function check(f, ref, a::T, b::T) where T
    k  = divided_diff(f, a, b)
    kr = T((ref(big(a)) - ref(big(b))) / (big(a) - big(b)))
    abs(k - kr) / abs(kr)
end
for T in (Float32, Float64), δ in T.(10.0 .^ (-12:3:12)), a in T.([1e-3, 1, 50])
    b = a * (1 + δ)
    @test check(Log(), log, a, b) < 10eps(T)
    @test check(Exp(), exp, a, b) < 10eps(T)
    @test divided_diff(Log(), a, b) == divided_diff(Log(), b, a)   # bitwise symmetry
end
=#
# ========================================================================================================
#   Tests of fw_GEVD.jl
# ========================================================================================================


# ========================================================================================================
#   Tests for SPD forward operations based on GEVD
#
#   Tests:
#       - gevdr
#       - phir
#       - common spectral wrappers (geodesicr, logmapr, expmapr)
#       - geodesicr_t
#       - logmapr
#       - expmapr
#       - expmapr_stable
#       - psir
#       - _psirGEVD
#       - parallel_transportr
#
# ========================================================================================================

println("\nTesting fw_GEVD.jl...")


for n in (4, 6)

    P = randP(n);
    Q = randP(n);

    # Dense matrices
    Pm = Matrix(P);
    Qm = Matrix(Q);

    # Symmetric tangent vectors
    S = randn(n, n);
    S = Symmetric((S + S') / 2);
    Sm = Matrix(S);

    # Symmetric/Hermitian objects for PosDefManifold.jl benchmarks.
    Psp = Hermitian(Pm);
    Qsp = Hermitian(Qm);
    Ssp = Hermitian(Sm);

    # ========================================================================================================
    # Generalized eigenvalue decomposition
    # ========================================================================================================

    @testset "gevdr_phir" begin

        λ, A, B = gevdr_phir(Pm, Qm)

        # Basic dimensions.
        @test length(λ) == n
        @test size(A) == (n, n)
        @test size(B) == (n, n)

        # Generalized eigendecomposition identities:
        #
        #     P = A'A
        #     Q = A'ΛA
        #     B = A⁻¹
        #
        @test A' * A ≈ Pm
        @test A' * Diagonal(λ) * A ≈ Qm

        # Generalized eigenvalue equation:
        #
        #     Q B = P B Λ
        #
        @test Qm * B ≈ Pm * B * Diagonal(λ)

        # P-orthonormality of generalized eigenvectors.
        @test B' * Pm * B ≈ I

        # A and B are inverses.
        @test A * B ≈ I
        @test B * A ≈ I

        # Equivalent inverse relation.
        @test B ≈ inv(A)

        # Generalized eigenvalues of two SPD matrices must be positive.
        @test all(>(zero(eltype(λ))), λ)

    end;


    @testset "gevdr_psir" begin

        λ, A, B = gevdr_psir(Pm, Qm)

        # Basic dimensions.
        @test length(λ) == n
        @test size(A) == (n, n)
        @test size(B) == (n, n)

        # Generalized eigendecomposition identities:
        #
        #     P = A'A
        #     Q = A'ΛA
        #     B = A⁻¹
        #
        @test A' * A ≈ Pm
        @test A' * Diagonal(λ) * A ≈ Qm

        # Generalized eigenvalue equation:
        #
        #     Q B = P B Λ
        #
        @test Qm * B ≈ Pm * B * Diagonal(λ)

        # P-orthonormality of generalized eigenvectors.
        @test B' * Pm * B ≈ I

        # A and B are inverses.
        @test A * B ≈ I
        @test B * A ≈ I

        # Equivalent inverse relation.
        @test B ≈ inv(A)

        # Generalized eigenvalues of two SPD matrices must be positive.
        @test all(>(zero(eltype(λ))), λ)

    end;


    # ========================================================================================================
    # Generic generalized spectral function
    # ========================================================================================================

    @testset "phir" begin

        # Use a positive-valued spectral function so that the result is SPD.
        Y = phir(Pm, Qm, sqrt)

        # phir explicitly symmetrizes its result.
        @test Y ≈ Y'

        # Direct GEVD reconstruction.
        λ, A, B = gevdr_phir(Pm, Qm)

        Yref = A' * Diagonal(sqrt.(λ)) * A

        @test Y ≈ Yref

        # Generic spectral function should also reproduce its own GEVD
        # representation for another nontrivial function.
        Ylog = phir(Pm, Qm, log)

        Ylogref = A' * Diagonal(log.(λ)) * A

        @test Ylog ≈ Ylogref
        @test Ylog ≈ Ylog'
    end;

    # ========================================================================================================
    # Auxiliary ψ function
    # ========================================================================================================

    @testset "psir" begin

        Y = psir(Pm, Qm, sqrt)

        @test size(Y) == (n, n)

        λ, A, B, Yinternal = _psirGEVD(Pm, Qm, sqrt)

        @test length(λ) == n
        @test size(A) == (n, n)
        @test size(B) == (n, n)
        @test size(Yinternal) == (n, n)

        # Internal primitive must agree with the public function.
        @test Yinternal ≈ Y

        # Explicit GEVD representation.
        Yref = B * Diagonal(sqrt.(λ)) * A

        @test Y ≈ Yref


        # for an arbitray function
        Y = psir(Pm, Qm, Invsqrt())

        @test size(Y) == (n, n)

        λ, A, B, Yinternal = _psirGEVD(Pm, Qm, Invsqrt())

        @test length(λ) == n
        @test size(A) == (n, n)
        @test size(B) == (n, n)
        @test size(Yinternal) == (n, n)

        # Internal primitive must agree with the public function.
        @test Yinternal ≈ Y

        # Explicit GEVD representation.
        Yref = B * Diagonal(inv.(sqrt.(λ))) * A

        @test Y ≈ Yref

    end;

    # ========================================================================================================
    # Wrappers for common GEVD functions
    # ========================================================================================================

    @testset "GEVD wrappers" begin

        t = 0.5

        # ------------------------------------------------------------------------------------
        # Geodesic
        # ------------------------------------------------------------------------------------

        @test geodesicr(Pm, Qm, t) ≈ phir(Pm, Qm, Power(t))

        # ------------------------------------------------------------------------------------
        # Geodesic with trainable exponent
        # ------------------------------------------------------------------------------------

        @test geodesicr_t(Pm, Qm, t) ≈ phir(Pm, Qm, Power(t))

        # ------------------------------------------------------------------------------------
        # Geometric mean
        # ------------------------------------------------------------------------------------

        @test geomeanr(Pm, Qm) ≈ phir(Pm, Qm, Sqrt())

        # ------------------------------------------------------------------------------------
        # Logarithmic map
        # ------------------------------------------------------------------------------------

        @test logmapr(Pm, Qm) ≈ phir(Pm, Qm, Log())

        # ------------------------------------------------------------------------------------
        # Exponential map
        #
        # Ξ is a tangent vector at P.
        # ------------------------------------------------------------------------------------

        Ξ = logmapr(Pm, Qm)
        @test expmapr(Pm, Ξ) ≈ phir(Pm, Ξ, Exp())
        @test expmapr_stable(Pm, Ξ) ≈ phir(Pm, Ξ, Exp())

    end;

    # ========================================================================================================
    # Basic geometric identities
    # ========================================================================================================

    @testset "GEVD geometric identities" begin

        # ------------------------------------------------------------------------------------
        # End points of the geodesic
        # ------------------------------------------------------------------------------------

        @test geodesicr(Pm, Qm, 0.0) ≈ Pm
        @test geodesicr(Pm, Qm, 1.0) ≈ Qm

        # ------------------------------------------------------------------------------------
        # Geometric mean = midpoint of the geodesic
        # ------------------------------------------------------------------------------------

        @test geomeanr(Pm, Qm) ≈ geodesicr(Pm, Qm, 0.5)

        # ------------------------------------------------------------------------------------
        # Symmetry of the geometric mean
        # ------------------------------------------------------------------------------------

        @test geomeanr(Pm, Qm) ≈  geomeanr(Qm, Pm)

        # ------------------------------------------------------------------------------------
        # Logarithmic map at the same base point
        # ------------------------------------------------------------------------------------

        @test logmapr(Pm, Pm) ≈ zeros(n, n) atol=100 * eps(eltype(Pm))

        # ------------------------------------------------------------------------------------
        # Exponential/logarithmic inverse relation
        # ------------------------------------------------------------------------------------

        Ξ = logmapr(Pm, Qm) 
        @test expmapr(Pm,Ξ) ≈ Qm
        @test expmapr_stable(Pm,Ξ) ≈ Qm

    end;


    # ========================================================================================================
    # Parallel transport
    # ========================================================================================================

    @testset "parallel transport" begin

        T = parallel_transportr(Pm, Qm, Sm)

        # The implementation explicitly symmetrizes the result.
        @test size(T) == (n, n)
        @test T ≈ T'

        # Check the defining transformation:
        #
        #     T = E' S E
        #
        # where E = psir(P,Q, Sqrt()).
        E = psir(Pm, Qm, Sqrt())

        @test T ≈ E' * Sm * E

    end;

    # ========================================================================================================
    # PosDefManifold.jl benchmark comparisons
    # ========================================================================================================

    @testset "PosDefManifold benchmarks" begin

        t = 0.5

        # ------------------------------------------------------------------------------------
        # Geodesic: PosDefManifold uses:geodesic(Fisher, P, Q, t)
        # ------------------------------------------------------------------------------------
        @test geodesicr(Pm, Qm, t) ≈ PosDefManifold.geodesic(Fisher, Psp, Qsp, t)

        # ------------------------------------------------------------------------------------
        # Geometric mean
        #
        # The midpoint of the Fisher geodesic is the geometric mean.
        # geometricMean itself returns (G, iter, conv), so the geodesic
        # midpoint is the cleanest pointwise benchmark here.
        # ------------------------------------------------------------------------------------
        @test geomeanr(Pm, Qm) ≈ PosDefManifold.geodesic(Fisher, Psp, Qsp, 0.5)

        # ------------------------------------------------------------------------------------
        # Logarithmic map: PosDefManifold convention: logmap(Fisher, Q, P)
        # returns Log_P(Q).
        # ------------------------------------------------------------------------------------
        Ξ = logmapr(Pm, Qm)
        @test Ξ ≈ PosDefManifold.logMap(Fisher, Qsp, Psp)


        # ------------------------------------------------------------------------------------
        # Exponential map: PosDefManifold convention: expmap(Fisher, Ξ, P)
        # where Ξ is tangent at P.
        # ------------------------------------------------------------------------------------
        @test expmapr(Pm, Ξ) ≈ PosDefManifold.expMap(Fisher, Hermitian(Ξ), Psp) atol=0
        @test expmapr_stable(Pm, Ξ) ≈ PosDefManifold.expMap(Fisher, Hermitian(Ξ), Psp) atol=0

        # ------------------------------------------------------------------------------------
        # Parallel transport
        # ------------------------------------------------------------------------------------
        @test parallel_transportr(Pm, Qm, Sm) ≈ parallelTransport(Ssp, Psp, Qsp)
    end;
end;

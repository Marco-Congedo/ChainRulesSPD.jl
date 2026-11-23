# ========================================================================================================
#   Tests of fw_EVD_batch.jl
# ========================================================================================================

println("\nTesting fw_EVD_batch.jl...")

n = 4;
B = 5;

Xbatch = cat([randP(n) for _ in 1:B]..., dims=3);

# ------------------------------------------------------------------
# Generic batched spectral primitive
# ------------------------------------------------------------------

@testset "spfr (batch)" begin

    Y = spfr(Xbatch, Log())

    @test size(Y) == size(Xbatch)

    for b in 1:B
        @test Y[:,:,b] ≈ logr(Xbatch[:,:,b])
    end
end;


# ------------------------------------------------------------------
# Internal forward helper
# ------------------------------------------------------------------

@testset "_spfrEVD_batch" begin

    Vs, Λs, Y = _spfrEVD_batch(Xbatch, Sqrt())

    @test size(Y) == size(Xbatch)

    @test length(Vs) == B
    @test length(Λs) == B

    for b in 1:B

        λ, V, Yb = _spfrEVD(Xbatch[:,:,b], Sqrt())

        @test Λs[b] ≈ λ
        @test abs.(Vs[b]) ≈ abs.(V)   # eigenvector signs are arbitrary
        @test Y[:,:,b] ≈ Yb

    end
end;


# ------------------------------------------------------------------
# Standard spectral wrappers. Benchmark against LinearAlgebra.jl
# ------------------------------------------------------------------

@testset "spectral wrappers batch" begin
    @test logr(Xbatch) ≈ cat([logr(Xbatch[:,:,b]) for b in 1:B]..., dims=3) ≈ cat([log(Symmetric(Xbatch[:,:,b])) for b in 1:B]..., dims=3)
    @test expr(Xbatch) ≈ cat([expr(Xbatch[:,:,b]) for b in 1:B]..., dims=3) ≈ cat([exp(Symmetric(Xbatch[:,:,b])) for b in 1:B]..., dims=3)
    @test sqrtr(Xbatch) ≈ cat([sqrtr(Xbatch[:,:,b]) for b in 1:B]..., dims=3) ≈ cat([sqrt(Symmetric(Xbatch[:,:,b])) for b in 1:B]..., dims=3)
    @test invsqrtr(Xbatch) ≈ cat([invsqrtr(Xbatch[:,:,b]) for b in 1:B]..., dims=3) ≈ cat([inv(sqrt(Symmetric(Xbatch[:,:,b]))) for b in 1:B]..., dims=3)
    t = 0.3
    @test powerr(Xbatch, t) ≈ cat([powerr(Xbatch[:,:,b], t) for b in 1:B]..., dims=3) ≈ cat([(Symmetric(Xbatch[:,:,b]))^t for b in 1:B]..., dims=3)
end;


# ------------------------------------------------------------------
# Robust logarithm
# ------------------------------------------------------------------

@testset "robust logarithm batch" begin
    Y = logr_relu(Xbatch, 1e-8)
    @test size(Y) == size(Xbatch)
    for b in 1:B
        @test Y[:,:,b] ≈ logr_relu(Xbatch[:,:,b], 1e-8)
    end
end;



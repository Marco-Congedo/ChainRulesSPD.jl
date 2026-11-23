# ========================================================================================================
#   Tests of fw_GEVD_batch.jl
# ========================================================================================================

println("\nTesting fw_GEVD_batch.jl...")

n = 4;
B = 5;

Pbatch = cat([randP(n) for _ in 1:B]..., dims=3);
Qbatch = cat([randP(n) for _ in 1:B]..., dims=3);
Sbatch = cat([randP(n) for _ in 1:B]..., dims=3);


# ------------------------------------------------------------------
# Generic phi primitive
# ------------------------------------------------------------------

@testset "phir (batch)" begin
    Y = phir(Pbatch, Qbatch, Log())
    @test size(Y) == size(Pbatch)
    for b in 1:B
        @test Y[:,:,b] ≈ phir(Pbatch[:,:,b], Qbatch[:,:,b], Log()) # better than log
    end
end;


# ------------------------------------------------------------------
# Generic psi primitive
# ------------------------------------------------------------------

@testset "psir (batch)" begin
    Y = psir(Pbatch, Qbatch, Sqrt())
    @test size(Y) == size(Pbatch)
    for b in 1:B
        @test Y[:,:,b] ≈ psir(Pbatch[:,:,b], Qbatch[:,:,b], sqrt) # or Sqrt()
    end
end;


# ------------------------------------------------------------------
# Common phi wrappers
# ------------------------------------------------------------------

@testset "phi wrappers batch" begin
    @test geomeanr(Pbatch,Qbatch) ≈
          cat([geomeanr(Pbatch[:,:,b],Qbatch[:,:,b]) for b in 1:B]..., dims=3)

    @test logmapr(Pbatch,Qbatch) ≈
          cat([logmapr(Pbatch[:,:,b],Qbatch[:,:,b]) for b in 1:B]..., dims=3)

    @test expmapr(Pbatch,Qbatch) ≈
          cat([expmapr(Pbatch[:,:,b],Qbatch[:,:,b]) for b in 1:B]..., dims=3)

    t = 0.2
    @test geodesicr(Pbatch,Qbatch,t) ≈
          cat([geodesicr(Pbatch[:,:,b],Qbatch[:,:,b],t) for b in 1:B]..., dims=3)

    @test geodesicr_t(Pbatch,Qbatch,t) ≈
          cat([geodesicr_t(Pbatch[:,:,b],Qbatch[:,:,b],t) for b in 1:B]..., dims=3)

end;


# ------------------------------------------------------------------
# Exponential map
# ------------------------------------------------------------------

@testset "expmap (batch)" begin
    Y = expmapr(Pbatch, Qbatch)
    @test size(Y) == size(Pbatch)
    for b in 1:B
        @test Y[:,:,b] ≈ expmapr(Pbatch[:,:,b], Qbatch[:,:,b])
    end
end;


# ------------------------------------------------------------------
# Parallel transport
# ------------------------------------------------------------------

@testset "parallel_transport (batch)" begin
    Y = parallel_transportr(Pbatch, Qbatch, Sbatch)
    @test size(Y) == size(Sbatch)
    for b in 1:B
        @test Y[:,:,b] ≈ parallel_transportr(Pbatch[:,:,b], Qbatch[:,:,b], Sbatch[:,:,b])
    end
end;
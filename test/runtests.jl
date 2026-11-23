# ========================================================================================================
#   TESTS of all the units
#
#   Copyright Marco Congedo, DR, PhD, CNRS
# ========================================================================================================

using LinearAlgebra
using PosDefManifold
using ChainRulesCore
using ChainRulesTestUtils # standard framework for testing BP rules
using Zygote
using Test
using Random
using ChainRulesSPD
using ChainRulesSPD: _spfrEVD
using ChainRulesSPD: _psirGEVD
using ChainRulesSPD: _spfrEVD_batch

Random.seed!(1234);

rtol=1e-8
atol=1e-8

include("test_loewner.jl")
include("test_frechet.jl")
include("test_fw_EVD.jl")
include("test_bw_EVD.jl")
include("test_fw_GEVD.jl")
include("test_bw_GEVD.jl")
include("test_fw_EVD_batch.jl")
include("test_bw_EVD_batch.jl")
include("test_fw_GEVD_batch.jl")
include("test_bw_GEVD_batch.jl")
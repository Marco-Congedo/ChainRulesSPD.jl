module ChainRulesSPD

using LinearAlgebra
using ForwardDiff
using ChainRulesCore

import ChainRulesCore.rrule

export

# from tools.jl
sym,
skew,

# from pFunctions.jl
derivative,
ReLU, ReLU′,
Log, Log′,
Log_ReLU, Log′_ReLU,
Exp, Exp′,
Sqrt, Sqrt′,
Invsqrt, Invsqrt′,
Power, Power′,
softplus, softplus′,
Softplus, Softplus′,
softexp, softexp′,
Softexp, Softexp′,

# from Loewner.jl
loewner,

# from frechet.jl
frechet,

# from fw_EVD.jl, fw_EVD_batch.jl
spfr,
relur,
logr,
logr_relu,
expr,
expmapr_stable,
sqrtr,
invsqrtr,
powerr,
powerr_t,

# from bw_EVD.jl, bw_EVD_batch.jl, bw_GEVD.jl, bw_GEVD_batch.jl
rrule,

# from fw_GEVD.jl, fw_GEVD_batch.jl
gevdr,
gevdr_phir,
gevdr_psir,
phir,
geodesicr,
geodesicr_t,
geomeanr,
logmapr,
expmapr,
psir,
parallel_transportr

include("tools.jl")
include("pFunctions.jl")
include("Loewner.jl")
include("frechet.jl")

include("fw_EVD.jl")
include("fw_EVD_batch.jl")
include("fw_GEVD.jl")
include("fw_GEVD_batch.jl")

include("bw_EVD.jl")
include("bw_EVD_batch.jl")
include("bw_GEVD.jl")
include("bw_GEVD_batch.jl")

end
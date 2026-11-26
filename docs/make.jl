# Nota Bene: Run it while in the \docs environment

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))  # local ChainRulesSPD
Pkg.instantiate() 

using Documenter, DocumenterCitations, DocumenterInterLinks, DocumenterTools
## ADD HERE ALL MODULES!
using   ChainRulesSPD 

CI = get(ENV, "CI", "false") == "true"        

makedocs(;
   sitename="ChainRulesSPD",
   authors="Marco Congedo and Contributors",
   format = Documenter.HTML(; prettyurls = CI,),
   modules = [ChainRulesSPD],
   #remotes = nothing, # ELIMINATE for deploying  ! ! ! ! ! ! ! ! ! ! !
 pages = [
        "index.md",
        "⛓️ ChainRulesSPD" => "ChainRulesSPD.md",
        "🗄️ API" => "API.md", 
        "💡 Examples" => "Examples.md",
    ]
)

if CI
    deploydocs(
    # root
    # target = "build", # add this folder to .gitignore!
    repo = "github.com/Marco-Congedo/ChainRulesSPD.jl.git",
    branch = "gh-pages",
    push_preview = false,
    )
else
    include("local_run.jl")  # optional local run
end

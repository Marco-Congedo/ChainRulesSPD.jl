#   This script is not part of the ChainRulesSPD.jl package.
#   It allows to build the package locally from the source code,
#   without actually installing the package.
#   You won't need this script for using the package.
#
#   MIT License
#   Copyright (c) 2026, Marco Congedo, CNRS, Grenobe, France:
#   https://sites.google.com/site/marcocongedo/home
#
#   DIRECTIONS:
#   1) If you have installed the ChainRulesSPD.jl from github or Julia registry, uninstall it.
#   2) Install the Revise Package
#   3) Run this block (With VS code, click anywhere here and hit ALT+Enter)
#
begin
  push!(LOAD_PATH, abspath(@__DIR__, "../..") )
  using Revise, ChainRulesSPD
end

# SPDX-License-Identifier: MIT OR Apache-2.0

using Documenter
using LiquidCortex

DocMeta.setdocmeta!(LiquidCortex, :DocTestSetup, :(using LiquidCortex); recursive=true)

makedocs(;
    sitename="LiquidCortex.jl",
    modules=[LiquidCortex],
    authors="Limen-Neural",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://rmems.github.io/LiquidCortex.jl/stable/",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
    # Public examples allocate CUDA reservoirs; CPU CI cannot execute them.
    doctest=false,
)

deploydocs(;
    repo="github.com/rmems/LiquidCortex.jl.git",
    devbranch="main",
    push_preview=false,
)

using CatBoost
using Documenter

makedocs(; modules=[CatBoost], sitename="CatBoost.jl", authors="Beacon Biosignals, Inc.",
         pages=["Introduction" => "index.md", "Wrapper" => "wrapper.md",
                "MLJ API" => "mlj_api.md"])

deploydocs(; repo="github.com/beacon-biosignals/CatBoost.jl.git", push_preview=true,
           devbranch="main")

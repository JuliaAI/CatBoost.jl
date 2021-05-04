using CatBoost
using Documenter

makedocs(; modules=[CatBoost], sitename="CatBoost.jl", authors="Beacon Biosignals, Inc.",
         pages=["API Documentation" => "index.md"])

deploydocs(; repo="github.com/beacon-biosignals/CatBoost.jl.git", push_preview=true,
           devbranch="main")

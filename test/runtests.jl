using Test, CatBoost, Aqua

EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

for ex in readdir(EXAMPLES_DIR)
    @testset "$ex" begin
        # Just check the examples run, for now.
        include(joinpath(EXAMPLES_DIR, ex))
    end
end

Aqua.test_all(CatBoost; ambiguities=false)

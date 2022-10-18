using Aqua
using CatBoost
using DataFrames
using MLJBase
using PythonCall
using Test

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")

@testset "`to_pandas` and `pandas_to_df`" begin
    df = DataFrame(; floats=0.5:0.5:3.0, ints=1:6)
    pd = CatBoost.to_pandas(df)
    @test pd isa Py
    df2 = pandas_to_df(pd)
    @test df2 == df
end

@testset "CatBoostRegressor" begin
    X = DataFrame(; a=[1, 4, 5, 6], b=[4, 5, 6, 7])
    y = [2, 4, 6, 7]

    # MLJ Interface
    model = CatBoostRegressor()
    mach = machine(model, X, y)
    MLJBase.fit!(mach)
    preds = MLJBase.predict(mach, X)
end

@testset "Examples" begin
    for ex in readdir(EXAMPLES_DIR)
        @testset "$ex" begin
            # Just check the examples run, for now.
            include(joinpath(EXAMPLES_DIR, ex))
        end
    end
end

Aqua.test_all(CatBoost; ambiguities=false)

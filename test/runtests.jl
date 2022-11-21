using Aqua
using CatBoost
using DataFrames
using MLJBase
using PythonCall
using Test

const PYTHON_API_EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples/python_api")
const MLJ_EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples/mlj")

@testset "`to_pandas` and `pandas_to_df`" begin
    df = DataFrame(; floats=0.5:0.5:3.0, ints=1:6)
    pd = CatBoost.to_pandas(df)
    @test pd isa Py
    df2 = pandas_to_df(pd)
    @test df2 == df
end

@testset "Python API Examples" begin
    for ex in readdir(PYTHON_API_EXAMPLES_DIR)
        @testset "$ex" begin
            # Just check the examples run, for now.
            include(joinpath(PYTHON_API_EXAMPLES_DIR, ex))
        end
    end
end

@testset "MLJ Examples" begin
    for ex in readdir(MLJ_EXAMPLES_DIR)
        @testset "$ex" begin
            # Just check the examples run, for now.
            include(joinpath(MLJ_EXAMPLES_DIR, ex))
        end
    end
end

Aqua.test_all(CatBoost; ambiguities=false)

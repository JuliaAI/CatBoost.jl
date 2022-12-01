using Aqua
using CatBoost
using DataFrames
using MLJBase
using MLJTestInterface
using PythonCall
using Test

const PYTHON_API_EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples/python_api")
const MLJ_EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples/mlj")

include("python_api.jl")
include("mlj_interface.jl")

Aqua.test_all(CatBoost; ambiguities=false)

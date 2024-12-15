using Aqua
using DataFrames
using MLJBase
using MLJTestInterface
using MLJTuning
using PythonCall
using StatisticalMeasures
using Statistics
using Test

const WRAPPER_EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples/wrapper")
const MLJ_EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples/mlj")

include("wrapper.jl")
include("mlj_interface.jl")

Aqua.test_all(CatBoost)

module CatBoost

using PythonCall
using DataFrames
using OrderedCollections
using Tables

using MLJModelInterface: MLJModelInterface
const MMI = MLJModelInterface
using MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor, Multiclass
const PKG = "CatBoost"

#####
##### Exports
#####

# Python interface
export catboost
export PyCatBoostRegressor, PyCatBoostClassifier, PyCatBoostClassifier
export fit!, cv, predict, predict_proba
export Pool
export pandas_to_df

# Datasets API.
export load_dataset

#####
##### _init_
#####

const catboost = PythonCall.pynew()
const catboost_datasets = PythonCall.pynew()
const numpy = PythonCall.pynew()
const pandas = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(catboost, pyimport("catboost"))
    PythonCall.pycopy!(catboost_datasets, pyimport("catboost.datasets"))
    PythonCall.pycopy!(numpy, pyimport("numpy"))

    # supress catboost future warning (Not sure if we want this)
    warnings = PythonCall.pynew()
    PythonCall.pycopy!(warnings, pyimport("warnings"))
    warnings.simplefilter(; action="ignore", category=pybuiltins.FutureWarning)

    return PythonCall.pycopy!(pandas, pyimport("pandas"))
end

#####
##### MLJ
#####
include("wrapper.jl")
include("mlj_interface.jl")
export CatBoostRegressor

end # module

module CatBoost

using PythonCall
using OrderedCollections
using Tables

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
    PythonCall.pycopy!(pandas, pyimport("pandas"))

    # supress catboost future warning (Not sure if we want this)
    warnings = PythonCall.pynew()
    PythonCall.pycopy!(warnings, pyimport("warnings"))
    warnings.simplefilter(; action="ignore", category=pybuiltins.FutureWarning)

    return nothing
end

#####
##### Python Interface
#####
include("wrapper.jl")

export catboost
export CatBoostRegressor, CatBoostClassifier
export fit!, cv, predict, predict_proba
export Pool
export pandas_to_tbl
# Datasets API
export load_dataset

#####
##### MLJ
#####
include("MLJCatBoostInterface.jl")

end # module

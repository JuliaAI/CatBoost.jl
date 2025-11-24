module CatBoost

using OrderedCollections: OrderedCollections
using PythonCall: PythonCall, Py, PyTable, pybuiltins, pyconvert, pyimport,
                  pytable
using Tables: Tables

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

    @doc """
        cv(pool::Py; kwargs...) -> Table

    Accepts a [`CatBoost.Pool`](@ref) positional argument to specify the training data,
    and keyword arguments to configure the settings. See the python documentation below
    for what keyword arguments are accepted.

    ---

    ## Python documentation for `catboost.cv`

    $(@doc catboost.cv)
    """
    cv

    @doc """
        Pool(data; label=nothing, cat_features=nothing, text_features=nothing,
                pairs=nothing, delimiter='\t', has_header=false, weight=nothing,
                group_id = nothing, group_weight=nothing, subgroup_id=nothing,
                pairs_weight=nothing, baseline=nothing, features_names=nothing,
                thread_count = -1) -> Py

    Creates a `Pool` object holding training data and labels. `data` may also be passed
    as a keyword argument.

    ---

    ## Python documentation for `catboost.Pool`

    $(@doc catboost.Pool)
    """
    Pool

    @doc """
        CatBoostClassifier(args...; kwargs...) -> Py

    Creates a `CatBoostClassifier` object.

    ---

    ## Python documentation for `catboost.CatBoostClassifier`

    $(@doc catboost.CatBoostClassifier)
    """
    CatBoostClassifier

    @doc """
        CatBoostRegressor(args...; kwargs...) -> Py

    Creates a `CatBoostRegressor` object.

    ---

    ## Python documentation for `catboost.CatBoostRegressor`

    $(@doc catboost.CatBoostRegressor)
    """
    CatBoostRegressor

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

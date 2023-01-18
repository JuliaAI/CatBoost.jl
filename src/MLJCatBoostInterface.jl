
module MLJCatBoostInterface

using ..CatBoost: catboost, numpy, to_pandas, feature_importances, predict, predict_proba
using PythonCall
using Tables

using MLJModelInterface: MLJModelInterface
const MMI = MLJModelInterface
using MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor, Multiclass
const PKG = "CatBoost"

"""
    mlj_to_kwargs(model)

Convert MLJ model struct to a dict of kwargs

Parameters
----------
- `model`: MLJ Model

Returns
-------
- `Dict{Symbol, Any}`:Dictionary of kwargs
"""
function mlj_to_kwargs(model)
    return Dict{Symbol,Any}(name => getfield(model, name)
                            for name in fieldnames(typeof(model)))
end

"""
    get_dtype_feature_ix(X, dtype)

Get the index of the columns with a specific dtype

Parameters
----------
- `X`: Table
- `dtype`: DataType 

Returns
-------
- `Vector{Int64}`
"""
function get_dtype_feature_ix(X, dtype)
    return findall(MMI.schema(X).scitypes .<: dtype)
end

"""
Get cat features for model
get_cat_features
"""
function prepare_input(X)
    table_input = Tables.columntable(X)
    columns = Tables.columnnames(table_input)

    order_factor_ix = get_dtype_feature_ix(table_input, OrderedFactor)
    new_columns = Dict([col => MMI.int(table_input[col]) for col in
                                                             columns[order_factor_ix]]...)
    keep_columns = [i for i in columns if all(i .!= keys(new_columns))]
    table_input = (; table_input[keep_columns]..., new_columns...)

    cat_features = get_dtype_feature_ix(table_input, Multiclass) .- 1 # convert to 0 based indexing
    text_features = get_dtype_feature_ix(table_input, MMI.Textual) .- 1 # convert to 0 based indexing

    return table_input, cat_features, text_features
end

include("mlj_catboostclassifier.jl")
include("mlj_catboostregressor.jl")

const CATBOOST_MODELS = Union{CatBoostClassifier,CatBoostRegressor}

include("mlj_serialization.jl")
include("mlj_docstrings.jl")

function MMI.feature_importances(m::CATBOOST_MODELS, fitresult, report)
    return report.feature_importances
end

MMI.metadata_pkg.((CatBoostClassifier, CatBoostRegressor), name="CatBoost.jl",
                  package_uuid="e2e10f9a-a85d-4fa9-b6b2-639a32100a12",
                  package_url="https://github.com/beacon-biosignals/CatBoost.jl",
                  is_pure_julia=false, package_license="MIT")

MMI.metadata_model(CatBoostClassifier;
                   input_scitype=Union{MMI.Table(MMI.Continuous, MMI.Count,
                                                 MMI.OrderedFactor, MMI.Multiclass),
                                       AbstractMatrix{MMI.Continuous}},
                   target_scitype=Union{AbstractVector{<:MMI.Finite}},
                   human_name="CatBoost classifier",
                   load_path="$PKG.MLJCatBoostInterface.CatBoostClassifier")

MMI.metadata_model(CatBoostRegressor;
                   input_scitype=Union{MMI.Table(MMI.Continuous, MMI.Count,
                                                 MMI.OrderedFactor, MMI.Multiclass),
                                       AbstractMatrix{MMI.Continuous}},
                   target_scitype=AbstractVector{<:MMI.Continuous},
                   human_name="CatBoost regressor",
                   load_path="$PKG.MLJCatBoostInterface.CatBoostRegressor")

export CatBoostClassifier, CatBoostRegressor

end

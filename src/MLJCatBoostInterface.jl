
module MLJCatBoostInterface

using ..CatBoost: catboost, numpy, to_pandas, feature_importance, predict, predict_proba,
                  Pool
using PythonCall
using Tables

using MLJModelInterface: MLJModelInterface
const MMI = MLJModelInterface
using MLJModelInterface: Table, Continuous, Count, Finite, OrderedFactor, Multiclass
using CategoricalArrays: CategoricalArray, CategoricalValue, unwrap
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

function drop_cols(a::NamedTuple{an}, cols::Tuple) where {an}
    names = Base.diff_names(an, cols)
    return NamedTuple{names}(a)
end

#####
##### Convert data to CatBoost pools
#####

function prepare_input(X, y)
    table_input = Tables.columntable(X)
    columns = Tables.columnnames(table_input)

    order_factor_cols = columns[get_dtype_feature_ix(table_input, OrderedFactor)]
    new_columns = NamedTuple{order_factor_cols}([MMI.int(table_input[col])
                                                 for col in order_factor_cols])
    table_input = (; drop_cols(table_input, order_factor_cols)..., new_columns...)

    cat_features = get_dtype_feature_ix(table_input, Multiclass) .- 1 # convert to 0 based indexing
    text_features = get_dtype_feature_ix(table_input, MMI.Textual) .- 1 # convert to 0 based indexing
    data_pool = Pool(table_input; label=numpy.array(Array(y)), cat_features, text_features)

    return data_pool
end

function prepare_input(X)
    table_input = Tables.columntable(X)
    columns = Tables.columnnames(table_input)

    order_factor_cols = columns[get_dtype_feature_ix(table_input, OrderedFactor)]
    new_columns = NamedTuple{order_factor_cols}([MMI.int(table_input[col])
                                                 for col in order_factor_cols])
    table_input = (; drop_cols(table_input, order_factor_cols)..., new_columns...)

    cat_features = get_dtype_feature_ix(table_input, Multiclass) .- 1 # convert to 0 based indexing
    text_features = get_dtype_feature_ix(table_input, MMI.Textual) .- 1 # convert to 0 based indexing

    X_pool = Pool(table_input; cat_features, text_features)

    return X_pool
end

#####
##### CatBoost Models
#####

include("mlj_catboostclassifier.jl")
include("mlj_catboostregressor.jl")

const CatBoostModels = Union{CatBoostClassifier,CatBoostRegressor}

#####
##### Reformat data for MLJ
#####

function MMI.reformat(::CatBoostRegressor, X, y)
    data_pool = prepare_input(X, y)
    return (data_pool,)
end

function MMI.reformat(::CatBoostClassifier, X, y)
    data_pool = prepare_input(X, y)
    return (data_pool, first(y))
end

function MMI.reformat(::CatBoostModels, X)
    X_pool = prepare_input(X)
    return (X_pool,)
end

function MMI.selectrows(::CatBoostModels, I, data_pool)
    py_I = numpy.array(numpy.array(I .- 1))
    return (data_pool.slice(py_I),)
end

function MMI.selectrows(::CatBoostModels, I::Colon, data_pool)
    return (data_pool,)
end

function MMI.selectrows(::CatBoostClassifier, I, data_pool, y_first)
    py_I = numpy.array(numpy.array(I .- 1))
    return (data_pool.slice(py_I), y_first)
end

function MMI.selectrows(::CatBoostClassifier, I::Colon, data_pool, y_first)
    return (data_pool, y_first)
end

#####
##### Additional MLJ functionality 
#####

function MMI.update(mlj_model::CatBoostModels, verbosity::Integer, fitresult, cache,
                    data_pool)
    if mlj_model.iterations > cache.mlj_model.iterations &&
       MMI.is_same_except(mlj_model, cache.mlj_model, :iterations)
        iterations = mlj_model.iterations - cache.mlj_model.iterations
        verbose = verbosity <= 1 ? false : true
        new_model = model_init(mlj_model; verbose, iterations)
        new_model.fit(data_pool; init_model=fitresult)
        report = (feature_importances=feature_importance(new_model),)
        cache = (; mlj_model=mlj_model)
    else
        new_model, cache, report = MMI.fit(mlj_model, verbosity, data_pool)
    end

    return new_model, cache, report
end

MMI.iteration_parameter(::Type{<:CatBoostModels}) = :iterations

include("mlj_serialization.jl")

#####
##### MLJ docstring, metadata
#####

function MMI.feature_importances(m::CatBoostModels, fitresult, report)
    return feature_importance(fitresult)
end

MMI.metadata_pkg.((CatBoostClassifier, CatBoostRegressor), name="CatBoost",
                  package_uuid="e2e10f9a-a85d-4fa9-b6b2-639a32100a12",
                  package_url="https://github.com/JuliaAI/CatBoost.jl", is_pure_julia=false,
                  package_license="MIT")

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

include("mlj_docstrings.jl")

export CatBoostClassifier, CatBoostRegressor

end

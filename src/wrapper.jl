
#####
##### Data
#####

"""
    Pool(data; label=nothing, cat_features=nothing, text_features=nothing,
         pairs=nothing, delimiter='\t', has_header=false, weight=nothing,
         group_id = nothing, group_weight=nothing, subgroup_id=nothing,
         pairs_weight=nothing, baseline=nothing, features_names=nothing,
         thread_count = -1) -> Py

Creates a `Pool` object holding training data and labels. `data` may also be passed
as a keyword argument.

"""
function Pool(data; kwargs...)
    return catboost.Pool(to_catboost(data);
                         (k => to_catboost(v) for (k, v) in pairs(kwargs))...)
end
Pool(; kwargs...) = catboost.Pool(; (k => to_catboost(v) for (k, v) in pairs(kwargs))...)

#####
##### Python Models
#####

CatBoostRegressor(args...; kwargs...) = catboost.CatBoostRegressor(args...; kwargs...)
CatBoostClassifier(args...; kwargs...) = catboost.CatBoostClassifier(args...; kwargs...)

fit!(cbm::Py, args...; kwargs...) = cbm.fit(all_to_catboost(args)...; kwargs...)

function predict(cbm::Py, args...; kwargs...)
    py_preds = cbm.predict(all_to_catboost(args)...; kwargs...)
    return pyconvert(Array, py_preds)
end

function predict_proba(cbm::Py, args...; kwargs...)
    py_preds = cbm.predict_proba(all_to_catboost(args)...; kwargs...)
    return pyconvert(Array, py_preds)
end

#####
##### Cross validation
#####

cv(pool::Py; kwargs...) = pandas_to_df(catboost.cv(pool; kwargs...))

#####
##### Conversion utilities
#####

"""
    to_catboost(arg)

`to_catboost` is called on each argument passed to [`fit`](@ref), [`predict`](@ref), [`predict_proba`](@ref), and [`cv`](@ref)
to allow customization of the conversion of Julia types to python types. 

By default, `to_catboost` simply checks if the argument satisfies `Tables.istable(arg)`, and if so, it outputs
a corresponding pandas table, and otherwise passes it on.

To customize the conversion for custom types, provide a method for this function.
"""
to_catboost(arg) = Tables.istable(arg) ? to_pandas(arg) : arg

# utility for calling `to_catboost` on each argument of a function
all_to_catboost(args) = (to_catboost(arg) for arg in args)

"""
    to_pandas(X)

Convert a table/array to a pandas dataframe
"""
function to_pandas(tbl)
    return pytable(tbl, :pandas)
end

function to_pandas(X::AbstractArray)
    tbl = DataFrame(X, :auto)
    return to_pandas(tbl)
end

"""
    pandas_to_df(pandas_df::Py)

Convert a pandas dataframe into a DataFrames.jl dataframe
"""
function pandas_to_df(pandas_df::Py)
    df = DataFrame(PyTable(pandas_df))
    return df
end

#####
##### feature importance
#####

"""
    feature_importances(py_model)

Generate a dataframe of feature importances
"""
function feature_importances(py_model)
    py_df_importance = pandas.DataFrame()
    py_df_importance["name"] = py_model.feature_names_
    py_df_importance["importance"] = py_model.feature_importances_
    return pandas_to_df(py_df_importance)
end

#####
##### Datasets API
#####

"""
    load_dataset(dataset_name::Symbol)

Import a catboost dataset
"""
function load_dataset(dataset_name::Symbol)
    train, test = getproperty(catboost_datasets, dataset_name)()
    return train, test
end

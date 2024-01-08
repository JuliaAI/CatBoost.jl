
#####
##### Data
#####

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

cv(pool::Py; kwargs...) = pandas_to_tbl(catboost.cv(pool; kwargs...))

#####
##### Conversion utilities
#####

"""
    CatBoost.to_catboost(arg)

`CatBoost.to_catboost` is called on each argument passed to [`fit`](@ref), [`predict`](@ref), [`predict_proba`](@ref), and [`cv`](@ref)
to allow customization of the conversion of Julia types to python types. 

By default, `to_catboost` simply checks if the argument satisfies `Tables.istable(arg)`, and if so, it outputs
a corresponding pandas table, and otherwise passes it on.

To customize the conversion for custom types, provide a method for this function.
"""
to_catboost(arg) = Tables.istable(arg) ? to_pandas(arg) : arg

# utility for calling `to_catboost` on each argument of a function
all_to_catboost(args) = (to_catboost(arg) for arg in args)

"""
    CatBoost.to_pandas(X)

Convert a table/array to a pandas dataframe
"""
function to_pandas(tbl)
    return pytable(tbl, :pandas)
end

function to_pandas(X::AbstractArray)
    tbl = Tables.columntable(X)
    return to_pandas(tbl)
end

"""
    pandas_to_tbl(pandas_df::Py)

Convert a pandas dataframe into a Tables.jl columntable
"""
function pandas_to_tbl(pandas_df::Py)
    tbl = Tables.columntable(PyTable(pandas_df))
    return tbl
end

#####
##### feature importance
#####

"""
    CatBoost.feature_importance(py_model::Py)

Generate a Vector{Pair{Symbol, Float64}} of feature importances
"""
function feature_importance(py_model::Py)
    py_df_importance = pandas.DataFrame()
    py_df_importance["name"] = py_model.feature_names_
    py_df_importance["importance"] = py_model.feature_importances_
    tbl_importance = pandas_to_tbl(py_df_importance)
    n_features = size(tbl_importance.name, 1)
    feat_importance = [Symbol(tbl_importance.name[i]) => tbl_importance.importance[i]
                       for i in
                           1:n_features]
    return feat_importance
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

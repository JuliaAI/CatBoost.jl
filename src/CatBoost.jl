module CatBoost

using PyCall
using Conda
using DataFrames
using OrderedCollections
using Tables

#####
##### Exports
#####

export catboost

export CatBoostRegressor, CatBoostClassifier
export fit!, cv, predict, predict_proba
export Pool
export pandas_to_df

# Datasets API.
export load_dataset

#####
##### _init_
#####

const catboost = PyNULL()
const catboost_datasets = PyNULL()
const pandas = PyNULL()

function load_python_deps!()
    copy!(catboost, pyimport("catboost"))
    copy!(catboost_datasets, pyimport("catboost.datasets"))
    copy!(pandas, pyimport("pandas"))
    return nothing
end

function __init__()
    try
        load_python_deps!()
    catch ee
        if PyCall.conda
            Conda.pip_interop(true)
            Conda.pip("install", ["catboost", "pandas"])
            load_python_deps!()
        else
            typeof(ee) <: PyCall.PyError || rethrow(ee)
            @warn("""
                 Python Dependencies not installed
                 Please either:
                 - Rebuild PyCall to use Conda, by running in the julia REPL:
                 - `ENV[PYTHON]=""; Pkg.build("PyCall"); Pkg.build("CatBoost")`
                 - Or install the depencences, eg by running pip
                 - `pip install catboost pandas`
                 """)
        end
    end

    @doc """
        cv(pool::PyObject; kwargs...) -> DataFrame

    Accepts a [`CatBoost.Pool`](@ref) positional argument to specify the training data,
    and keyword arguments to configure the settings. See the python documentation below
    for what keyword arguments are accepted.

    ---

    ## Python documentation for `catboost.cv`

    $(@doc catboost.cv)
    """
    cv

    return nothing
end

#####
##### Data
#####

"""
    Pool(data; label=nothing, cat_features=nothing, text_features=nothing,
         pairs=nothing, delimiter='\t', has_header=false, weight=nothing,
         group_id = nothing, group_weight=nothing, subgroup_id=nothing,
         pairs_weight=nothing, baseline=nothing, features_names=nothing,
         thread_count = -1) -> PyObject

Creates a `Pool` object holding training data and labels. `data` may also be passed
as a keyword argument.

"""
function Pool(data; kwargs...)
    return catboost.Pool(to_catboost(data);
                         (k => to_catboost(v) for (k, v) in pairs(kwargs))...)
end
Pool(; kwargs...) = catboost.Pool(; (k => to_catboost(v) for (k, v) in pairs(kwargs))...)

#####
##### Models
#####

CatBoostRegressor(args...; kwargs...) = catboost.CatBoostRegressor(args...; kwargs...)
CatBoostClassifier(args...; kwargs...) = catboost.CatBoostClassifier(args...; kwargs...)

fit!(cbm, args...; kwargs...) = cbm.fit(all_to_catboost(args)...; kwargs...)
predict(cbm, args...; kwargs...) = cbm.predict(all_to_catboost(args)...; kwargs...)
function predict_proba(cbm, args...; kwargs...)
    return cbm.predict_proba(all_to_catboost(args)...; kwargs...)
end

#####
##### Cross validation
#####

cv(pool::PyObject; kwargs...) = pandas_to_df(catboost.cv(pool; kwargs...))

#####
##### Conversion utilities
#####

"""
    to_catboost(arg)

`to_catboost` is called on each argument passed to [`fit`](@ref), [`predict`](@ref), [`predict_proba`](@ref), and [`cv`](@ref)
to allow customization of the conversion of Julia types to python types. If `to_catboost` emits a Julia type, then
PyCall will try to convert it appropriately (automatically).

By default, `to_catboost` simply checks if the argument satisfies `Tables.istable(arg)`, and if so, it outputs
a corresponding pandas table, and otherwise passes it on.

To customize the conversion for custom types, provide a method for this function.
"""
to_catboost(arg) = Tables.istable(arg) ? to_pandas(arg) : arg

# utility for calling `to_catboost` on each argument of a function
all_to_catboost(args) = (to_catboost(arg) for arg in args)

# the Julia-side code does not copy the columns, but the `pandas.DataFrame`
# constructor seems to make a copy here. Maybe that can be avoided?
function to_pandas(tbl)
    # ensure we have a column table
    col_table = Tables.columns(tbl)
    # write it in a way that pandas will understand (after PyCall conversion)
    dict_table = OrderedDict(col => Tables.getcolumn(col_table, col)
                             for col in Tables.columnnames(col_table))
    return pandas.DataFrame(; data=dict_table)
end

function pandas_to_df(pandas_df::PyObject)
    colnames = map(pandas_df.columns) do c
        ret = c isa PyObject ? PyAny(c) : c
        return ret isa Int ? ret + 1 : ret
    end
    df = DataFrame(Any[Array(getproperty(pandas_df, c).values) for c in colnames],
                   map(Symbol, colnames))
    return df
end

#####
##### Datasets API
#####

function load_dataset(dataset_name::Symbol)
    train, test = getproperty(catboost_datasets, dataset_name)()
    return train, test
end

end # module

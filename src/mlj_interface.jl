
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

include("mlj_catboostregressor.jl")

const CATBOOST_MODELS = Union{CatBoostRegressor}

include("mlj_serialization.jl")
include("mlj_docstrings.jl")

function MMI.feature_importances(m::CATBOOST_MODELS, fitresult, report)
    return report.feature_importances
end

MMI.metadata_pkg.((CatBoostRegressor), name="CatBoost.jl",
                  package_uuid="e2e10f9a-a85d-4fa9-b6b2-639a32100a12",
                  package_url="https://github.com/beacon-biosignals/CatBoost.jl",
                  is_pure_julia=false, package_license="MIT")

MMI.metadata_model(CatBoostRegressor; input_scitype=Table(Continuous, Count),
                   target_scitype=AbstractVector{Continuous},
                   human_name="CatBoost regressor", load_path="$PKG.CatBoostRegressor")

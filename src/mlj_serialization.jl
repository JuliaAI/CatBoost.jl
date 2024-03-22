# Taken from https://github.com/JuliaAI/MLJXGBoostInterface.jl
# It is likely also not the optimal method for serializing models, but it works

"""
    _persistent(model::CatBoostModels, fitresult)

Private method.

Return a persistent (ie, Julia-serializable) representation of the
CatBoost.jl model `fitresult`.

Restore the model with [`fitresult`](@ref)
"""
function _persistent(::CatBoostRegressor, fitresult)
    ctb_file, io = mktemp()
    close(io)

    fitresult.save_model(ctb_file)
    persistent_booster = read(ctb_file)
    rm(ctb_file)
    return persistent_booster
end
function _persistent(::CatBoostClassifier, fitresult)
    model, y_first = fitresult
    if model === nothing
        # Case 1: Single unique class
        return (nothing, fitresult.single_class, y_first)
    else
        # Case 2: Multiple unique classes
        ctb_file, io = mktemp()
        close(io)

        model.save_model(ctb_file)
        persistent_booster = read(ctb_file)
        rm(ctb_file)
        return (persistent_booster, y_first)
    end
end

"""
    _booster(persistent)

Private method.

Return the CatBoost.jl model which has `persistent` as its persistent
(Julia-serializable) representation. See [`persistent`](@ref) method.
"""
function _booster(::CatBoostRegressor, persistent)
    ctb_file, io = mktemp()
    write(io, persistent)
    close(io)

    booster = catboost.CatBoostRegressor().load_model(ctb_file)

    rm(ctb_file)

    return booster
end
function _booster(::CatBoostClassifier, persistent)
    ctb_file, io = mktemp()
    write(io, persistent)
    close(io)

    booster = catboost.CatBoostClassifier().load_model(ctb_file)

    rm(ctb_file)

    return booster
end

function MMI.save(model::CatBoostModels, fitresult; kwargs...)
    return _persistent(model, fitresult)
end

function MMI.restore(model::CatBoostRegressor, serializable_fitresult)
    return _booster(model, serializable_fitresult)
end

function MMI.restore(model::CatBoostClassifier, serializable_fitresult)
    if serializable_fitresult[1] === nothing
        # Case 1: Single unique class
        return (model=nothing, single_class=serializable_fitresult[2],
                y_first=serializable_fitresult[3])
    else
        # Case 2: Multiple unique classes
        persistent_booster, y_first = serializable_fitresult
        return (_booster(model, persistent_booster), y_first)
    end
end

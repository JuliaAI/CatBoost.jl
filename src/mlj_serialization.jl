# Taken from https://github.com/JuliaAI/MLJXGBoostInterface.jl
# It is likely also not the optimal method for serializing models, but it works
"""
    _persistent(booster)

Private method.

Return a persistent (ie, Julia-serializable) representation of the
CatBoost.jl model `booster`.

Restore the model with [`booster`](@ref)
"""
function _persistent(booster)
    ctb_file, io = mktemp()
    close(io)

    booster.save_model(ctb_file)
    persistent_booster = read(ctb_file)
    rm(ctb_file)
    return persistent_booster
end

"""
    _booster(persistent)

Private method.

Return the CatBoost.jl model which has `persistent` as its persistent
(Julia-serializable) representation. See [`persistent`](@ref) method.
"""
function _booster(persistent)
    ctb_file, io = mktemp()
    write(io, persistent)
    close(io)

    booster = catboost.CatBoost().load_model(ctb_file)

    rm(ctb_file)

    return booster
end

function MMI.save(::CATBOOST_MODELS, fr; kw...)
    (booster, a_target_element) = fr
    return (_persistent(booster), a_target_element)
end

function MMI.restore(::CATBOOST_MODELS, fr)
    (persistent, a_target_element) = fr
    return (_booster(persistent), a_target_element)
end
